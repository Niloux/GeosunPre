#include "io/las.h"
#include "io/cache.h"
#include "io/roi.h"
#include "io/scene.h"
#include "geometry/model.h"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace geosun {
bool KeepLasPoint(const LasPoint& p) {
  return !(p.classification & 128) && (p.classification & 31) != 7;
}
Eigen::Vector3d LasCoordinates(const LasHeader& h, const LasPoint& p) {
  Eigen::Vector3d point =
      Eigen::Vector3d(p.raw[0], p.raw[1], p.raw[2]).cwiseProduct(h.scale) + h.origin;
  if (h.converted)
    point = h.frame.Apply(point);
  if (!point.allFinite())
    throw std::runtime_error("Nonfinite LAS coordinates");
  return point;
}
namespace {
template <typename T>
T Read(const uint8_t* data, size_t offset) {
  T value;
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}
struct Aggregate {
  uint64_t key, count;
  int64_t xyz[3];
  uint64_t rgb[3];
};
static_assert(sizeof(Aggregate) == 64);
void Add(Aggregate& a, const Aggregate& b) {
  a.count += b.count;
  for (int j = 0; j < 3; ++j) {
    a.xyz[j] += b.xyz[j];
    a.rgb[j] += b.rgb[j];
  }
}
std::vector<Aggregate> Reduce(std::vector<Aggregate> records) {
  std::sort(records.begin(), records.end(),
            [](const auto& a, const auto& b) { return a.key < b.key; });
  size_t out = 0;
  for (size_t i = 0; i < records.size(); ++i) {
    if (out && records[out - 1].key == records[i].key)
      Add(records[out - 1], records[i]);
    else
      records[out++] = records[i];
  }
  records.resize(out);
  return records;
}
unsigned Bucket(uint64_t key) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  return unsigned(key & 127);
}
std::string Header(uint64_t count) {
  return "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(count) +
         "\nproperty float x\nproperty float y\nproperty float z\nproperty "
         "float nx\nproperty float ny\nproperty float nz\nproperty uchar "
         "red\nproperty uchar green\nproperty uchar blue\nend_header\n";
}
std::string Number(unsigned i) {
  std::ostringstream s;
  s << std::setfill('0') << std::setw(3) << i;
  return s.str();
}
void WriteRecords(const fs::path& file, const std::vector<Aggregate>& records) {
  std::ofstream stream(file, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(records.data()),
               std::streamsize(records.size() * sizeof(Aggregate)));
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write voxel run");
}
void MergeBucket(const fs::path& bucket, const fs::path& workspace,
                 const std::function<void(const Aggregate&)>& emit) {
  // Sorting is capped at 128 MiB, even if one disk partition becomes very
  // large.
  constexpr size_t limit = 2 * 1024 * 1024;
  std::ifstream input(bucket, std::ios::binary);
  auto remaining = fs::file_size(bucket);
  if (remaining % sizeof(Aggregate))
    throw std::runtime_error("Truncated voxel partition");
  remaining /= sizeof(Aggregate);
  std::vector<fs::path> runs;
  while (remaining) {
    std::vector<Aggregate> records(size_t(std::min<uintmax_t>(limit, remaining)));
    if (!input.read(reinterpret_cast<char*>(records.data()),
                    std::streamsize(records.size() * sizeof(Aggregate))))
      throw std::runtime_error("Cannot read voxel partition");
    remaining -= records.size();
    // A partition that fits in one sort run is emitted directly.
    if (runs.empty() && !remaining) {
      for (const auto& value : Reduce(std::move(records)))
        emit(value);
      return;
    }
    const auto path = workspace / fs::u8path(bucket.stem().u8string() + "-run-" +
                                             std::to_string(runs.size()) + ".bin");
    WriteRecords(path, Reduce(std::move(records)));
    runs.push_back(path);
  }
  struct Item {
    Aggregate value;
    size_t run;
  };
  auto later = [](const Item& a, const Item& b) { return a.value.key > b.value.key; };
  std::priority_queue<Item, std::vector<Item>, decltype(later)> queue(later);
  std::vector<std::ifstream> streams;
  for (const auto& run : runs) {
    streams.emplace_back(run, std::ios::binary);
    Aggregate value;
    if (streams.back().read(reinterpret_cast<char*>(&value), sizeof(value)))
      queue.push({value, streams.size() - 1});
    else
      throw std::runtime_error("Cannot read voxel sort run");
  }
  Aggregate current{};
  bool have = false;
  while (!queue.empty()) {
    const auto item = queue.top();
    queue.pop();
    if (have && current.key == item.value.key)
      Add(current, item.value);
    else {
      if (have)
        emit(current);
      current = item.value;
      have = true;
    }
    Aggregate next;
    auto& stream = streams[item.run];
    if (stream.read(reinterpret_cast<char*>(&next), sizeof(next)))
      queue.push({next, item.run});
    else if (!stream.eof() || stream.gcount())
      throw std::runtime_error("Truncated voxel sort run");
  }
  if (have)
    emit(current);
}
}  // namespace
LasHeader ReadLasHeader(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  std::array<uint8_t, 227> data{};
  if (!stream.read(reinterpret_cast<char*>(data.data()), data.size()) ||
      std::memcmp(data.data(), "LASF", 4))
    throw std::runtime_error("Expected a complete LAS header");
  if (data[24] != 1 || data[25] != 2 || data[104] > 3)
    throw std::runtime_error("Only uncompressed LAS 1.2 point formats 0-3 are supported");
  LasHeader h;
  h.format = data[104];
  h.record_size = Read<uint16_t>(data.data(), 105);
  h.count = Read<uint32_t>(data.data(), 107);
  h.offset = Read<uint32_t>(data.data(), 96);
  const uint16_t sizes[] = {20, 28, 26, 34};
  for (int i = 0; i < 3; ++i) {
    h.scale[i] = Read<double>(data.data(), 131 + 8 * i);
    h.origin[i] = Read<double>(data.data(), 155 + 8 * i);
    h.upper[i] = Read<double>(data.data(), 179 + 16 * i);
    h.lower[i] = Read<double>(data.data(), 187 + 16 * i);
  }
  const auto header_size = Read<uint16_t>(data.data(), 94);
  if (header_size < 227 || h.offset < header_size || h.record_size < sizes[h.format] || !h.count ||
      uint64_t(h.offset) + uint64_t(h.count) * h.record_size > fs::file_size(path) ||
      !h.scale.allFinite() || h.scale.minCoeff() <= 0 || !h.origin.allFinite() ||
      !h.upper.allFinite() || !h.lower.allFinite() || (h.upper - h.lower).minCoeff() < 0)
    throw std::runtime_error("Invalid or truncated LAS records/header");
  auto vector = [](const Eigen::Vector3d& v) { return std::vector<double>{v.x(), v.y(), v.z()}; };
  const auto vlrs = Read<uint32_t>(data.data(), 100);
  h.metadata = {{"version", "1.2"},
                {"point_format", h.format},
                {"point_count", h.count},
                {"record_size", h.record_size},
                {"point_data_offset", h.offset},
                {"global_encoding", Read<uint16_t>(data.data(), 6)},
                {"vlr_count", vlrs},
                {"scales", vector(h.scale)},
                {"offsets", vector(h.origin)},
                {"bounds_min", vector(h.lower)},
                {"bounds_max", vector(h.upper)},
                {"crs", vlrs ? "not_parsed" : "absent"},
                {"coordinate_units", "unverified; registration thresholds assume metres"}};
  h.file_lower = h.lower;
  h.file_upper = h.upper;
  return h;
}
void ConvertLasFrame(LasHeader& h, const Similarity& frame, const std::string& name) {
  if (h.converted || !frame.translation.allFinite() || std::abs(frame.scale - 1) > .001)
    throw std::runtime_error("Invalid LAS frame conversion");
  h.frame = frame;
  h.converted = true;
  h.lower = Eigen::Vector3d::Constant(INFINITY);
  h.upper = Eigen::Vector3d::Constant(-INFINITY);
  for (int corner = 0; corner < 8; ++corner) {
    Eigen::Vector3d p;
    for (int k = 0; k < 3; ++k)
      p[k] = (corner >> k & 1) ? h.file_upper[k] : h.file_lower[k];
    p = frame.Apply(p);
    h.lower = h.lower.cwiseMin(p);
    h.upper = h.upper.cwiseMax(p);
  }
  auto vector = [](const Eigen::Vector3d& v) { return std::vector<double>{v.x(), v.y(), v.z()}; };
  h.metadata["frame"] = name;
  h.metadata["frame_from_input"] = frame.ToJson();
  h.metadata["frame_bounds_min"] = vector(h.lower);
  h.metadata["frame_bounds_max"] = vector(h.upper);
}
void ScanLas(const fs::path& path, const LasHeader& h,
             const std::function<void(uint64_t, const std::vector<LasPoint>&)>& consume) {
  const auto before = FileMetadata(path);
  std::ifstream stream(path, std::ios::binary);
  stream.seekg(h.offset);
  constexpr uint64_t chunk = 500000;
  std::vector<uint8_t> bytes;
  std::vector<LasPoint> points;
  for (uint64_t start = 0; start < h.count; start += chunk) {
    const size_t count = size_t(std::min<uint64_t>(chunk, h.count - start));
    bytes.resize(count * h.record_size);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
      throw std::runtime_error("LAS changed or was truncated while reading");
    points.resize(count);
    for (size_t i = 0; i < count; ++i) {
      const auto* p = bytes.data() + i * h.record_size;
      for (int j = 0; j < 3; ++j)
        points[i].raw[j] = Read<int32_t>(p, 4 * j);
      points[i].classification = p[15];
      points[i].gps_time = (h.format == 1 || h.format == 3) ? Read<double>(p, 20) : 0.0;
      for (int j = 0; j < 3; ++j)
        points[i].rgb[j] = (h.format == 2 || h.format == 3)
                               ? Read<uint16_t>(p, (h.format == 2 ? 20 : 28) + 2 * j)
                               : 127;
    }
    consume(start, points);
  }
  if (before != FileMetadata(path))
    throw std::runtime_error("LAS changed during reading");
}
std::vector<Eigen::Vector3d> SampleLas(const fs::path& path, const LasHeader& h, Json& metadata,
                                       size_t maximum, const fs::path& cache_path) {
  if (maximum < 1 || maximum > 5000000)
    throw std::runtime_error("Expected 1-5000000 LAS sample points");
  const Json identity = {{"las_file", FileMetadata(path)},
                         {"maximum", maximum},
                         {"converted", h.converted},
                         {"frame", h.converted ? h.frame.ToJson() : Json(nullptr)}};
  if (!cache_path.empty() && fs::is_regular_file(cache_path)) {
    try {
      BinaryReader in(cache_path);
      char magic[4]{};
      in.Bytes(magic, 4);
      if (std::memcmp(magic, "GSLS", 4) == 0 && in.Value<uint32_t>() == 1 &&
          Json::parse(in.String()) == identity) {
        Json cached_meta = Json::parse(in.String());
        const uint64_t count = in.Value<uint64_t>();
        static_assert(sizeof(Eigen::Vector3d) == 3 * sizeof(double));
        std::vector<Eigen::Vector3d> xyz;
        xyz.resize(size_t(count));
        if (count)
          in.Bytes(xyz.data(), size_t(count) * sizeof(Eigen::Vector3d));
        in.End();
        metadata = std::move(cached_meta);
        return xyz;
      }
    } catch (...) {
      // Fall through and rebuild stale or incomplete LAS sample cache.
    }
  }
  const uint64_t step = std::max<uint64_t>(1, (uint64_t(h.count) + maximum - 1) / maximum);
  std::mt19937_64 generator(0);
  std::uniform_int_distribution<uint64_t> random(0, step - 1);
  std::vector<uint64_t> indices;
  for (uint64_t i = 0; i < h.count; i += step)
    indices.push_back(std::min<uint64_t>(i + random(generator), h.count - 1));
  std::vector<Eigen::Vector3d> xyz;
  xyz.reserve(indices.size());
  size_t position = 0, rejected = 0;
  ScanLas(path, h, [&](uint64_t start, const std::vector<LasPoint>& points) {
    while (position < indices.size() && indices[position] < start + points.size()) {
      const auto& point = points.at(size_t(indices[position++] - start));
      if (KeepLasPoint(point))
        xyz.push_back(LasCoordinates(h, point));
      else
        ++rejected;
    }
  });
  if (xyz.size() < 30)
    throw std::runtime_error("Too few finite LAS samples");
  metadata = h.metadata;
  metadata.update({{"sampling", "one random record per ordinal stratum; fixed seed"},
                   {"sample_rng", "C++ mt19937_64"},
                   {"sample_seed", 0},
                   {"record_stride", step},
                   {"sampled_records", indices.size()},
                   {"sampled_points", xyz.size()},
                   {"excluded_withheld_or_noise", rejected},
                   {"chunk_points", 500000}});
  if (!cache_path.empty()) {
    fs::create_directories(cache_path.parent_path());
    BinaryWriter out(cache_path);
    out.Bytes("GSLS", 4);
    out.Value(uint32_t(1));
    out.String(identity.dump());
    out.String(metadata.dump());
    out.Value(uint64_t(xyz.size()));
    if (!xyz.empty())
      out.Bytes(xyz.data(), xyz.size() * sizeof(Eigen::Vector3d));
    out.Commit();
  }
  return xyz;
}
void VerifyInitializationPly(const fs::path& path, uint64_t points) {
  if (!points)
    throw std::runtime_error("Initialization PLY must contain points");
  const auto header = Header(points);
  std::ifstream stream(path, std::ios::binary);
  std::string actual(header.size(), '\0');
  stream.read(actual.data(), actual.size());
  if (actual != header || fs::file_size(path) != header.size() + points * 27)
    throw std::runtime_error("Initialization PLY header or vertex count differs");
  std::vector<uint8_t> vertices(500000 * 27);
  for (uint64_t start = 0; start < points; start += 500000) {
    const size_t count = size_t(std::min<uint64_t>(500000, points - start));
    if (!stream.read(reinterpret_cast<char*>(vertices.data()), std::streamsize(count * 27)))
      throw std::runtime_error("Truncated PLY payload");
    for (size_t i = 0; i < count; ++i)
      for (int axis = 0; axis < 6; ++axis) {
        const float v = Read<float>(vertices.data() + i * 27, axis * 4);
        if (axis < 3 ? !std::isfinite(v) : v != 0)
          throw std::runtime_error("Invalid initialization PLY coordinates or normals");
      }
  }
}
Json ExportLasPly(const fs::path& las, const LasHeader& h, const fs::path& output, double voxel,
                  const Reconstruction* model, const Reconstruction* visual_model,
                  const std::function<fs::path(uint32_t)>& static_mask_path, Json* roi_out,
                  const std::function<uint64_t(uint64_t voxel_total, uint64_t roi_in_total)>&
                      cap_resolver) {
  if (!std::isfinite(voxel) || voxel <= 0 || fs::absolute(las) == fs::absolute(output))
    throw std::runtime_error("Invalid PLY voxel size or output");
  if (fs::exists(output))
    throw std::runtime_error("PLY output already exists");
  fs::create_directories(output.parent_path());
  const auto source = FileMetadata(las);
  // PLY stores float32 XYZ. Keep its rounding within 1 mm of the working-frame coordinates.
  constexpr double float32_limit_m = 16777.216;
  if (std::max(h.lower.cwiseAbs().maxCoeff(), h.upper.cwiseAbs().maxCoeff()) > float32_limit_m)
    throw std::runtime_error("LAS coordinates exceed 16777 m; float32 PLY would lose millimetre "
                             "precision. Use local LAS coordinates");
  auto workspace = output;
  workspace += ".voxel-work";
  if (fs::exists(workspace))
    throw std::runtime_error("Unfinished LAS initialization exists: " + workspace.u8string());
  fs::create_directories(workspace);
  constexpr uint64_t MaxLinearGrid = (uint64_t(1) << 60) - 1;
  std::array<int64_t, 3> lower, shape;
  uint64_t grid = 1;
  for (int j = 0; j < 3; ++j) {
    const double lo = std::floor(h.lower[j] / voxel) - 1, hi = std::floor(h.upper[j] / voxel) + 1;
    if (std::abs(lo) >= double(INT64_MAX) || std::abs(hi) >= double(INT64_MAX) ||
        hi - lo + 1 >= double(INT64_MAX))
      throw std::runtime_error("LAS voxel grid exceeds supported extent");
    lower[j] = int64_t(lo);
    shape[j] = int64_t(hi - lo + 1);
    if (uint64_t(shape[j]) > MaxLinearGrid / grid)
      throw std::runtime_error("LAS voxel grid is too large");
    grid *= uint64_t(shape[j]);
  }
  constexpr double MaxPlyRange = 50.0;
  constexpr uint64_t MaxPlyPoints = 2500000;
  constexpr uint64_t MaxVisualSupplementPoints = 100000;
  constexpr double VisualCoverRadiusM = 0.25;
  constexpr double MaxVisualRangeM = 250.0;
  constexpr int64_t VisualCellLimit = int64_t(1) << 20;
  auto visual_cell_coord = [&](const Eigen::Vector3d& p, double step,
                               std::array<int64_t, 3>& out) -> bool {
    for (int k = 0; k < 3; ++k) {
      const double v = std::floor(p[k] / step);
      if (!std::isfinite(v) || std::abs(v) >= double(VisualCellLimit - 2))
        return false;
      out[k] = int64_t(v);
    }
    return true;
  };
  auto pack_visual_cell = [&](int64_t cx, int64_t cy, int64_t cz) -> uint64_t {
    return (uint64_t(cx + VisualCellLimit) << 42) | (uint64_t(cy + VisualCellLimit) << 21) |
           uint64_t(cz + VisualCellLimit);
  };
  struct VisualCandidate {
    Eigen::Vector3d xyz;
    std::array<uint8_t, 3> color{127, 127, 127};
    uint32_t track_size = 0;
    double error = 0;
  };
  std::vector<VisualCandidate> visual_cands;
  size_t visual_geom_candidates = 0;
  if (visual_model && static_mask_path) {
    const double min_parallax_cos = std::cos(2.0 * Pi / 180.0);
    std::vector<Eigen::Vector3d> cam_centers(visual_model->images.size(), Eigen::Vector3d::Zero());
    for (size_t id = 0; id < visual_model->images.size(); ++id)
      if (visual_model->images[id].registered)
        cam_centers[id] = visual_model->images[id].pose.Center();
    std::vector<uint64_t> geom_ids;
    for (uint64_t pid = 0; pid < visual_model->points.size(); ++pid) {
      const auto& p = visual_model->points[pid];
      if (p.track.size() < 3 || !p.xyz.allFinite() ||
          p.xyz.cwiseAbs().maxCoeff() > float32_limit_m || p.error > 1.5)
        continue;
      double min_dist = INFINITY;
      bool valid_obs = true;
      std::vector<Eigen::Vector3d> dirs;
      dirs.reserve(p.track.size());
      for (const auto& o : p.track) {
        if (o.image >= visual_model->images.size()) {
          valid_obs = false;
          break;
        }
        const auto& im = visual_model->images[o.image];
        if (!im.registered || o.feature >= im.pixels.size()) {
          valid_obs = false;
          break;
        }
        const auto& cam = visual_model->cameras[im.camera];
        const Eigen::Vector2d& px = im.pixels[o.feature];
        if (px.x() < 2.0 || px.y() < 2.0 || px.x() > cam.width - 3.0 ||
            px.y() > cam.height - 3.0) {
          valid_obs = false;
          break;
        }
        const Eigen::Vector3d v = im.pose.Apply(p.xyz);
        if (!v.allFinite() || v.z() <= 0.2) {
          valid_obs = false;
          break;
        }
        min_dist = std::min(min_dist, (p.xyz - cam_centers[o.image]).norm());
        dirs.push_back((p.xyz - cam_centers[o.image]).normalized());
      }
      if (!valid_obs || min_dist < 0.25 || min_dist > MaxVisualRangeM)
        continue;
      bool parallax_ok = false;
      for (size_t i = 0; i < dirs.size() && !parallax_ok; ++i)
        for (size_t j = i + 1; j < dirs.size(); ++j)
          if (dirs[i].dot(dirs[j]) <= min_parallax_cos) {
            parallax_ok = true;
            break;
          }
      if (parallax_ok)
        geom_ids.push_back(pid);
    }
    visual_geom_candidates = geom_ids.size();
    if (!geom_ids.empty()) {
      std::vector<uint32_t> zero_center_hits(geom_ids.size(), 0),
          strict_static_views(geom_ids.size(), 0);
      std::vector<std::vector<std::pair<uint32_t, Eigen::Vector2d>>> by_image(
          visual_model->images.size());
      for (uint32_t gi = 0; gi < uint32_t(geom_ids.size()); ++gi)
        for (const auto& o : visual_model->points[geom_ids[gi]].track)
          by_image[o.image].emplace_back(gi, visual_model->images[o.image].pixels[o.feature]);
      for (uint32_t id = 0; id < uint32_t(by_image.size()); ++id) {
        if (by_image[id].empty())
          continue;
        const auto& cam = visual_model->cameras[visual_model->images[id].camera];
        const cv::Mat mask = ReadImage(static_mask_path(id), cv::IMREAD_GRAYSCALE);
        if (mask.type() != CV_8UC1 || mask.cols != cam.width || mask.rows != cam.height)
          throw std::runtime_error("Static mask dimensions differ during PLY export");
        for (const auto& [gi, px] : by_image[id]) {
          const int ix = int(std::floor(px.x())), iy = int(std::floor(px.y()));
          if (ix < 0 || iy < 0 || ix >= mask.cols || iy >= mask.rows ||
              mask.at<uint8_t>(iy, ix) == 0) {
            ++zero_center_hits[gi];
            continue;
          }
          bool all_static = ix >= 2 && iy >= 2 && ix + 2 < mask.cols && iy + 2 < mask.rows;
          for (int dy = -2; dy <= 2 && all_static; ++dy) {
            const uint8_t* row = mask.ptr<uint8_t>(iy + dy);
            for (int dx = -2; dx <= 2; ++dx)
              if (row[ix + dx] != 255) {
                all_static = false;
                break;
              }
          }
          if (all_static)
            ++strict_static_views[gi];
        }
      }
      for (uint32_t gi = 0; gi < uint32_t(geom_ids.size()); ++gi) {
        const auto& pt = visual_model->points[geom_ids[gi]];
        if (zero_center_hits[gi] == 0 && strict_static_views[gi] >= 3 &&
            uint64_t(strict_static_views[gi]) * 4 >= uint64_t(pt.track.size()) * 3)
          visual_cands.push_back(
              {pt.xyz, pt.color, uint32_t(pt.track.size()), pt.error});
      }
    }
  }
  struct CoverSlot {
    uint64_t key = UINT64_MAX;
    uint32_t begin = 0, count = 0;
  };
  std::vector<CoverSlot> cover_slots;
  std::vector<uint32_t> cover_indices;
  std::vector<uint8_t> visual_covered(visual_cands.size(), 0);
  size_t uncovered_visual = visual_cands.size();
  size_t cover_mask = 0;
  if (!visual_cands.empty()) {
    std::unordered_map<uint64_t, std::vector<uint32_t>> raw_cells;
    for (uint32_t ci = 0; ci < uint32_t(visual_cands.size()); ++ci) {
      std::array<int64_t, 3> c{};
      if (!visual_cell_coord(visual_cands[ci].xyz, VisualCoverRadiusM, c))
        continue;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            raw_cells[pack_visual_cell(c[0] + dx, c[1] + dy, c[2] + dz)].push_back(ci);
    }
    if (!raw_cells.empty()) {
      size_t table_size = 16;
      while (table_size < raw_cells.size() * 2)
        table_size <<= 1;
      cover_slots.resize(table_size);
      cover_mask = table_size - 1;
      for (const auto& [k, vec] : raw_cells) {
        const uint32_t begin = uint32_t(cover_indices.size());
        cover_indices.insert(cover_indices.end(), vec.begin(), vec.end());
        uint64_t hval = k ^ (k >> 33);
        hval *= 0xff51afd7ed558ccdULL;
        hval ^= hval >> 33;
        size_t pos = size_t(hval) & cover_mask;
        while (cover_slots[pos].key != UINT64_MAX)
          pos = (pos + 1) & cover_mask;
        cover_slots[pos] = {k, begin, uint32_t(vec.size())};
      }
    }
  }
  const double cover_r2 = VisualCoverRadiusM * VisualCoverRadiusM;
  struct View {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    uint32_t camera = 0;
  };
  std::vector<View> views;
  Eigen::Vector3d vis_lo = Eigen::Vector3d::Zero(), vis_hi = Eigen::Vector3d::Zero();
  bool vis_empty = false;
  double cell_m = 0.5, half_diag = 0;
  std::array<int64_t, 3> vis_shape{};
  std::vector<uint8_t> coarse;
  if (model) {
    if (!model->Registered())
      throw std::runtime_error("Cannot filter initialization PLY without registered cameras");
    views.reserve(model->Registered());
    Eigen::Vector3d cam_min = Eigen::Vector3d::Constant(INFINITY),
                    cam_max = Eigen::Vector3d::Constant(-INFINITY);
    for (const auto& img : model->images) {
      if (!img.registered)
        continue;
      const Eigen::Vector3d center = img.pose.Center();
      cam_min = cam_min.cwiseMin(center);
      cam_max = cam_max.cwiseMax(center);
      views.push_back({img.pose.Rotation().toRotationMatrix(), img.pose.Translation(), img.camera});
    }
    vis_lo = (cam_min.array() - MaxPlyRange).matrix().cwiseMax(h.lower);
    vis_hi = (cam_max.array() + MaxPlyRange).matrix().cwiseMin(h.upper);
    if ((vis_hi - vis_lo).minCoeff() < 0)
      vis_empty = true;
    else {
      constexpr uint64_t MaxCoarseCells = 16ULL * 1024 * 1024;
      uint64_t vis_cells = 0;
      while (true) {
        uint64_t total = 1;
        bool fits = true;
        for (int j = 0; j < 3; ++j) {
          const int64_t count =
              std::max<int64_t>(1, int64_t(std::ceil((vis_hi[j] - vis_lo[j]) / cell_m)) + 1);
          vis_shape[j] = count;
          if (uint64_t(count) > MaxCoarseCells / total) {
            fits = false;
            break;
          }
          total *= uint64_t(count);
        }
        if (fits) {
          vis_cells = total;
          break;
        }
        cell_m *= 2.0;
      }
      coarse.assign(size_t(vis_cells), uint8_t(0xFF));
      half_diag = 0.5 * std::sqrt(3.0) * cell_m;
    }
  }
  const double half_fov = FieldOfViewDegrees * Pi / 360.0;
  const double sin_half_fov = std::sin(half_fov), cos_half_fov = std::cos(half_fov);
  auto classify = [&](const Eigen::Vector3d& xyz) -> int {
    if (!model)
      return 0;
    if (vis_empty || (xyz.array() < vis_lo.array()).any() ||
        (xyz.array() > vis_hi.array()).any())
      return -1;
    std::array<int64_t, 3> cell{};
    for (int j = 0; j < 3; ++j)
      cell[j] = std::clamp<int64_t>(int64_t(std::floor((xyz[j] - vis_lo[j]) / cell_m)), 0,
                                      vis_shape[j] - 1);
    const size_t idx = size_t((cell[0] * vis_shape[1] + cell[1]) * vis_shape[2] + cell[2]);
    uint8_t state = coarse[idx];
    if (state == 0xFF) {
      const Eigen::Vector3d c_world(vis_lo.x() + (cell[0] + .5) * cell_m,
                                    vis_lo.y() + (cell[1] + .5) * cell_m,
                                    vis_lo.z() + (cell[2] + .5) * cell_m);
      double best_dist = MaxPlyRange + 1.0;
      for (const auto& view : views) {
        const Eigen::Vector3d v = view.rotation * c_world + view.translation;
        const double dist = v.norm();
        const double eff_dist = std::max(0.0, dist - half_diag);
        if (eff_dist > MaxPlyRange || eff_dist >= best_dist)
          continue;
        if (dist > half_diag) {
          const double rho = std::hypot(v.x(), v.y());
          const double theta = std::atan2(rho, v.z());
          const double alpha = std::asin(std::min(1.0, half_diag / dist));
          if (theta - alpha > half_fov)
            continue;
          Eigen::Vector3d probe = v;
          if (theta > half_fov && rho > 1e-12)
            probe = Eigen::Vector3d(v.x() / rho * sin_half_fov, v.y() / rho * sin_half_fov,
                                    cos_half_fov);
          const auto& cam = model->cameras[view.camera];
          const Eigen::Vector2d px = cam.Project(probe);
          const double margin_px =
              std::max(cam.params[0], cam.params[1]) * half_diag / std::max(0.2, v.z());
          if (px.x() + margin_px < 0 || px.x() - margin_px >= cam.width ||
              px.y() + margin_px < 0 || px.y() - margin_px >= cam.height)
            continue;
        }
        best_dist = eff_dist;
        if (best_dist <= 10.0)
          break;
      }
      if (best_dist <= 10.0)
        state = 1;
      else if (best_dist <= 20.0)
        state = 2;
      else if (best_dist <= 40.0)
        state = 3;
      else if (best_dist <= MaxPlyRange)
        state = 4;
      else
        state = 0;
      coarse[idx] = state;
    }
    return state ? int(state) - 1 : -1;
  };
  std::vector<std::ofstream> buckets;
  for (unsigned i = 0; i < 128; ++i)
    buckets.emplace_back(workspace / (Number(i) + ".bin"), std::ios::binary);
  const RoiParams roi_params;
  std::unordered_map<uint64_t, uint32_t> raw_cell_counts;
  uint64_t accepted = 0, rejected = 0, culled = 0, rgb_max = 0, vertices = 0;
  ScanLas(las, h, [&](uint64_t start, const std::vector<LasPoint>& block) {
    std::vector<Aggregate> values;
    values.reserve(block.size());
    for (const auto& p : block) {
      if (!KeepLasPoint(p)) {
        ++rejected;
        continue;
      }
      const auto xyz = LasCoordinates(h, p);
      const int level = classify(xyz);
      if (level < 0) {
        ++culled;
        continue;
      }
      ++accepted;
      if (model && roi_out) {
        int32_t rx = 0, ry = 0;
        if (GridCellCoord2D(xyz.x(), xyz.y(), roi_params.cell_size_m, rx, ry))
          ++raw_cell_counts[PackGridCell2D(rx, ry)];
      }
      if (uncovered_visual && !cover_slots.empty()) {
        std::array<int64_t, 3> vc{};
        if (visual_cell_coord(xyz, VisualCoverRadiusM, vc)) {
          const uint64_t key = pack_visual_cell(vc[0], vc[1], vc[2]);
          uint64_t hval = key ^ (key >> 33);
          hval *= 0xff51afd7ed558ccdULL;
          hval ^= hval >> 33;
          size_t pos = size_t(hval) & cover_mask;
          while (cover_slots[pos].key != UINT64_MAX) {
            if (cover_slots[pos].key == key) {
              auto& slot = cover_slots[pos];
              if (slot.count) {
                bool any_uncovered = false;
                for (uint32_t idx = 0; idx < slot.count; ++idx) {
                  const uint32_t cid = cover_indices[slot.begin + idx];
                  if (visual_covered[cid])
                    continue;
                  if ((visual_cands[cid].xyz - xyz).squaredNorm() <= cover_r2) {
                    visual_covered[cid] = 1;
                    --uncovered_visual;
                  } else {
                    any_uncovered = true;
                  }
                }
                if (!any_uncovered)
                  slot.count = 0;
              }
              break;
            }
            pos = (pos + 1) & cover_mask;
          }
        }
      }
      uint64_t linear_key = 0;
      for (int j = 0; j < 3; ++j) {
        const int64_t cell = int64_t(std::floor(xyz[j] / voxel)) - lower[j];
        if (cell < 0 || cell >= shape[j])
          throw std::runtime_error("LAS coordinates exceed declared bounds");
        linear_key = linear_key * uint64_t(shape[j]) + (uint64_t(cell) >> level);
      }
      Aggregate a{};
      a.key = (uint64_t(level) << 60) | linear_key;
      a.count = 1;
      for (int j = 0; j < 3; ++j) {
        a.xyz[j] = p.raw[j];
        a.rgb[j] = p.rgb[j];
        rgb_max = std::max<uint64_t>(rgb_max, p.rgb[j]);
      }
      values.push_back(a);
    }
    const auto reduced = Reduce(std::move(values));
    std::array<std::vector<Aggregate>, 128> groups;
    for (const auto& value : reduced)
      groups[Bucket(value.key)].push_back(value);
    for (unsigned i = 0; i < 128; ++i) {
      auto& stream = buckets[i];
      stream.write(reinterpret_cast<const char*>(groups[i].data()),
                   std::streamsize(groups[i].size() * sizeof(Aggregate)));
      if (!stream)
        throw std::runtime_error("Cannot write LAS voxel partition");
    }
    if (start == 0 || (start + block.size()) % 10000000 == 0 || start + block.size() == h.count)
      Log("Native LAS initialization " + std::to_string(start + block.size()) + "/" +
          std::to_string(h.count));
  });
  for (auto& stream : buckets) {
    stream.close();
    if (!stream)
      throw std::runtime_error("Incomplete voxel partition write");
  }
  if (!accepted)
    throw std::runtime_error("No usable LAS records for initialization");
  const bool rgb = h.format == 2 || h.format == 3;
  const double color_scale = rgb && rgb_max > 255 ? 255. / 65535. : 1.;
  Eigen::Vector3d minimum = Eigen::Vector3d::Constant(INFINITY),
                  maximum = Eigen::Vector3d::Constant(-INFINITY);
  const auto payload = workspace / "vertices.bin";
  std::ofstream destination(payload, std::ios::binary);
  for (unsigned i = 0; i < 128; ++i)
    MergeBucket(workspace / (Number(i) + ".bin"), workspace, [&](const Aggregate& v) {
      std::array<uint8_t, 27> vertex{};
      Eigen::Vector3d mean;
      for (int j = 0; j < 3; ++j)
        mean[j] = double(v.xyz[j]) / v.count * h.scale[j] + h.origin[j];
      if (h.converted)
        mean = h.frame.Apply(mean);
      for (int j = 0; j < 3; ++j) {
        const float xyz = float(mean[j]);
        std::memcpy(vertex.data() + j * 4, &xyz, 4);
        minimum[j] = std::min(minimum[j], double(xyz));
        maximum[j] = std::max(maximum[j], double(xyz));
        vertex[24 + j] =
            uint8_t(std::clamp(std::nearbyint(double(v.rgb[j]) / v.count * color_scale), 0., 255.));
      }
      destination.write(reinterpret_cast<const char*>(vertex.data()), vertex.size());
      ++vertices;
    });
  destination.close();
  if (!destination)
    throw std::runtime_error("Incomplete PLY vertex write");
  if (source != FileMetadata(las))
    throw std::runtime_error("LAS changed during initialization");
  const uint64_t voxel_vertices = vertices;
  std::optional<RoiPolygon> roi;
  uint64_t roi_in_total = 0;
  uint64_t roi_out_total = 0;
  if (model && roi_out) {
    roi = BuildRoiPolygon2D(raw_cell_counts, *model, roi_params);
    raw_cell_counts.clear();
    std::vector<float> inside_z;
    std::ifstream data(payload, std::ios::binary);
    std::vector<uint8_t> chunk(500000 * 27);
    for (uint64_t start = 0; start < voxel_vertices; start += 500000) {
      const size_t count = size_t(std::min<uint64_t>(500000, voxel_vertices - start));
      if (!data.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(count * 27)))
        throw std::runtime_error("Truncated PLY vertex cache");
      for (size_t k = 0; k < count; ++k) {
        float xyz[3];
        std::memcpy(xyz, chunk.data() + k * 27, sizeof(xyz));
        if (std::isfinite(xyz[2]) &&
            PointInPolygon2D(roi->vertices_xy, double(xyz[0]), double(xyz[1])))
          inside_z.push_back(xyz[2]);
      }
    }
    data.close();
    FinalizeRoiZRange(*roi, inside_z, *model, roi_params);
    for (float z : inside_z) {
      const double zd = double(z);
      if (zd >= roi->z_min && zd <= roi->z_max)
        ++roi_in_total;
    }
    roi_out_total = voxel_vertices - roi_in_total;
    const std::string coordinate_frame =
        h.converted ? h.metadata.at("frame").get<std::string>() : std::string("input_las_xyz");
    *roi_out = RoiToJson(*roi, roi_params, coordinate_frame);
  }
  std::vector<VisualCandidate> visual_out;
  if (!visual_cands.empty()) {
    std::vector<VisualCandidate> uncovered;
    uncovered.reserve(uncovered_visual);
    for (size_t i = 0; i < visual_cands.size(); ++i)
      if (!visual_covered[i])
        uncovered.push_back(visual_cands[i]);
    std::sort(uncovered.begin(), uncovered.end(),
              [](const VisualCandidate& a, const VisualCandidate& b) {
                if (a.track_size != b.track_size)
                  return a.track_size > b.track_size;
                return a.error < b.error;
              });
    const double vis_voxel = std::max(voxel, 0.10);
    std::unordered_set<uint64_t> vis_cells;
    vis_cells.reserve(std::min<size_t>(uncovered.size(), MaxVisualSupplementPoints));
    for (const auto& c : uncovered) {
      std::array<int64_t, 3> vc{};
      if (!visual_cell_coord(c.xyz, vis_voxel, vc))
        continue;
      if (!vis_cells.insert(pack_visual_cell(vc[0], vc[1], vc[2])).second)
        continue;
      visual_out.push_back(c);
      if (visual_out.size() >= MaxVisualSupplementPoints)
        break;
    }
  }
  const uint64_t visual_vertices = visual_out.size();
  uint64_t effective_max_points = MaxPlyPoints;
  if (model && cap_resolver) {
    const uint64_t geom_points = roi ? roi_in_total : voxel_vertices;
    const uint64_t resolved_cap = cap_resolver(voxel_vertices, geom_points);
    effective_max_points = std::clamp<uint64_t>(resolved_cap, 100000ULL, MaxPlyPoints);
  }
  const uint64_t las_cap =
      effective_max_points > visual_vertices ? (effective_max_points - visual_vertices) : 0ULL;
  const uint64_t las_vertices =
      (model && voxel_vertices > las_cap) ? las_cap : voxel_vertices;
  const uint64_t total_vertices = las_vertices + visual_vertices;
  const auto pending = workspace / "points3D.ply";
  std::ofstream final(pending, std::ios::binary);
  final << Header(total_vertices);
  uint64_t roi_las_written = 0;
  uint64_t outside_roi_las_written = 0;
  if (model && voxel_vertices > las_cap) {
    std::ifstream data(payload, std::ios::binary);
    minimum = Eigen::Vector3d::Constant(INFINITY);
    maximum = Eigen::Vector3d::Constant(-INFINITY);
    if (roi) {
      uint64_t out_target = std::min<uint64_t>(roi_out_total, las_cap / 20);
      const uint64_t in_target = std::min<uint64_t>(roi_in_total, las_cap - out_target);
      if (in_target + out_target < las_cap)
        out_target = std::min<uint64_t>(roi_out_total, las_cap - in_target);
      uint64_t in_acc = 0, out_acc = 0;
      std::vector<uint8_t> chunk(500000 * 27);
      for (uint64_t start = 0; start < voxel_vertices; start += 500000) {
        const size_t count = size_t(std::min<uint64_t>(500000, voxel_vertices - start));
        if (!data.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(count * 27)))
          throw std::runtime_error("Truncated PLY vertex cache");
        for (size_t k = 0; k < count; ++k) {
          const uint8_t* v = chunk.data() + k * 27;
          float xyz[3];
          std::memcpy(xyz, v, sizeof(xyz));
          const bool inside = double(xyz[2]) >= roi->z_min && double(xyz[2]) <= roi->z_max &&
                              PointInPolygon2D(roi->vertices_xy, double(xyz[0]), double(xyz[1]));
          bool take = false;
          if (inside) {
            in_acc += in_target;
            if (in_acc >= roi_in_total) {
              in_acc -= roi_in_total;
              take = true;
              ++roi_las_written;
            }
          } else {
            out_acc += out_target;
            if (out_acc >= roi_out_total) {
              out_acc -= roi_out_total;
              take = true;
              ++outside_roi_las_written;
            }
          }
          if (take) {
            for (int j = 0; j < 3; ++j) {
              minimum[j] = std::min(minimum[j], double(xyz[j]));
              maximum[j] = std::max(maximum[j], double(xyz[j]));
            }
            final.write(reinterpret_cast<const char*>(v), 27);
          }
        }
      }
      data.close();
      if (roi_las_written != in_target || outside_roi_las_written != out_target ||
          roi_las_written + outside_roi_las_written != las_cap)
        throw std::runtime_error("Unexpected ROI-prioritized PLY subsample count");
    } else {
      uint64_t acc = 0, emitted = 0;
      std::array<uint8_t, 27> vertex{};
      for (uint64_t i = 0; i < voxel_vertices; ++i) {
        if (!data.read(reinterpret_cast<char*>(vertex.data()), vertex.size()))
          throw std::runtime_error("Truncated PLY vertex cache");
        acc += las_cap;
        if (acc >= voxel_vertices) {
          acc -= voxel_vertices;
          float xyz[3];
          std::memcpy(xyz, vertex.data(), sizeof(xyz));
          for (int j = 0; j < 3; ++j) {
            minimum[j] = std::min(minimum[j], double(xyz[j]));
            maximum[j] = std::max(maximum[j], double(xyz[j]));
          }
          final.write(reinterpret_cast<const char*>(vertex.data()), vertex.size());
          ++emitted;
        }
      }
      data.close();
      if (emitted != las_cap)
        throw std::runtime_error("Unexpected PLY subsample count");
    }
  } else {
    std::ifstream data(payload, std::ios::binary);
    final << data.rdbuf();
    data.close();
    if (roi) {
      roi_las_written = roi_in_total;
      outside_roi_las_written = roi_out_total;
    }
  }
  for (const auto& vc : visual_out) {
    std::array<uint8_t, 27> vertex{};
    for (int j = 0; j < 3; ++j) {
      const float xyz = float(vc.xyz[j]);
      std::memcpy(vertex.data() + j * 4, &xyz, 4);
      minimum[j] = std::min(minimum[j], double(xyz));
      maximum[j] = std::max(maximum[j], double(xyz));
      vertex[24 + j] = vc.color[j];
    }
    final.write(reinterpret_cast<const char*>(vertex.data()), vertex.size());
  }
  vertices = total_vertices;
  final.close();
  if (!final)
    throw std::runtime_error("Incomplete PLY write");
  VerifyInitializationPly(pending, vertices);
  PublishFile(pending, output);
  // Only this invocation's newly created scratch directory is removed after
  // publication.
  if (fs::weakly_canonical(workspace).parent_path() != fs::weakly_canonical(output.parent_path()))
    throw std::runtime_error("Unexpected voxel workspace location");
  RemoveAll(workspace);
  Json result = {{"source", "las"},
                 {"source_las_file", source},
                 {"source_point_count", h.count},
                 {"accepted_records", accepted},
                 {"excluded_withheld_or_noise", rejected},
                 {"points", vertices},
                 {"las_points", las_vertices},
                 {"visual_supplement_points", visual_vertices},
                 {"voxel_size_m", voxel},
                 {"sampling",
                  model ? (visual_model
                               ? "registered camera frustums (<=50 m) distance-tiered voxels "
                                 "(1x/2x/4x/8x at 10/20/40/50 m) plus static multi-view visual "
                                 "anchors beyond LAS cover (>0.25 m, <=250 m), capped at " +
                                     std::to_string(effective_max_points)
                               : "registered camera frustums (<=50 m), distance-tiered voxels "
                                 "(1x/2x/4x/8x at 10/20/40/50 m), capped at " +
                                     std::to_string(effective_max_points))
                        : "all non-withheld, non-class-7 LAS records; one centroid per voxel"},
                 {"color", "voxel mean of LAS RGB; auto-detect 8/16-bit range; gray if absent"},
                 {"normals", "zero"},
                 {"coordinate_frame", h.converted ? h.metadata.at("frame") : Json("input_las_xyz")},
                 {"rgb_encoding",
                  rgb ? (rgb_max > 255 ? "uint16_0_65535" : "uint16_0_255") : "absent_gray_127"},
                 {"bounds_min", {minimum.x(), minimum.y(), minimum.z()}},
                 {"bounds_max", {maximum.x(), maximum.y(), maximum.z()}},
                 {"ply_size_bytes", fs::file_size(output)}};
  if (model) {
    result["culled_out_of_view"] = culled;
    result["voxel_points_before_cap"] = voxel_vertices;
    result["max_range_m"] = MaxPlyRange;
    result["max_points"] = effective_max_points;
  }
  if (roi) {
    result["roi_las_points"] = roi_las_written;
    result["outside_roi_las_points"] = outside_roi_las_written;
    result["roi_voxel_points_before_cap"] = roi_in_total;
  }
  if (visual_model) {
    result["max_visual_supplement_points"] = MaxVisualSupplementPoints;
    result["visual_cover_radius_m"] = VisualCoverRadiusM;
    result["max_visual_range_m"] = MaxVisualRangeM;
    result["visual_geometry_candidates"] = visual_geom_candidates;
    result["visual_unmasked_candidates"] = visual_cands.size();
  }
  return result;
}

}  // namespace geosun

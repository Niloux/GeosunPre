#include "io/las.h"

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

namespace geosun {
bool KeepLasPoint(const LasPoint& p) {
  return !(p.classification & 128) && (p.classification & 31) != 7;
}
Eigen::Vector3d LasCoordinates(const LasHeader& h, const LasPoint& p) {
  const Eigen::Vector3d point =
      Eigen::Vector3d(p.raw[0], p.raw[1], p.raw[2]).cwiseProduct(h.scale) + h.origin;
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
  std::vector<fs::path> runs;
  for (;;) {
    std::vector<Aggregate> records(limit);
    input.read(reinterpret_cast<char*>(records.data()),
               std::streamsize(records.size() * sizeof(Aggregate)));
    const auto bytes = input.gcount();
    if (bytes % sizeof(Aggregate))
      throw std::runtime_error("Truncated voxel partition");
    records.resize(size_t(bytes) / sizeof(Aggregate));
    if (records.empty())
      break;
    const auto path = workspace / fs::u8path(bucket.stem().u8string() + "-run-" +
                                             std::to_string(runs.size()) + ".bin");
    WriteRecords(path, Reduce(std::move(records)));
    runs.push_back(path);
    if (input.eof())
      break;
    if (!input)
      throw std::runtime_error("Cannot read voxel partition");
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
  return h;
}
void ScanLas(const fs::path& path, const LasHeader& h,
             const std::function<void(uint64_t, const std::vector<LasPoint>&)>& consume) {
  const auto before = FileMetadata(path);
  std::ifstream stream(path, std::ios::binary);
  stream.seekg(h.offset);
  constexpr uint64_t chunk = 500000;
  for (uint64_t start = 0; start < h.count; start += chunk) {
    const size_t count = size_t(std::min<uint64_t>(chunk, h.count - start));
    std::vector<uint8_t> bytes(count * h.record_size);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
      throw std::runtime_error("LAS changed or was truncated while reading");
    std::vector<LasPoint> points(count);
    for (size_t i = 0; i < count; ++i) {
      const auto* p = bytes.data() + i * h.record_size;
      for (int j = 0; j < 3; ++j)
        points[i].raw[j] = Read<int32_t>(p, 4 * j);
      points[i].classification = p[15];
      if (h.format == 1 || h.format == 3)
        points[i].gps_time = Read<double>(p, 20);
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
                                       size_t maximum) {
  if (maximum < 1 || maximum > 5000000)
    throw std::runtime_error("Expected 1-5000000 LAS sample points");
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
Json ExportLasPly(const fs::path& las, const fs::path& output, double voxel) {
  if (!std::isfinite(voxel) || voxel <= 0 || fs::absolute(las) == fs::absolute(output))
    throw std::runtime_error("Invalid PLY voxel size or output");
  if (fs::exists(output))
    throw std::runtime_error("PLY output already exists");
  fs::create_directories(output.parent_path());
  const auto source = FileMetadata(las);
  const auto h = ReadLasHeader(las);
  // PLY stores float32 XYZ. Keep its rounding within 1 mm of the input LAS coordinates.
  constexpr double float32_limit_m = 16777.216;
  if (std::max(h.lower.cwiseAbs().maxCoeff(), h.upper.cwiseAbs().maxCoeff()) > float32_limit_m)
    throw std::runtime_error("LAS coordinates exceed 16777 m; float32 PLY would lose millimetre "
                             "precision. Use local LAS coordinates");
  auto workspace = output;
  workspace += ".voxel-work";
  if (fs::exists(workspace))
    throw std::runtime_error("Unfinished LAS initialization exists: " + workspace.u8string());
  fs::create_directories(workspace);
  std::array<int64_t, 3> lower, shape;
  uint64_t grid = 1;
  for (int j = 0; j < 3; ++j) {
    const double lo = std::floor(h.lower[j] / voxel) - 1, hi = std::floor(h.upper[j] / voxel) + 1;
    if (std::abs(lo) >= double(INT64_MAX) || std::abs(hi) >= double(INT64_MAX) ||
        hi - lo + 1 >= double(INT64_MAX))
      throw std::runtime_error("LAS voxel grid exceeds supported extent");
    lower[j] = int64_t(lo);
    shape[j] = int64_t(hi - lo + 1);
    if (uint64_t(shape[j]) > uint64_t(INT64_MAX) / grid)
      throw std::runtime_error("LAS voxel grid is too large");
    grid *= uint64_t(shape[j]);
  }
  std::vector<std::ofstream> buckets;
  for (unsigned i = 0; i < 128; ++i)
    buckets.emplace_back(workspace / (Number(i) + ".bin"), std::ios::binary);
  uint64_t accepted = 0, rejected = 0, rgb_max = 0, vertices = 0;
  ScanLas(las, h, [&](uint64_t start, const std::vector<LasPoint>& block) {
    std::vector<Aggregate> values;
    values.reserve(block.size());
    for (const auto& p : block) {
      if (!KeepLasPoint(p)) {
        ++rejected;
        continue;
      }
      ++accepted;
      const auto xyz = LasCoordinates(h, p);
      uint64_t key = 0;
      for (int j = 0; j < 3; ++j) {
        const int64_t cell = int64_t(std::floor(xyz[j] / voxel)) - lower[j];
        if (cell < 0 || cell >= shape[j])
          throw std::runtime_error("LAS coordinates exceed declared bounds");
        key = key * uint64_t(shape[j]) + uint64_t(cell);
      }
      Aggregate a{};
      a.key = key;
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
      std::cerr << "Native LAS initialization " << start + block.size() << '/' << h.count << '\n';
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
      for (int j = 0; j < 3; ++j) {
        const float xyz = float(double(v.xyz[j]) / v.count * h.scale[j] + h.origin[j]);
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
  const auto pending = workspace / "points3D.ply";
  std::ofstream final(pending, std::ios::binary);
  final << Header(vertices);
  std::ifstream data(payload, std::ios::binary);
  final << data.rdbuf();
  final.close();
  if (!final)
    throw std::runtime_error("Incomplete PLY write");
  VerifyInitializationPly(pending, vertices);
  fs::rename(pending, output);
  // Only this invocation's newly created scratch directory is removed after
  // publication.
  data.close();
  if (fs::weakly_canonical(workspace).parent_path() != fs::weakly_canonical(output.parent_path()))
    throw std::runtime_error("Unexpected voxel workspace location");
  fs::remove_all(workspace);
  return {{"version", 1},
          {"source", "las"},
          {"source_las_file", source},
          {"source_point_count", h.count},
          {"accepted_records", accepted},
          {"excluded_withheld_or_noise", rejected},
          {"points", vertices},
          {"voxel_size_m", voxel},
          {"sampling", "all non-withheld, non-class-7 LAS records; one centroid per voxel"},
          {"color", "voxel mean of LAS RGB; auto-detect 8/16-bit range; gray if absent"},
          {"normals", "zero"},
          {"coordinate_frame", "input_las_xyz"},
          {"rgb_encoding",
           rgb ? (rgb_max > 255 ? "uint16_0_65535" : "uint16_0_255") : "absent_gray_127"},
          {"bounds_min", {minimum.x(), minimum.y(), minimum.z()}},
          {"bounds_max", {maximum.x(), maximum.y(), maximum.z()}},
          {"ply_size_bytes", fs::file_size(output)}};
}

}  // namespace geosun

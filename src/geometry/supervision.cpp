#include "geometry/supervision.h"
#include "geometry/supervision_kernels.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "geometry/monocular.h"
#include "io/las.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>

namespace geosun {
namespace {
constexpr const char* MonocularSource = "las_cleaned_moge2_normals";
using namespace supervision_gpu;
constexpr size_t DepthBudget = 1024 * 1024 * 1024;
constexpr size_t PointChunk = 500000;
// Mask decoding, PNG encoding and read-back run beside the GPU with this many images in flight.
constexpr size_t Lookahead = 6;
// LAS surfels: K nearest neighbours within the hashed-grid reach set the point spacing and the
// Device bytes per point of a normal-estimation slab, all buffers at their worst-case size.
constexpr size_t SlabPointBytes = 192;
constexpr size_t MaxSlabPoints = 64 * 1000 * 1000;
constexpr int SlabBins = 1 << 16;
struct RawBounds {
  int3 min, max;
};
bool Overlaps(const RawBounds& a, const RawBounds& b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
         a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
}
bool Contains(const RawBounds& b, const std::array<int32_t, 3>& raw) {
  return raw[0] >= b.min.x && raw[0] <= b.max.x && raw[1] >= b.min.y && raw[1] <= b.max.y &&
         raw[2] >= b.min.z && raw[2] <= b.max.z;
}
Eigen::Vector3d WorkingToFile(const LasHeader& header, const Eigen::Vector3d& world) {
  if (!header.converted)
    return world;
  return (header.frame.rotation.conjugate() * (world - header.frame.translation)) /
         header.frame.scale;
}
int32_t ClampRawCoord(double value) {
  constexpr double lo = double(std::numeric_limits<int32_t>::lowest());
  constexpr double hi = double(std::numeric_limits<int32_t>::max());
  if (!std::isfinite(value))
    return value < 0 ? std::numeric_limits<int32_t>::lowest()
                     : std::numeric_limits<int32_t>::max();
  return int32_t(std::clamp(value, lo, hi));
}
RawBounds WorldBoxToRawBounds(const LasHeader& header, const Eigen::Vector3d& lo,
                              const Eigen::Vector3d& hi) {
  Eigen::Vector3d file_lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()),
                  file_hi = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  for (int mask = 0; mask < 8; ++mask) {
    const Eigen::Vector3d corner(mask & 1 ? hi.x() : lo.x(), mask & 2 ? hi.y() : lo.y(),
                                 mask & 4 ? hi.z() : lo.z());
    const Eigen::Vector3d f = WorkingToFile(header, corner);
    file_lo = file_lo.cwiseMin(f);
    file_hi = file_hi.cwiseMax(f);
  }
  Eigen::Array3d r_lo, r_hi;
  for (int k = 0; k < 3; ++k) {
    const double a = (file_lo[k] - header.origin[k]) / header.scale[k],
                 b = (file_hi[k] - header.origin[k]) / header.scale[k];
    r_lo[k] = std::floor(std::min(a, b));
    r_hi[k] = std::ceil(std::max(a, b));
  }
  return {make_int3(ClampRawCoord(r_lo[0]), ClampRawCoord(r_lo[1]), ClampRawCoord(r_lo[2])),
          make_int3(ClampRawCoord(r_hi[0]), ClampRawCoord(r_hi[1]), ClampRawCoord(r_hi[2]))};
}
std::optional<RawBounds> ModelReachBounds(const LasHeader& header, const Reconstruction& model,
                                          double reach_m) {
  Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()),
                  hi = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  bool any = false;
  for (const auto& img : model.images)
    if (img.registered) {
      const auto c = img.pose.Center();
      lo = lo.cwiseMin(c);
      hi = hi.cwiseMax(c);
      any = true;
    }
  if (!any)
    return std::nullopt;
  return WorldBoxToRawBounds(header, lo - Eigen::Vector3d::Constant(reach_m),
                             hi + Eigen::Vector3d::Constant(reach_m));
}
RawBounds BatchReachBounds(const LasHeader& header, const Reconstruction& model,
                           const ShutterMotion* motion, const std::vector<uint32_t>& images,
                           size_t first, size_t count, uint32_t cid) {
  Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()),
                  hi = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  double max_shift = 0;
  for (size_t j = 0; j < count; ++j) {
    const uint32_t id = images[first + j];
    const auto c = model.images[id].pose.Center();
    lo = lo.cwiseMin(c);
    hi = hi.cwiseMax(c);
    if (motion)
      max_shift = std::max(max_shift, (motion->readout[cid] * motion->velocity[id]).norm());
  }
  const double reach = MaxDepth + 2.0 + 1.5 * max_shift;
  return WorldBoxToRawBounds(header, lo - Eigen::Vector3d::Constant(reach),
                             hi + Eigen::Vector3d::Constant(reach));
}
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() {
    cudaStreamSynchronize(value);
    cudaStreamDestroy(value);
  }
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { CudaCheck(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
  ~Event() { cudaEventDestroy(value); }
};
struct PinnedPoints {
  Surfel* data = nullptr;
  PinnedPoints() {
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), PointChunk * sizeof(Surfel)));
  }
  ~PinnedPoints() { cudaFreeHost(data); }
};
struct PointTransfer {
  PinnedPoints host;
  DeviceBuffer device;
  Event copied, consumed;
  bool used = false;
  explicit PointTransfer(bool streaming) : device(streaming ? PointChunk * sizeof(Surfel) : 0) {}
};
class LasPointCache {
  struct SurfelChunk {
    uint64_t offset = 0;
    uint32_t count = 0;
    RawBounds bounds{};
  };
  std::unique_ptr<DeviceBuffer> resident_;
  std::array<std::unique_ptr<PointTransfer>, 2> transfers_;
  // Drain both streams before releasing any buffers, including constructor failure.
  Stream compute_, upload_;
  fs::path coordinates_file_, file_;
  uint64_t count_ = 0, excluded_ = 0, culled_out_of_reach_ = 0, uploaded_bytes_ = 0,
           chunks_projected_ = 0, chunks_skipped_ = 0;
  size_t budget_ = 0, batches_ = 0, slabs_ = 0;
  double prepare_seconds_ = 0, surfel_seconds_ = 0;
  SurfelStats stats_;
  std::vector<SurfelChunk> chunks_;

 public:
  ~LasPointCache() {
    if (compute_.value)
      cudaStreamSynchronize(compute_.value);
    if (upload_.value)
      cudaStreamSynchronize(upload_.value);
    std::error_code ec;
    if (!coordinates_file_.empty())
      fs::remove(coordinates_file_, ec);
    if (!file_.empty())
      fs::remove(file_, ec);
  }
  LasPointCache(const fs::path& las, const LasHeader& header, const fs::path& output, int cache_mb,
                const Reconstruction* model = nullptr) {
    const auto started = Clock::now();
    fs::create_directories(output);
    // Kept coordinates go to a compact file with a histogram along the longest axis, which
    // splits normal estimation into slabs that fit the device together with their halo.
    // Slabs split the file's integer coordinates, before any frame conversion.
    const Eigen::Vector3d extent = header.file_upper - header.file_lower;
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (extent[k] > extent[axis])
        axis = k;
    double lower = std::floor((header.file_lower[axis] - header.origin[axis]) /
                              header.scale[axis]),
           upper = std::ceil((header.file_upper[axis] - header.origin[axis]) /
                             header.scale[axis]);
    const auto reach_bounds = model ? ModelReachBounds(header, *model, MaxDepth + 2.5)
                                    : std::nullopt;
    if (reach_bounds) {
      const int32_t rlo = axis == 0 ? reach_bounds->min.x
                        : axis == 1 ? reach_bounds->min.y
                                    : reach_bounds->min.z;
      const int32_t rhi = axis == 0 ? reach_bounds->max.x
                        : axis == 1 ? reach_bounds->max.y
                                    : reach_bounds->max.z;
      lower = std::max(lower, double(rlo));
      upper = std::min(upper, double(rhi));
      if (upper < lower)
        upper = lower;
    }
    const double bin_width = std::max(1., (upper - lower + 1) / SlabBins);
    std::vector<uint64_t> histogram(SlabBins);
    auto bin = [&](int raw) {
      return std::clamp(int(std::floor((raw - lower) / bin_width)), 0, SlabBins - 1);
    };
    coordinates_file_ = output / "las-coordinates.cache";
    const auto& coordinates = coordinates_file_;
    {
      std::ofstream packed(coordinates, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot create LAS coordinate cache");
      std::vector<int3> kept;
      ScanLas(las, header, [&](uint64_t, const std::vector<LasPoint>& chunk) {
        kept.clear();
        for (const auto& point : chunk) {
          if (!KeepLasPoint(point)) {
            ++excluded_;
            continue;
          }
          if (reach_bounds && !Contains(*reach_bounds, point.raw)) {
            ++culled_out_of_reach_;
            continue;
          }
          kept.push_back(make_int3(point.raw[0], point.raw[1], point.raw[2]));
          ++histogram[bin(point.raw[axis])];
        }
        packed.write(reinterpret_cast<const char*>(kept.data()),
                     std::streamsize(kept.size() * sizeof(int3)));
        if (!packed)
          throw std::runtime_error("Cannot write LAS coordinate cache");
        count_ += kept.size();
      });
      packed.close();
      if (!packed)
        throw std::runtime_error("Incomplete LAS coordinate cache write");
    }
    if (!count_)
      throw std::runtime_error("No valid LAS points within camera reach");
    size_t free, total;
    CudaCheck(cudaMemGetInfo(&free, &total));
    budget_ = std::min(size_t(cache_mb) * 1024 * 1024, free / 4);
    const size_t capacity = std::min(MaxSlabPoints, free / 2 / SlabPointBytes),
                 own_capacity = capacity * 3 / 4;
    // Each slab is a run of histogram bins; its points own the surfels they produce.
    std::vector<std::pair<int, int>> slabs;
    uint64_t filled = 0;
    for (int b = 0, first = 0; b < SlabBins; ++b) {
      if (histogram[b] > own_capacity)
        throw std::runtime_error("LAS normal slab bin exceeds device capacity");
      if (filled + histogram[b] > own_capacity) {
        slabs.emplace_back(first, b);
        first = b;
        filled = 0;
      }
      filled += histogram[b];
      if (b == SlabBins - 1)
        slabs.emplace_back(first, SlabBins);
    }
    if (size_t(count_) * sizeof(Surfel) <= budget_)
      resident_ = std::make_unique<DeviceBuffer>(size_t(count_) * sizeof(Surfel));
    std::ofstream surfels;
    if (!resident_) {
      file_ = output / "las-surfels.cache";
      surfels.open(file_, std::ios::binary);
      if (!surfels)
        throw std::runtime_error("Cannot create LAS surfel cache");
    }
    prepare_seconds_ = Seconds(started);
    const auto surfel_started = Clock::now();
    // Halo points hold every neighbour of an own point together with that neighbour's own
    // neighbours, whose spacing the stray test reads.
    const double halo = (2 * NormalRings + 1) * NormalCell / header.scale[axis];
    uint64_t stored = 0;
    std::vector<int3> chunk(PointChunk), own, rest;
    for (const auto& [first, last] : slabs) {
      const double lo = lower + first * bin_width,
                   hi = last == SlabBins ? std::numeric_limits<double>::infinity()
                                         : lower + last * bin_width;
      own.clear();
      rest.clear();
      int3 origin = make_int3(INT_MAX, INT_MAX, INT_MAX);
      std::ifstream packed(coordinates, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS coordinate cache");
      for (uint64_t read = 0; read < count_; read += PointChunk) {
        const size_t n = size_t(std::min<uint64_t>(PointChunk, count_ - read));
        if (!packed.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(n * sizeof(int3))))
          throw std::runtime_error("Truncated LAS coordinate cache");
        for (size_t i = 0; i < n; ++i) {
          const auto& p = chunk[i];
          const int a = axis == 0 ? p.x : axis == 1 ? p.y : p.z, b = bin(a);
          const bool mine = b >= first && b < last;
          if (!mine && (a < lo - halo || a >= hi + halo))
            continue;
          (mine ? own : rest).push_back(p);
          origin = make_int3(std::min(origin.x, p.x), std::min(origin.y, p.y),
                             std::min(origin.z, p.z));
        }
        if (own.size() + rest.size() > capacity)
          throw std::runtime_error("LAS normal slab and halo exceed device capacity");
      }
      if (own.empty())
        continue;
      const size_t mine = own.size();
      own.insert(own.end(), rest.begin(), rest.end());
      const auto scale =
          make_float3(float(header.scale.x()), float(header.scale.y()), float(header.scale.z()));
      const auto result = SlabSurfels(own, mine, origin, scale, compute_.value, stats_);
      for (size_t base = 0; base < result.size(); base += PointChunk) {
        const size_t n = std::min(PointChunk, result.size() - base);
        int3 cmin = result[base].raw, cmax = result[base].raw;
        for (size_t k = 1; k < n; ++k) {
          const auto& raw = result[base + k].raw;
          cmin = make_int3(std::min(cmin.x, raw.x), std::min(cmin.y, raw.y),
                           std::min(cmin.z, raw.z));
          cmax = make_int3(std::max(cmax.x, raw.x), std::max(cmax.y, raw.y),
                           std::max(cmax.z, raw.z));
        }
        chunks_.push_back({stored + base, uint32_t(n), {cmin, cmax}});
      }
      if (resident_)
        CudaCheck(cudaMemcpy(static_cast<Surfel*>(resident_->data) + stored, result.data(),
                             result.size() * sizeof(Surfel), cudaMemcpyHostToDevice));
      else {
        surfels.write(reinterpret_cast<const char*>(result.data()),
                      std::streamsize(result.size() * sizeof(Surfel)));
        if (!surfels)
          throw std::runtime_error("Cannot write LAS surfel cache");
      }
      stored += result.size();
      ++slabs_;
    }
    fs::remove(coordinates_file_);
    coordinates_file_.clear();
    if (stored != count_)
      throw std::runtime_error("LAS normal slabs do not cover every kept point");
    if (!resident_) {
      surfels.close();
      if (!surfels)
        throw std::runtime_error("Incomplete LAS surfel cache write");
    }
    for (auto& slot : transfers_)
      slot = std::make_unique<PointTransfer>(!resident_);
    surfel_seconds_ = Seconds(surfel_started);
    Log("LAS surfels: " + std::to_string(count_) + " points, " + std::to_string(stats_.normals) +
        " with normals, " + std::to_string(slabs_) + " slabs, " +
        (resident_ ? "GPU cache" : "compact file with double-buffered upload"));
  }
  cudaStream_t StreamHandle() const { return compute_.value; }
  uint64_t Count() const { return count_; }
  uint64_t Excluded() const { return excluded_; }
  void ProjectNearest(const Projection* cameras, const float3* rays, float* nearest, int width,
                      int height, size_t images, const RawBounds* batch_bounds = nullptr) {
    Project(cameras, rays, nearest, nullptr, width, height, images, false, batch_bounds);
    ++batches_;
  }
  void Project(const Projection* cameras, const float3* rays, float* nearest, Blend* blends,
               int width, int height, size_t images, const RawBounds* batch_bounds = nullptr) {
    for (const bool accumulate : {false, true})
      Project(cameras, rays, nearest, blends, width, height, images, accumulate, batch_bounds);
    ++batches_;
  }
  void Project(const Projection* cameras, const float3* rays, float* nearest, Blend* blends,
               int width, int height, size_t images, bool accumulate,
               const RawBounds* batch_bounds) {
    std::ifstream packed;
    if (!resident_) {
      packed.open(file_, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS surfel cache");
    }
    size_t index = 0;
    uint64_t file_offset = 0;
    for (const auto& chunk : chunks_) {
      if (batch_bounds && !Overlaps(chunk.bounds, *batch_bounds)) {
        ++chunks_skipped_;
        continue;
      }
      ++chunks_projected_;
      const Surfel* points;
      PointTransfer* transfer = nullptr;
      if (resident_)
        points = static_cast<const Surfel*>(resident_->data) + chunk.offset;
      else {
        transfer = transfers_[index++ % transfers_.size()].get();
        if (transfer->used)
          CudaCheck(cudaEventSynchronize(transfer->copied.value));
        if (file_offset != chunk.offset) {
          packed.seekg(std::streamoff(chunk.offset * sizeof(Surfel)), std::ios::beg);
          if (!packed)
            throw std::runtime_error("Cannot seek LAS surfel cache");
          file_offset = chunk.offset;
        }
        const size_t bytes = size_t(chunk.count) * sizeof(Surfel);
        if (!packed.read(reinterpret_cast<char*>(transfer->host.data), std::streamsize(bytes)))
          throw std::runtime_error("Truncated LAS surfel cache");
        file_offset += chunk.count;
        if (transfer->used)
          CudaCheck(cudaStreamWaitEvent(upload_.value, transfer->consumed.value, 0));
        CudaCheck(cudaMemcpyAsync(transfer->device.data, transfer->host.data, bytes,
                                  cudaMemcpyHostToDevice, upload_.value));
        CudaCheck(cudaEventRecord(transfer->copied.value, upload_.value));
        CudaCheck(cudaStreamWaitEvent(compute_.value, transfer->copied.value, 0));
        points = static_cast<const Surfel*>(transfer->device.data);
        uploaded_bytes_ += bytes;
      }
      supervision_gpu::Project(points, chunk.count, images, cameras, rays, nearest, blends, width,
                               height, accumulate, compute_.value);
      if (transfer) {
        CudaCheck(cudaEventRecord(transfer->consumed.value, compute_.value));
        transfer->used = true;
      }
    }
  }
  void Finish() {
    if (compute_.value)
      CudaCheck(cudaStreamSynchronize(compute_.value));
    if (upload_.value)
      CudaCheck(cudaStreamSynchronize(upload_.value));
    resident_.reset();
    for (auto& slot : transfers_)
      slot.reset();
    if (!file_.empty()) {
      fs::remove(file_);
      file_.clear();
    }
  }
  Json Surfels() const {
    auto median = [](std::vector<float> values) {
      if (values.empty())
        return 0.;
      std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
      return double(values[values.size() / 2]);
    };
    return {{"estimator", "knn_spacing_trimmed_pca_tangent_plane"},
            {"neighbours", Neighbours},
            {"minimum_neighbours", MinNeighbours},
            {"grid_cell_m", NormalCell},
            {"search_reach_m", NormalRings * NormalCell},
            {"minimum_plane_radius_m", NormalRadius},
            {"inlier_band_m", PlaneBand},
            {"plane_rms_limit_m", PlaneRms},
            {"minimum_inlier_fraction", .6},
            {"footprint", "tangent disk of 2 x min(kth-neighbour distance, neighbour median) / sqrt(k), at most 0.05 m, planar points with k neighbours within the reach only"},
            {"maximum_spacing_to_neighbour_median_spacing", StraySpacing},
            {"maximum_splat_pixels", MaxSplat},
            {"slabs", slabs_},
            {"points_with_normal", stats_.normals},
            {"median_footprint_m", median(stats_.radius)},
            {"median_plane_rms_m", median(stats_.rms)},
            {"seconds", surfel_seconds_}};
  }
  Json Profile() const {
    return {{"storage", resident_ ? "gpu" : "file"},
            {"las_scans", 1},
            {"budget_bytes", budget_},
            {"surfel_bytes", count_ * sizeof(Surfel)},
            {"culled_out_of_reach", culled_out_of_reach_},
            {"chunks_total", chunks_.size()},
            {"chunks_projected", chunks_projected_},
            {"chunks_skipped", chunks_skipped_},
            {"gpu_cache_bytes", resident_ ? resident_->bytes : 0},
            {"transfer_buffer_bytes", resident_ ? 0 : 2 * PointChunk * sizeof(Surfel)},
            {"host_to_device_bytes", uploaded_bytes_},
            {"projection_batches", batches_},
            {"prepare_seconds", prepare_seconds_}};
  }
};
fs::path ExtraMask(const SupervisionOptions& options, const std::string& name) {
  return options.mask_dir / fs::u8path(name + ".png");
}
fs::path StaticMaskPath(const fs::path& masks, uint32_t image) {
  return masks / (std::to_string(image) + ".png");
}
cv::Mat StaticMask(const Scene& scene, const fs::path& masks, uint32_t image, const Camera& camera,
                   const SupervisionOptions& options) {
  auto mask = ReadImage(StaticMaskPath(masks, image), cv::IMREAD_GRAYSCALE);
  mask = mask == 255;
  if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, scene.images[image].name))) {
    auto extra = ReadImage(ExtraMask(options, scene.images[image].name), cv::IMREAD_GRAYSCALE);
    if (extra.size() != mask.size())
      throw std::runtime_error("Geometry exclusion mask must match its original image");
    mask &= extra == 255;
  }
  if (options.crop) {
    const auto& w = (*options.crop)[scene.images[image].camera];
    if (w.x + w.width > mask.cols || w.y + w.height > mask.rows)
      throw std::runtime_error("Crop window " + w.ToJson().dump() + " lies outside the " +
                               std::to_string(mask.cols) + "x" + std::to_string(mask.rows) +
                               " static mask of " + scene.images[image].name);
    mask = mask(cv::Rect(w.x, w.y, w.width, w.height)).clone();
  }
  // A map pixel is valid only when its whole footprint is. The invalid indicator is resampled,
  // so a fully valid footprint sums exact zeros; resampling the valid one instead lets float
  // area weights fall short of 1 on fixed columns of non-integer scales.
  cv::Mat invalid, resized;
  cv::Mat(mask == 0).convertTo(invalid, CV_32F, 1. / 255);
  cv::resize(invalid, resized, {camera.width, camera.height}, 0, 0, cv::INTER_AREA);
  mask = resized == 0.f;
  mask /= 255;
  return mask;
}
std::pair<uint64_t, uint64_t> Counts(const cv::Mat& depth, const cv::Mat& normal,
                                     bool normal_needs_depth) {
  static const auto squares = [] {
    std::array<double, 256> values;
    for (int v = 0; v < 256; ++v)
      values[v] = std::pow(v / 127.5 - 1, 2);
    return values;
  }();
  uint64_t depths = 0, normals = 0;
  for (int y = 0; y < depth.rows; ++y) {
    const auto* d = depth.ptr<uint16_t>(y);
    const auto* n = normal.ptr<uint8_t>(y);
    for (int x = 0; x < depth.cols; ++x, n += 3) {
      const bool has_depth = d[x] != 0;
      const bool has_normal = n[0] || n[1] || n[2];
      if (has_normal) {
        if (normal_needs_depth && !has_depth)
          throw std::runtime_error("Normal has no corresponding valid LAS depth");
        const double norm = squares[n[0]] + squares[n[1]] + squares[n[2]];
        if (std::abs(norm - 1) > .025)
          throw std::runtime_error("Nonunit encoded LAS normal");
      }
      depths += has_depth;
      normals += has_normal;
    }
  }
  return {depths, normals};
}
// The geometry engine a monocular request was made with.
Json GeometryIdentity(const fs::path& root) {
  const auto manifest = ReadJson(root / "manifest.json");
  if (!manifest.at("models").contains("geometry"))
    throw std::runtime_error("Supervision models have no geometry engine");
  return {{"models", root.u8string()},
          {"manifest", FileMetadata(root / "manifest.json")},
          {"engine", FileMetadata(root / "engines/geometry.engine")},
          {"sha256", manifest.at("models").at("geometry").at("sha256")},
          {"source", manifest.at("geometry_source")},
          {"face_size", manifest.at("geometry_size")},
          {"tokens", manifest.at("geometry_tokens")}};
}
}  // namespace
Camera MapCamera(const Camera& source, int max_size) {
  auto camera = source;
  const double scale = std::min(1., double(max_size) / std::max(source.width, source.height));
  camera.width = std::max(1, int(std::lround(source.width * scale)));
  camera.height = std::max(1, int(std::lround(source.height * scale)));
  for (int i = 0; i < 4; ++i)
    camera.params[i] *= i % 2 ? double(camera.height) / source.height
                              : double(camera.width) / source.width;
  return camera;
}
SupervisionOptions ParseSupervisionOptions(const Arguments& args) {
  SupervisionOptions options;
  options.enabled = args.Get("--no-las-supervision") != "true";
  options.static_masks = args.Get("--no-static-masks") != "true";
  options.max_size = args.Integer("--supervision-max-size", 1024);
  options.boundary = args.Integer("--supervision-boundary", 4);
  options.cache_mb = args.Integer("--supervision-cache-mb", 4096);
  if (!args.Get("--supervision-mask-dir").empty())
    options.mask_dir = args.Path("--supervision-mask-dir");
  if (options.max_size < 128 || options.max_size > 4096 || options.boundary > 32)
    throw std::runtime_error("Supervision max size must be 128-4096 and boundary 1-32 pixels");
  if (options.cache_mb < 1 || options.cache_mb > 8192)
    throw std::runtime_error("Supervision cache budget must be 1-8192 MiB");
  if (!options.enabled &&
      (args.values.count("--supervision-max-size") || args.values.count("--supervision-boundary") ||
       args.values.count("--supervision-cache-mb") || !options.mask_dir.empty() ||
       args.values.count("--supervision-source") || args.values.count("--supervision-models")))
    throw std::runtime_error("Supervision options cannot be used with --no-las-supervision");
  if (!options.mask_dir.empty() && !fs::is_directory(options.mask_dir))
    throw std::runtime_error("Geometry exclusion mask directory does not exist");
  if (!options.enabled)
    return options;
  options.source = args.Get("--supervision-source", "moge");
  if (options.source != "las" && options.source != "moge")
    throw std::runtime_error("Supervision source must be las or moge");
  if (!args.Get("--supervision-models").empty()) {
    if (options.source != "moge")
      throw std::runtime_error("--supervision-models requires --supervision-source moge");
    options.geometry_models = args.Path("--supervision-models");
  }
  return options;
}
void ResolveSupervisionModels(SupervisionOptions& options, const fs::path& package) {
  if (!options.enabled || options.source != "moge")
    return;
  // Without --supervision-models, the model package of the reconstruction.
  if (options.geometry_models.empty())
    options.geometry_models = package;
  if (!ReadJson(options.geometry_models / "manifest.json").at("models").contains("geometry"))
    throw std::runtime_error(
        "MoGe supervision needs the geometry engine: " + options.geometry_models.u8string() +
        "; prepare it with run.ps1 prepare-models, pass --supervision-models, or use "
        "--supervision-source las");
}
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& masks_dir,
                        const SupervisionOptions& options, bool rolling) {
  if (!options.enabled)
    return nullptr;
  if (FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed since reconstruction; choose a new work directory");
  Json masks = Json::object(), static_masks = Json::object();
  for (const auto& image : scene.images) {
    if (!model.images[image.id].registered)
      continue;
    static_masks[image.name] = FileMetadata(StaticMaskPath(masks_dir, image.id));
    if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, image.name)))
      masks[image.name] = FileMetadata(ExtraMask(options, image.name));
  }
  const bool monocular = options.source == "moge";
  Json request = {{"max_size", options.max_size},
                  {"boundary_pixels", options.boundary},
                  {"las_file", FileMetadata(scene.las)},
                  {"exclusion_mask_dir", options.mask_dir.u8string()},
                  {"static_masks", static_masks},
                  {"exclusion_masks", masks}};
  if (options.crop)
    request["crop_windows"] = {(*options.crop)[0].ToJson(), (*options.crop)[1].ToJson()};
  if (monocular) {
    request["source"] = "moge";
    request["pipeline"] = MonocularSource;
    request["rolling_shutter"] = rolling;
    request["geometry"] = GeometryIdentity(options.geometry_models);
  }
  return request;
}
namespace {
Projection MakeProjection(const LasHeader& header, const Reconstruction& model,
                          const ShutterMotion* motion, uint32_t id, uint32_t cid,
                          const Camera& camera) {
  Projection p{};
  const auto& pose = model.images[id].pose;
  // The LAS frame conversion folds into each camera, which projects file coordinates.
  const Eigen::Matrix3d frame = header.frame.scale * header.frame.rotation.toRotationMatrix(),
                        rotation = pose.Rotation().toRotationMatrix() * frame;
  const Eigen::Vector3d translation =
      pose.Rotation().toRotationMatrix() * (frame * header.origin + header.frame.translation) +
      pose.Translation();
  for (int r = 0; r < 3; ++r) {
    p.translation[r] = translation[r];
    p.scales[r] = header.scale[r];
    for (int c = 0; c < 3; ++c)
      p.rotation[r * 3 + c] = rotation(r, c);
  }
  std::copy(camera.params.begin(), camera.params.end(), p.params);
  const double cos_max = std::cos(MaxTheta) - 1e-6;
  if (motion) {
    const Eigen::Vector3d moved = motion->readout[cid] * motion->velocity[id];
    for (int k = 0; k < 3; ++k) {
      p.motion[k] = motion->rotation[id][k];
      p.motion[3 + k] = moved[k];
    }
    p.rolling = 1;
    p.axis = motion->axis;
    const double t2 = (Pi / 2) * (Pi / 2);
    const double max_radial = (Pi / 2) * (1 + std::abs(camera.params[4]) * t2 +
                                          std::abs(camera.params[5]) * t2 * t2 +
                                          std::abs(camera.params[6]) * t2 * t2 * t2 +
                                          std::abs(camera.params[7]) * t2 * t2 * t2 * t2);
    const double axis_size = motion->axis ? camera.height : camera.width;
    const double axis_focal = std::abs(camera.params[motion->axis ? 1 : 0]);
    const double axis_center = camera.params[motion->axis ? 3 : 2];
    const double max_s = (std::abs(axis_center - .5 * axis_size) + axis_focal * max_radial) /
                             std::max(1., axis_size) +
                         .5;
    const double shift = max_s * moved.norm();
    const double rot = max_s * motion->rotation[id].norm();
    p.max_q2 = std::pow(MaxDepth + shift + 1e-4, 2);
    p.min_qz_slope = std::max(0., cos_max - rot);
    p.qz_slack = (1. + cos_max + rot) * shift + 1e-4;
  } else {
    p.max_q2 = (MaxDepth + 1e-4) * (MaxDepth + 1e-4);
    p.min_qz_slope = cos_max;
    p.qz_slack = 0;
  }
  return p;
}
}  // namespace
struct LasSurfelCache::Implementation : LasPointCache {
  using LasPointCache::LasPointCache;
};
LasSurfelCache::LasSurfelCache(const fs::path& las, const LasHeader& header, const fs::path& scratch,
                               int cache_mb, const Reconstruction* model)
    : impl_(std::make_unique<Implementation>(las, header, scratch, cache_mb, model)) {}
LasSurfelCache::~LasSurfelCache() = default;
void LasSurfelCache::Finish() { impl_->Finish(); }
void ProjectLasDepths(LasSurfelCache& cache, const LasHeader& header, const Reconstruction& model,
                      const ShutterMotion* motion, int max_size,
                      const std::function<void(uint32_t, cv::Mat&&)>& consume) {
  auto& points = cache.Impl();
  const auto stream = points.StreamHandle();
  for (uint32_t cid = 0; cid < model.cameras.size(); ++cid) {
    std::vector<uint32_t> images;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        images.push_back(i);
    if (images.empty())
      continue;
    const auto camera = MapCamera(model.cameras[cid], max_size);
    const int w = camera.width, h = camera.height, size = w * h;
    const size_t batch =
        std::min<size_t>(160, std::max<size_t>(1, DepthBudget / (size * sizeof(float))));
    std::vector<float3> ray_host(size);
    const double valid_radius = camera.Radius();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = (x + .5 - camera.params[2]) / camera.params[0],
                     dy = (y + .5 - camera.params[3]) / camera.params[1];
        const Eigen::Vector3d ray = dx * dx + dy * dy <= valid_radius * valid_radius
                                       ? camera.Ray({x + .5, y + .5}) : Eigen::Vector3d::Zero();
        ray_host[y * w + x] = make_float3(float(ray.x()), float(ray.y()), float(ray.z()));
      }
    DeviceBuffer nearest(batch * size * sizeof(float)), cameras(batch * sizeof(Projection)),
        rays(size * sizeof(float3));
    CudaCheck(
        cudaMemcpyAsync(rays.data, ray_host.data(), rays.bytes, cudaMemcpyHostToDevice, stream));
    for (size_t first = 0; first < images.size(); first += batch) {
      const size_t count = std::min(batch, images.size() - first);
      std::vector<Projection> projection(count);
      for (size_t j = 0; j < count; ++j)
        projection[j] = MakeProjection(header, model, motion, images[first + j], cid, camera);
      CudaCheck(cudaMemcpyAsync(cameras.data, projection.data(), count * sizeof(Projection),
                                cudaMemcpyHostToDevice, stream));
      CudaCheck(cudaMemsetAsync(nearest.data, 0x7f, count * size * sizeof(float), stream));
      const auto batch_bounds =
          BatchReachBounds(header, model, motion, images, first, count, cid);
      points.ProjectNearest(static_cast<const Projection*>(cameras.data),
                            static_cast<const float3*>(rays.data),
                            static_cast<float*>(nearest.data), w, h, count, &batch_bounds);
      ResolveNearest(static_cast<float*>(nearest.data), count * size_t(size), stream);
      for (size_t j = 0; j < count; ++j) {
        cv::Mat result(h, w, CV_32FC1);
        CudaCheck(cudaMemcpyAsync(result.data,
                                  static_cast<const float*>(nearest.data) + j * size_t(size),
                                  size_t(size) * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CudaCheck(cudaStreamSynchronize(stream));
        consume(images[first + j], std::move(result));
      }
    }
  }
  CudaCheck(cudaStreamSynchronize(stream));
}
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model,
                          const fs::path& masks_dir, const fs::path& output,
                          const SupervisionOptions& options, LasSurfelCache& cache,
                          const ShutterMotion* motion) {
  const auto started = Clock::now();
  const auto header = ReadSceneLas(scene);
  const auto request = SupervisionRequest(scene, model, masks_dir, options, motion != nullptr);
  std::vector<Json> rows(model.images.size());
  const bool monocular = options.source == "moge";
  std::unique_ptr<Models> geometry;
  std::unique_ptr<MonocularEstimator> estimator;
  if (monocular) {
    geometry = std::make_unique<Models>(options.geometry_models);
    estimator = std::make_unique<MonocularEstimator>(*geometry);
  }
  Json lenses = Json::array();
  auto& points = cache.Impl();
  const auto stream = points.StreamHandle();
  uint64_t depth_pixels = 0, normal_pixels = 0;
  size_t done = 0;
  struct Written {
    uint32_t id;
    Json row;
  };
  std::deque<std::future<Written>> writes;
  // Where the export waits: projecting the LAS batches, MoGe inference and CPU-side result writes.
  double project_seconds = 0, estimate_seconds = 0, write_wait_seconds = 0, face_load_wait_seconds = 0;
  auto finish = [&] {
    const auto waiting = Clock::now();
    // Leave the queue before get(): a throwing task must not leave an invalid future behind.
    auto pending = std::move(writes.front());
    writes.pop_front();
    auto written = pending.get();
    write_wait_seconds += Seconds(waiting);
    depth_pixels += written.row.at("depth_pixels").get<uint64_t>();
    normal_pixels += written.row.at("normal_pixels").get<uint64_t>();
    rows[written.id] = std::move(written.row);
    ++done;
    if (done % 16 == 0 || done == model.Registered())
      Log("LAS supervision: " + std::to_string(done) + "/" + std::to_string(model.Registered()) +
          " images, " + std::to_string(Seconds(started)) + " s");
  };
  struct LoadedMask {
    cv::Mat pixels;
    int count;
    cv::Mat prefiltered;  // Prefiltered RGB photo for rolling-shutter GPU MoGe resampling
  };
  // Lens plans outlive their camera's loop: the monocular fits run with the pending writes.
  std::array<std::optional<MonocularLens>, 2> lens_plans;
  // Pending writes use the lens plans above, which are destroyed before the
  // queue itself; on an exception, wait for the tasks first (their results are discarded).
  struct DrainWrites {
    std::deque<std::future<Written>>& queue;
    ~DrainWrites() {
      for (auto& write : queue)
        if (write.valid())
          write.wait();
    }
  } drain_writes{writes};
  for (uint32_t cid = 0; cid < model.cameras.size(); ++cid) {
    std::vector<uint32_t> images;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        images.push_back(i);
    if (images.empty())
      continue;
    const auto camera = MapCamera(model.cameras[cid], options.max_size);
    const int w = camera.width, h = camera.height, size = w * h;
    auto& lens = lens_plans.at(cid);
    if (monocular) {
      lens = PlanMonocularLens(model.cameras[cid], camera, estimator->FaceSize(), FaceHalfDegrees);
      lenses.push_back({{"camera", cid}, {"plan", lens->ToJson()}});
    }
    const size_t batch = std::min<size_t>(
        32, std::max<size_t>(1, DepthBudget / (size * (sizeof(float) + sizeof(Blend)))));
    std::vector<float3> ray_host(size);
    const double valid_radius = camera.Radius();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = (x + .5 - camera.params[2]) / camera.params[0],
                     dy = (y + .5 - camera.params[3]) / camera.params[1];
        const Eigen::Vector3d ray = dx * dx + dy * dy <= valid_radius * valid_radius
                                       ? camera.Ray({x + .5, y + .5}) : Eigen::Vector3d::Zero();
        ray_host[y * w + x] = make_float3(float(ray.x()), float(ray.y()), float(ray.z()));
      }
    DeviceBuffer nearest(batch * size * sizeof(float)), blends(batch * size * sizeof(Blend)),
        cameras(batch * sizeof(Projection)), rays(size * sizeof(float3)),
        projected(size * sizeof(float)), codes(size * sizeof(unsigned)),
        clean(size * sizeof(float)), filled(size * sizeof(float)),
        seeds(size * sizeof(float)), mask(size), limit(size), dilated(size), edge(size),
        millimetres(size * sizeof(uint16_t)), normals(size * sizeof(uchar3));
    CudaCheck(
        cudaMemcpyAsync(rays.data, ray_host.data(), rays.bytes, cudaMemcpyHostToDevice, stream));
    // Masks for the following images decode while the GPU projects the current batch.
    std::deque<std::future<LoadedMask>> masks;
    size_t requested = 0;
    auto load_masks = [&] {
      for (; masks.size() < Lookahead && requested < images.size(); ++requested)
        masks.push_back(std::async(std::launch::async, [&, id = images[requested]] {
          auto pixels = StaticMask(scene, masks_dir, id, camera, options);
          const int count = cv::countNonZero(pixels);
          LoadedMask loaded{std::move(pixels), count, {}};
          if (lens) {
            auto photo = ReadImage(scene.images[id].source, cv::IMREAD_COLOR);
            if (options.crop) {
              const auto& window = (*options.crop)[cid];
              photo = photo(cv::Rect(window.x, window.y, window.width, window.height)).clone();
            }
            cv::cvtColor(photo, photo, cv::COLOR_BGR2RGB);
            loaded.prefiltered = lens->Prefilter(photo);
          }
          return loaded;
        }));
    };
    for (size_t first = 0; first < images.size(); first += batch) {
      load_masks();
      const size_t count = std::min(batch, images.size() - first);
      std::vector<Projection> projection(count);
      for (size_t j = 0; j < count; ++j)
        projection[j] = MakeProjection(header, model, motion, images[first + j], cid, camera);
      CudaCheck(cudaMemcpyAsync(cameras.data, projection.data(), count * sizeof(Projection),
                                cudaMemcpyHostToDevice, stream));
      CudaCheck(cudaMemsetAsync(nearest.data, 0x7f, count * size * sizeof(float), stream));
      CudaCheck(cudaMemsetAsync(blends.data, 0, count * size * sizeof(Blend), stream));
      const auto batch_bounds =
          BatchReachBounds(header, model, motion, images, first, count, cid);
      const auto projecting = Clock::now();
      points.Project(static_cast<const Projection*>(cameras.data),
                     static_cast<const float3*>(rays.data), static_cast<float*>(nearest.data),
                     static_cast<Blend*>(blends.data), w, h, count, &batch_bounds);
      CudaCheck(cudaStreamSynchronize(stream));
      project_seconds += Seconds(projecting);
      for (size_t j = 0; j < count; ++j) {
        const uint32_t id = images[first + j];
        const auto loading = Clock::now();
        const auto static_mask = masks.front().get();
        face_load_wait_seconds += Seconds(loading);
        masks.pop_front();
        load_masks();
        CudaCheck(cudaMemcpyAsync(mask.data, static_mask.pixels.data, size, cudaMemcpyHostToDevice,
                                  stream));
        EncodeMaps(
            static_cast<const Blend*>(blends.data) + j * size, w, h, options.boundary,
            float(camera.params[0]), static_cast<const uint8_t*>(mask.data),
            static_cast<const float3*>(rays.data), static_cast<float*>(projected.data),
            static_cast<unsigned*>(codes.data), static_cast<float*>(clean.data),
            static_cast<float*>(filled.data), static_cast<uint8_t*>(limit.data),
            static_cast<uint8_t*>(dilated.data), static_cast<float*>(seeds.data),
            static_cast<uint8_t*>(edge.data), static_cast<uint16_t*>(millimetres.data),
            static_cast<uchar3*>(normals.data), stream);
        cv::Mat depth(h, w, CV_16UC1), normal(h, w, CV_8UC3);
        CudaCheck(cudaMemcpyAsync(depth.data, millimetres.data, millimetres.bytes,
                                  cudaMemcpyDeviceToHost, stream));
        CudaCheck(cudaMemcpyAsync(normal.data, normals.data, normals.bytes, cudaMemcpyDeviceToHost,
                                  stream));
        cv::Mat semantic;
        if (monocular) {
          semantic.create(h, w, CV_8UC1);
          CudaCheck(cudaMemcpyAsync(semantic.data, limit.data, limit.bytes, cudaMemcpyDeviceToHost,
                                    stream));
        }
        CudaCheck(cudaStreamSynchronize(stream));
        // MoGe faces replace the maps; the LAS depth only scales them and gates their depth. The
        // GPU estimates the faces here (un-rotated along omega when rolling-shutter is active);
        // the fit and encoding run with the write on the CPU.
        std::vector<FaceEstimate> estimates;
        cv::Mat allowed;
        const bool rolling_moge = motion != nullptr;
        const Eigen::Vector3d omega = motion ? motion->rotation[id] : Eigen::Vector3d::Zero();
        const int axis = motion ? motion->axis : 1;
        if (monocular) {
          allowed = static_mask.pixels.clone();
          allowed.setTo(0, semantic);
          const auto estimating = Clock::now();
          estimates =
              estimator->Estimate(*lens, static_mask.prefiltered, omega, axis, rolling_moge);
          estimate_seconds += Seconds(estimating);
        }
        const MonocularLens* plan = monocular ? &*lens : nullptr;
        const auto& name = scene.images[id].name;
        const auto dp = output / "depths" / fs::u8path(name + ".png"),
                   np = output / "normals" / fs::u8path(name + ".png");
        fs::create_directories(dp.parent_path());
        fs::create_directories(np.parent_path());
        if (writes.size() >= Lookahead)
          finish();
        writes.push_back(std::async(std::launch::async, [=, estimates = std::move(estimates),
                                                         static_pixels = static_mask.count]() mutable {
          Json monocular_row;
          if (plan) {
            const Eigen::Vector3d* omega_ptr = rolling_moge ? &omega : nullptr;
            cv::Mat cleaned_depth, moge_normal;
            monocular_row = CleanLasAndEncodeNormals(*plan, estimates, depth, normal, allowed,
                                                     omega_ptr, axis, MinIncidence, cleaned_depth,
                                                     moge_normal);
            depth = std::move(cleaned_depth);
            normal = std::move(moge_normal);
          }
          const auto valid = Counts(depth, normal, !monocular);
          WriteImage(dp, depth);
          WriteImage(np, normal);
          Json row = {{"image_index", id},
                      {"name", name},
                      {"width", w},
                      {"height", h},
                      {"depth_size", fs::file_size(dp)},
                      {"normal_size", fs::file_size(np)},
                      {"static_pixels", static_pixels},
                      {"depth_pixels", valid.first},
                      {"normal_pixels", valid.second}};
          if (monocular)
            row["monocular"] = std::move(monocular_row);
          return Written{id, std::move(row)};
        }));
      }
    }
  }
  while (!writes.empty())
    finish();
  const uint64_t accepted_records = points.Count(), excluded_records = points.Excluded();
  const Json point_cache_profile = points.Profile(), surfel_summary = points.Surfels();
  const Json geometry_profile = monocular ? geometry->Profile() : Json();
  points.Finish();
  estimator.reset();
  geometry.reset();
  uint64_t las_input_pixels = 0, moge_edge_dropped_pixels = 0, ransac_dropped_near_pixels = 0,
           ransac_dropped_far_pixels = 0, far_normal_rescued_pixels = 0,
           penetration_normal_dropped_pixels = 0, normal_disagree_dropped_pixels = 0,
           las_normal_used_pixels = 0, moge_normal_fill_pixels = 0;
  if (monocular) {
    for (const auto& row : rows)
      if (!row.is_null()) {
        const auto& m = row.at("monocular");
        las_input_pixels += m.at("las_input_pixels").get<uint64_t>();
        moge_edge_dropped_pixels += m.at("moge_edge_dropped_pixels").get<uint64_t>();
        ransac_dropped_near_pixels += m.at("ransac_dropped_near_pixels").get<uint64_t>();
        ransac_dropped_far_pixels += m.at("ransac_dropped_far_pixels").get<uint64_t>();
        far_normal_rescued_pixels += m.at("far_normal_rescued_pixels").get<uint64_t>();
        penetration_normal_dropped_pixels += m.at("penetration_normal_dropped_pixels").get<uint64_t>();
        normal_disagree_dropped_pixels += m.at("normal_disagree_dropped_pixels").get<uint64_t>();
        las_normal_used_pixels += m.at("las_normal_used_pixels").get<uint64_t>();
        moge_normal_fill_pixels += m.at("moge_normal_fill_pixels").get<uint64_t>();
      }
    Log("LAS MoGe cleaning: input=" + std::to_string(las_input_pixels) +
        ", edge_dropped=" + std::to_string(moge_edge_dropped_pixels) +
        ", near_dropped=" + std::to_string(ransac_dropped_near_pixels) +
        ", far_dropped=" + std::to_string(ransac_dropped_far_pixels) +
        ", far_rescued=" + std::to_string(far_normal_rescued_pixels) +
        ", penetration_normal_dropped=" + std::to_string(penetration_normal_dropped_pixels) +
        ", normal_disagree_dropped=" + std::to_string(normal_disagree_dropped_pixels) +
        ", las_normal_used=" + std::to_string(las_normal_used_pixels) +
        ", moge_normal_fill=" + std::to_string(moge_normal_fill_pixels));
  }
  if (!depth_pixels || !normal_pixels)
    throw std::runtime_error("LAS supervision has no reliable depth or normal pixels");
  Json images = Json::array();
  for (const auto& row : rows)
    if (!row.is_null())
      images.push_back(row);
  if (request != SupervisionRequest(scene, model, masks_dir, options, motion != nullptr))
    throw std::runtime_error("LAS or geometry exclusion masks changed during supervision export");
  Json summary = {{"request", request},
          {"source", "valid original LAS records"},
          {"depth_source", "las"},
          {"normal_source", "las"},
          {"projection", motion ? "rolling_shutter_readout_line" : "global_shutter"},
          {"accepted_records", accepted_records},
          {"excluded_records", excluded_records},
          {"point_cache", point_cache_profile},
          {"depth_format", "uint16_png_mm"},
          {"depth_type", "ray_distance"},
          {"invalid_depth", 0},
          {"normal_format", "rgb8_png"},
          {"normal_frame", "opencv_camera_facing_camera"},
          {"normal_encoding", "round(127.5 * (n + 1))"},
          {"invalid_normal_rgb", {0, 0, 0}},
          {"depth_validity", "isolated_outlier_removal_occlusion_boundary_and_local_depth_support"},
          {"depth_support_pixels", 4},
          {"occlusion_step", "background beyond 1.1 x foreground + 0.05 m within 5x5"},
          {"occlusion_margin", {{"background_pixels", 2},
                                {"foreground_spread_m", OcclusionSpread},
                                {"foreground_pixels", "ceil(spread x fx / depth), at most 24"}}},
          {"semantic_margin_pixels", options.boundary},
          {"depth_rendering", "las_surfel_tangent_disks_blended_within_0.01_m_plus_1_percent_of_nearest"},
          {"normal_blend_minimum_agreement", MinNormalAgreement},
          {"normal_estimator", motion ? "las_knn_pca_surfel_readout_line" : "las_knn_pca_surfel"},
          {"surfels", surfel_summary},
          {"minimum_abs_normal_dot_ray", MinIncidence},
          {"depth_pixels", depth_pixels},
          {"normal_pixels", normal_pixels},
          {"images", images},
          {"timing",
           {{"las_project_seconds", project_seconds},
            {"moge_estimate_seconds", estimate_seconds},
            {"photo_and_mask_wait_seconds", face_load_wait_seconds},
            {"result_write_wait_seconds", write_wait_seconds}}},
          {"seconds", Seconds(started)}};
  if (monocular) {
    summary["source"] = MonocularSource;
    summary["depth_source"] = "las_moge_cleaned";
    summary["normal_source"] = "las_planar_moge2_fill";
    summary["anchor_source"] = "valid original LAS records";
    summary["geometry_model"] = request.at("geometry");
    summary["lenses"] = lenses;
    summary["depth_rendering"] =
        "las_surfel_tangent_disks_cleaned_by_moge_edges_and_per_face_ransac";
    summary["depth_validity"] =
        "isolated_outlier_removal_occlusion_boundary_local_depth_support_moge_edge_mask_and_per_face_ransac";
    summary["normal_estimator"] = motion ? "las_knn_pca_surfel_moge2_vitl_readout_line"
                                         : "las_knn_pca_surfel_moge2_vitl";
    summary["normal_validity"] =
        "static mask off semantic margin, incidence, planar LAS surfel priority on RANSAC-kept off-edge "
        "pixels with MoGe-2 backfill, excluding > 35 deg off-edge conflicts and unrescued LAS "
        "glass/mirror penetration";
    summary["monocular"] = {
        {"cleaning",
         "per-face MoGe depth/normal edge mask dilated 2 px, multi-face inverse-depth RANSAC "
         "with planar LAS normal rescue (<= 25 deg and <= 3x depth), planar LAS normal priority "
         "with MoGe-2 backfill, and off-edge MoGe-vs-LAS normal disagreement (> 35 deg) gating"},
        {"las_input_pixels", las_input_pixels},
        {"moge_edge_dropped_pixels", moge_edge_dropped_pixels},
        {"ransac_dropped_near_pixels", ransac_dropped_near_pixels},
        {"ransac_dropped_far_pixels", ransac_dropped_far_pixels},
        {"far_normal_rescued_pixels", far_normal_rescued_pixels},
        {"penetration_normal_dropped_pixels", penetration_normal_dropped_pixels},
        {"normal_disagree_dropped_pixels", normal_disagree_dropped_pixels},
        {"las_normal_used_pixels", las_normal_used_pixels},
        {"moge_normal_fill_pixels", moge_normal_fill_pixels},
        {"normals_need_depth", false},
        {"inference", geometry_profile}};
  }
  return summary;
}
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary) {
  const bool monocular = summary.value("source", std::string()) == MonocularSource;
  if (summary.at("depth_type") != "ray_distance" ||
      summary.at("depth_format") != "uint16_png_mm" || summary.at("normal_format") != "rgb8_png" ||
      summary.at("normal_frame") != "opencv_camera_facing_camera" ||
      summary.at("images").size() != photos.size())
    throw std::runtime_error("Unexpected LAS supervision contract");
  uint64_t depth_total = 0, normal_total = 0;
  std::deque<std::future<std::pair<uint64_t, uint64_t>>> checks;
  auto finish = [&] {
    const auto counts = checks.front().get();
    checks.pop_front();
    depth_total += counts.first;
    normal_total += counts.second;
  };
  for (size_t i = 0; i < photos.size(); ++i) {
    const auto& image = photos[i];
    const auto& row = summary.at("images")[i];
    if (row.at("image_index") != image.at("image_index") || row.at("name") != image.at("name"))
      throw std::runtime_error("LAS supervision image identity differs");
    const auto camera = MapCamera(model.cameras[image.at("camera").get<size_t>()],
                                  summary.at("request").at("max_size"));
    const auto rel = fs::u8path(image.at("name").get<std::string>() + ".png");
    const auto dp = output / "depths" / rel, np = output / "normals" / rel;
    if (fs::file_size(dp) != row.at("depth_size").get<uint64_t>() ||
        fs::file_size(np) != row.at("normal_size").get<uint64_t>())
      throw std::runtime_error("LAS supervision file size differs");
    if (row.at("width") != camera.width || row.at("height") != camera.height)
      throw std::runtime_error("LAS supervision map dimensions or pixel types differ");
    if (checks.size() >= Lookahead)
      finish();
    checks.push_back(std::async(std::launch::async, [=, depths = row.at("depth_pixels").get<uint64_t>(),
                                                     normals = row.at("normal_pixels").get<uint64_t>()] {
      const auto depth = ReadImage(dp, cv::IMREAD_UNCHANGED),
                 normal = ReadImage(np, cv::IMREAD_UNCHANGED);
      if (depth.type() != CV_16UC1 || normal.type() != CV_8UC3 || depth.size() != normal.size() ||
          depth.cols != camera.width || depth.rows != camera.height)
        throw std::runtime_error("LAS supervision map dimensions or pixel types differ");
      const auto counts = Counts(depth, normal, !monocular);
      if (counts.first != depths || counts.second != normals)
        throw std::runtime_error("LAS supervision valid pixel counts differ");
      return counts;
    }));
  }
  while (!checks.empty())
    finish();
  if (!depth_total || !normal_total || depth_total != summary.at("depth_pixels").get<uint64_t>() ||
      normal_total != summary.at("normal_pixels").get<uint64_t>())
    throw std::runtime_error("LAS supervision totals differ");
}
}  // namespace geosun

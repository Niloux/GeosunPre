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
constexpr const char* SupervisionSource = "moge2_ssi";
constexpr const char* DepthSource = "moge2_ssi";
constexpr const char* DepthAlignment = "las_face_bilateral_v1";
constexpr const char* NormalSource = "moge2_normals";
constexpr const char* DepthFormat = "uint16_png_ssi";
constexpr const char* DepthType = "ray_distance";
constexpr const char* NormalFrame = "opencv_camera";
// SSI encoding maps valid [min_d, max_d] linearly onto [0.02, 1.0] * 65535 -> >= 1311.
constexpr uint16_t MinValidSsiDepth = 1311;

using namespace supervision_gpu;
constexpr size_t DepthBudget = 1024 * 1024 * 1024;
constexpr size_t PointChunk = 500000;
// Mask decoding, PNG encoding and read-back run beside the GPU with this many images in flight.
constexpr size_t Lookahead = 6;
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
  uint64_t count_ = 0;
  size_t slabs_ = 0;
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
          if (!KeepLasPoint(point))
            continue;
          if (reach_bounds && !Contains(*reach_bounds, point.raw))
            continue;
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
    const size_t budget = std::min(size_t(cache_mb) * 1024 * 1024, free / 4);
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
    if (size_t(count_) * sizeof(Surfel) <= budget)
      resident_ = std::make_unique<DeviceBuffer>(size_t(count_) * sizeof(Surfel));
    std::ofstream surfels;
    if (!resident_) {
      file_ = output / "las-surfels.cache";
      surfels.open(file_, std::ios::binary);
      if (!surfels)
        throw std::runtime_error("Cannot create LAS surfel cache");
    }
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
    Log("LAS surfels: " + std::to_string(count_) + " points, " + std::to_string(stats_.normals) +
        " with normals, " + std::to_string(slabs_) + " slabs, " +
        (resident_ ? "GPU cache" : "compact file with double-buffered upload"));
  }
  cudaStream_t StreamHandle() const { return compute_.value; }
  void ProjectNearest(const Projection* cameras, const float3* rays, float* nearest, int width,
                      int height, size_t images, const RawBounds* batch_bounds = nullptr) {
    std::ifstream packed;
    if (!resident_) {
      packed.open(file_, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS surfel cache");
    }
    size_t index = 0;
    uint64_t file_offset = 0;
    for (const auto& chunk : chunks_) {
      if (batch_bounds && !Overlaps(chunk.bounds, *batch_bounds))
        continue;
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
      }
      supervision_gpu::ProjectNearest(points, chunk.count, images, cameras, rays, nearest, width,
                                      height, compute_.value);
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
cv::Mat AllowedMask(const cv::Mat& static_mask, int boundary) {
  const cv::Mat invalid = (static_mask == 0);
  cv::Mat dilated;
  const cv::Mat kernel = cv::Mat::ones(2 * boundary + 1, 2 * boundary + 1, CV_8UC1);
  cv::dilate(invalid, dilated, kernel, {-1, -1}, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
  return dilated == 0;
}
std::pair<uint64_t, uint64_t> Counts(const cv::Mat& depth, const cv::Mat& normal) {
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
      if (has_depth && d[x] < MinValidSsiDepth)
        throw std::runtime_error("SSI depth pixel outside [0.02, 1.0] * 65535 range");
      if (has_normal) {
        if (!has_depth)
          throw std::runtime_error("Normal has no corresponding valid SSI depth");
        const double norm = squares[n[0]] + squares[n[1]] + squares[n[2]];
        if (std::abs(norm - 1) > .025)
          throw std::runtime_error("Nonunit encoded normal");
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
       args.values.count("--supervision-models")))
    throw std::runtime_error("Supervision options cannot be used with --no-las-supervision");
  if (!options.mask_dir.empty() && !fs::is_directory(options.mask_dir))
    throw std::runtime_error("Geometry exclusion mask directory does not exist");
  if (!options.enabled)
    return options;
  if (!args.Get("--supervision-models").empty())
    options.geometry_models = args.Path("--supervision-models");
  return options;
}
void ResolveSupervisionModels(SupervisionOptions& options, const fs::path& package) {
  if (!options.enabled)
    return;
  // Without --supervision-models, the model package of the reconstruction.
  if (options.geometry_models.empty())
    options.geometry_models = package;
  if (!ReadJson(options.geometry_models / "manifest.json").at("models").contains("geometry"))
    throw std::runtime_error(
        "Supervision needs the geometry engine: " + options.geometry_models.u8string() +
        "; prepare it with run.ps1 prepare-models or pass --supervision-models");
}
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& masks_dir,
                        const SupervisionOptions& options, bool rolling) {
  if (!options.enabled)
    return nullptr;
  Json masks = Json::object(), static_masks = Json::object();
  for (const auto& image : scene.images) {
    if (!model.images[image.id].registered)
      continue;
    static_masks[image.name] = FileMetadata(StaticMaskPath(masks_dir, image.id));
    if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, image.name)))
      masks[image.name] = FileMetadata(ExtraMask(options, image.name));
  }
  Json request = {{"source", SupervisionSource},
                  {"depth_source", DepthSource},
                  {"depth_alignment", DepthAlignment},
                  {"normal_source", NormalSource},
                  {"max_size", options.max_size},
                  {"boundary_pixels", options.boundary},
                  {"exclusion_mask_dir", options.mask_dir.u8string()},
                  {"las_file", FileMetadata(scene.las)},
                  {"static_masks", static_masks},
                  {"exclusion_masks", masks},
                  {"rolling_shutter", rolling},
                  {"geometry", GeometryIdentity(options.geometry_models)}};
  if (options.crop)
    request["crop_windows"] = {(*options.crop)[0].ToJson(), (*options.crop)[1].ToJson()};
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
Json ExportSupervision(const Scene& scene, const Reconstruction& model, const fs::path& masks_dir,
                       const fs::path& output, const SupervisionOptions& options,
                       const ShutterMotion* motion, LasSurfelCache* surfel_cache) {
  const auto started = Clock::now();
  const auto request = SupervisionRequest(scene, model, masks_dir, options, motion != nullptr);

  struct LasScratchGuard {
    fs::path dir;
    void Cleanup() {
      if (dir.empty())
        return;
      std::error_code ec;
      fs::remove_all(dir, ec);
      dir.clear();
    }
    ~LasScratchGuard() { Cleanup(); }
  };
  const fs::path scratch_dir = output / ".las_scratch";
  fs::create_directories(scratch_dir);
  LasScratchGuard scratch_guard{scratch_dir};

  struct LasDepthRecord {
    uint64_t offset = 0;
    int width = 0, height = 0;
    bool valid = false;
  };
  std::vector<LasDepthRecord> las_records(model.images.size());
  const fs::path las_depths_path = scratch_dir / "las_depths.u16";
  double las_project_seconds = 0.0;
  {
    const auto las_started = Clock::now();
    const auto header = ReadSceneLas(scene);
    std::unique_ptr<LasSurfelCache> owned_cache;
    LasSurfelCache* active_cache = surfel_cache;
    if (!active_cache) {
      owned_cache = std::make_unique<LasSurfelCache>(scene.las, header, scratch_dir,
                                                     options.cache_mb, &model);
      active_cache = owned_cache.get();
    }
    std::ofstream out(las_depths_path, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot create temporary LAS depth cache");
    uint64_t offset = 0;
    std::vector<uint16_t> buffer;
    ProjectLasDepths(*active_cache, header, model, motion, options.max_size,
                     [&](uint32_t id, cv::Mat&& depth) {
                       if (depth.type() != CV_32FC1)
                         throw std::runtime_error("Unexpected projected LAS depth map format");
                       const int mw = depth.cols, mh = depth.rows;
                       const size_t n = size_t(mw) * size_t(mh);
                       buffer.resize(n);
                       const cv::Mat contig = depth.isContinuous() ? depth : depth.clone();
                       const float* src = contig.ptr<float>();
                       for (size_t i = 0; i < n; ++i) {
                         const float d = src[i];
                         buffer[i] = d > 0.f && std::isfinite(d)
                                         ? uint16_t(std::clamp(std::lround(double(d) * 1000.0),
                                                               1L, 65535L))
                                         : uint16_t(0);
                       }
                       out.write(reinterpret_cast<const char*>(buffer.data()),
                                 std::streamsize(n * sizeof(uint16_t)));
                       if (!out)
                         throw std::runtime_error("Cannot write temporary LAS depth cache");
                       las_records[id] = {offset, mw, mh, true};
                       offset += uint64_t(n * sizeof(uint16_t));
                     });
    out.close();
    if (!out)
      throw std::runtime_error("Incomplete temporary LAS depth cache write");
    las_project_seconds = Seconds(las_started);
    // Release LAS GPU buffers and temporary files before TensorRT loads the geometry engine.
    active_cache->Finish();
    owned_cache.reset();
    Log("Supervision LAS depths projected in " + std::to_string(las_project_seconds) + " s");
  }

  std::vector<Json> rows(model.images.size());
  auto geometry = std::make_unique<Models>(options.geometry_models);
  auto estimator = std::make_unique<MonocularEstimator>(*geometry);
  Json lenses = Json::array();
  uint64_t depth_pixels = 0, normal_pixels = 0;
  size_t done = 0;
  struct Written {
    uint32_t id;
    Json row;
  };
  std::deque<std::future<Written>> writes;
  double estimate_seconds = 0, write_wait_seconds = 0, load_wait_seconds = 0;
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
      Log("Supervision: " + std::to_string(done) + "/" + std::to_string(model.Registered()) +
          " images, " + std::to_string(Seconds(started)) + " s");
  };
  struct LoadedInput {
    cv::Mat allowed;
    int static_pixels = 0;
    cv::Mat prefiltered;
    cv::Mat las_depth;
  };
  // Lens plans outlive their camera's loop: the CPU blend/write tasks run concurrently.
  std::array<std::optional<MonocularLens>, 2> lens_plans;
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
    const int w = camera.width, h = camera.height;
    auto& lens = lens_plans.at(cid);
    lens = PlanMonocularLens(model.cameras[cid], camera, estimator->FaceSize(), FaceHalfDegrees);
    lenses.push_back({{"camera", cid}, {"plan", lens->ToJson()}});

    std::deque<std::future<LoadedInput>> inputs;
    struct DrainInputs {
      std::deque<std::future<LoadedInput>>& queue;
      ~DrainInputs() {
        for (auto& item : queue)
          if (item.valid())
            item.wait();
      }
    } drain_inputs{inputs};
    size_t requested = 0;
    auto load_inputs = [&] {
      for (; inputs.size() < Lookahead && requested < images.size(); ++requested)
        inputs.push_back(std::async(std::launch::async, [&, id = images[requested]] {
          const auto static_mask = StaticMask(scene, masks_dir, id, camera, options);
          const int static_count = cv::countNonZero(static_mask);
          cv::Mat allowed = AllowedMask(static_mask, options.boundary);
          auto photo = ReadImage(scene.images[id].source, cv::IMREAD_COLOR);
          if (options.crop) {
            const auto& window = (*options.crop)[cid];
            photo = photo(cv::Rect(window.x, window.y, window.width, window.height)).clone();
          }
          cv::cvtColor(photo, photo, cv::COLOR_BGR2RGB);
          cv::Mat las_depth;
          const auto& rec = las_records[id];
          if (rec.valid) {
            if (rec.width != w || rec.height != h)
              throw std::runtime_error(
                  "Projected LAS depth dimensions differ from supervision grid");
            std::ifstream in(las_depths_path, std::ios::binary);
            if (!in)
              throw std::runtime_error("Cannot read temporary LAS depth cache");
            in.seekg(std::streamoff(rec.offset), std::ios::beg);
            const size_t n = size_t(rec.width) * size_t(rec.height);
            std::vector<uint16_t> raw(n);
            if (!in.read(reinterpret_cast<char*>(raw.data()),
                         std::streamsize(n * sizeof(uint16_t))))
              throw std::runtime_error("Truncated temporary LAS depth cache");
            las_depth = cv::Mat(rec.height, rec.width, CV_32FC1);
            float* dst = las_depth.ptr<float>();
            for (size_t k = 0; k < n; ++k)
              dst[k] = raw[k] ? float(raw[k]) * 0.001f : 0.f;
          }
          return LoadedInput{std::move(allowed), static_count, lens->Prefilter(photo),
                             std::move(las_depth)};
        }));
    };
    load_inputs();
    for (size_t idx = 0; idx < images.size(); ++idx) {
      const uint32_t id = images[idx];
      const auto loading = Clock::now();
      auto loaded = inputs.front().get();
      load_wait_seconds += Seconds(loading);
      inputs.pop_front();
      load_inputs();

      const bool rolling_moge = motion != nullptr;
      const Eigen::Vector3d omega = motion ? motion->rotation[id] : Eigen::Vector3d::Zero();
      const int axis = motion ? motion->axis : 1;
      const auto estimating = Clock::now();
      auto estimates = estimator->Estimate(*lens, loaded.prefiltered, omega, axis, rolling_moge);
      estimate_seconds += Seconds(estimating);

      const MonocularLens* plan = &*lens;
      const auto& name = scene.images[id].name;
      const auto dp = output / "depths" / fs::u8path(name + ".png"),
                 np = output / "normals" / fs::u8path(name + ".png");
      fs::create_directories(dp.parent_path());
      fs::create_directories(np.parent_path());
      if (writes.size() >= Lookahead)
        finish();
      writes.push_back(std::async(
          std::launch::async,
          [=, allowed = std::move(loaded.allowed), estimates = std::move(estimates),
           las_depth = std::move(loaded.las_depth), static_pixels = loaded.static_pixels]() mutable {
            const Eigen::Vector3d* omega_ptr = rolling_moge ? &omega : nullptr;
            const cv::Mat* las_ptr = las_depth.empty() ? nullptr : &las_depth;
            auto ssi = EncodeMogeSsiMaps(*plan, estimates, allowed, omega_ptr, axis, las_ptr);
            const auto valid = Counts(ssi.depth_u16, ssi.normal_rgb);
            WriteImage(dp, ssi.depth_u16);
            WriteImage(np, ssi.normal_rgb);
            Json row = {{"image_index", id},
                        {"name", name},
                        {"width", w},
                        {"height", h},
                        {"depth_size", fs::file_size(dp)},
                        {"normal_size", fs::file_size(np)},
                        {"static_pixels", static_pixels},
                        {"depth_pixels", valid.first},
                        {"normal_pixels", valid.second},
                        {"monocular", std::move(ssi.report)}};
            return Written{id, std::move(row)};
          }));
    }
  }
  while (!writes.empty())
    finish();
  scratch_guard.Cleanup();
  const Json geometry_profile = geometry->Profile();
  estimator.reset();
  geometry.reset();

  if (!depth_pixels || !normal_pixels)
    throw std::runtime_error("MoGe-2 SSI supervision has no valid depth or normal pixels");
  Json images = Json::array();
  for (const auto& row : rows)
    if (!row.is_null())
      images.push_back(row);
  if (request != SupervisionRequest(scene, model, masks_dir, options, motion != nullptr))
    throw std::runtime_error("Static or geometry exclusion masks changed during supervision export");
  return {{"request", request},
          {"source", SupervisionSource},
          {"depth_source", DepthSource},
          {"depth_alignment", DepthAlignment},
          {"normal_source", NormalSource},
          {"projection", motion ? "rolling_shutter_readout_line" : "global_shutter"},
          {"depth_format", DepthFormat},
          {"depth_type", DepthType},
          {"depth_encoding", "linear_[min_d,max_d]_to_[0.02,1.0]*65535"},
          {"invalid_depth", 0},
          {"normal_format", "rgb8_png"},
          {"normal_frame", NormalFrame},
          {"normal_encoding", "round(127.5 * (n + 1))"},
          {"invalid_normal_rgb", {0, 0, 0}},
          {"normal_estimator", motion ? "moge2_faces_readout_line" : "moge2_faces"},
          {"semantic_margin_pixels", options.boundary},
          {"geometry_model", request.at("geometry")},
          {"lenses", lenses},
          {"depth_pixels", depth_pixels},
          {"normal_pixels", normal_pixels},
          {"images", images},
          {"inference", geometry_profile},
          {"timing",
           {{"las_project_seconds", las_project_seconds},
            {"moge_estimate_seconds", estimate_seconds},
            {"photo_and_mask_wait_seconds", load_wait_seconds},
            {"result_write_wait_seconds", write_wait_seconds}}},
          {"seconds", Seconds(started)}};
}
void CheckSupervision(const fs::path& output, const Reconstruction& model, const Json& photos,
                      const Json& summary) {
  if (summary.value("source", std::string()) != SupervisionSource ||
      summary.value("depth_source", std::string()) != DepthSource ||
      summary.value("normal_source", std::string()) != NormalSource ||
      summary.value("depth_type", std::string()) != DepthType ||
      summary.value("depth_format", std::string()) != DepthFormat ||
      summary.value("normal_format", std::string()) != "rgb8_png" ||
      summary.value("normal_frame", std::string()) != NormalFrame ||
      summary.at("images").size() != photos.size())
    throw std::runtime_error("Unexpected MoGe-2 SSI supervision contract");
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
      throw std::runtime_error("Supervision image identity differs");
    const auto camera = MapCamera(model.cameras[image.at("camera").get<size_t>()],
                                  summary.at("request").at("max_size"));
    const auto rel = fs::u8path(image.at("name").get<std::string>() + ".png");
    const auto dp = output / "depths" / rel, np = output / "normals" / rel;
    if (fs::file_size(dp) != row.at("depth_size").get<uint64_t>() ||
        fs::file_size(np) != row.at("normal_size").get<uint64_t>())
      throw std::runtime_error("Supervision file size differs");
    if (row.at("width") != camera.width || row.at("height") != camera.height)
      throw std::runtime_error("Supervision map dimensions differ");
    if (checks.size() >= Lookahead)
      finish();
    checks.push_back(std::async(
        std::launch::async, [=, depths = row.at("depth_pixels").get<uint64_t>(),
                             normals = row.at("normal_pixels").get<uint64_t>()] {
          const auto depth = ReadImage(dp, cv::IMREAD_UNCHANGED),
                     normal = ReadImage(np, cv::IMREAD_UNCHANGED);
          if (depth.type() != CV_16UC1 || normal.type() != CV_8UC3 ||
              depth.size() != normal.size() || depth.cols != camera.width ||
              depth.rows != camera.height)
            throw std::runtime_error("Supervision map dimensions or pixel types differ");
          const auto counts = Counts(depth, normal);
          if (counts.first != depths || counts.second != normals)
            throw std::runtime_error("Supervision valid pixel counts differ");
          return counts;
        }));
  }
  while (!checks.empty())
    finish();
  if (!depth_total || !normal_total || depth_total != summary.at("depth_pixels").get<uint64_t>() ||
      normal_total != summary.at("normal_pixels").get<uint64_t>())
    throw std::runtime_error("Supervision totals differ");
}
}  // namespace geosun

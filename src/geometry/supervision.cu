#define EIGEN_NO_CUDA
#include "geometry/supervision.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "io/las.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <future>
#include <numeric>

namespace geosun {
namespace {
constexpr int Version = 2;
constexpr double MaxDepth = 65.535, MinIncidence = .3;
// Half of FieldOfViewDegrees in radians, kept as the published constant.
constexpr double MaxTheta = 1.396263401595464;
constexpr size_t DepthBudget = 256 * 1024 * 1024;
constexpr size_t PointChunk = 500000;
// Mask decoding, PNG encoding and read-back run beside the GPU with this many images in flight.
constexpr size_t Lookahead = 6;
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
  int3* data = nullptr;
  PinnedPoints() {
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), PointChunk * sizeof(int3)));
  }
  ~PinnedPoints() { cudaFreeHost(data); }
};
struct PointTransfer {
  PinnedPoints host;
  DeviceBuffer device;
  Event copied, consumed;
  bool used = false;
  explicit PointTransfer(bool streaming) : device(streaming ? PointChunk * sizeof(int3) : 0) {}
};
// motion holds the readout rotation and the readout translation (readout time times velocity)
// of a rolling-shutter image, both in the reference camera frame; see ShutterMotion.
struct Projection {
  double rotation[9], translation[3], params[8], scales[3], motion[6];
  int rolling, axis;
};
__device__ void FisheyePixel(double x, double y, double z, const double* p, double& u, double& v) {
  const double r = hypot(x, y), theta = atan2(r, z), tt = theta * theta;
  const double radial = theta * (1 + tt * (p[4] + tt * (p[5] + tt * (p[6] + tt * p[7])))),
               scale = r > 1e-15 ? radial / r : 0;
  u = p[0] * x * scale + p[2];
  v = p[1] * y * scale + p[3];
}
__global__ void ProjectLas(const int3* points, int count, const Projection* cameras,
                            float* maps, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto& c = cameras[blockIdx.y];
  const auto raw = points[i];
  const double x = raw.x * c.scales[0], y = raw.y * c.scales[1], z = raw.z * c.scales[2];
  double px = c.rotation[0] * x + c.rotation[1] * y + c.rotation[2] * z + c.translation[0],
         py = c.rotation[3] * x + c.rotation[4] * y + c.rotation[5] * z + c.translation[1],
         pz = c.rotation[6] * x + c.rotation[7] * y + c.rotation[8] * z + c.translation[2];
  if (pz <= 0)
    return;
  if (c.rolling) {
    // The pixel's readout line sets its camera; a few fixed-point steps settle the line.
    const double qx = px, qy = py, qz = pz;
    for (int k = 0; k < 3; ++k) {
      double u, v;
      FisheyePixel(px, py, pz, c.params, u, v);
      const double s = c.axis ? (v - .5 * height) / height : (u - .5 * width) / width;
      const double bx = qx - s * c.motion[3], by = qy - s * c.motion[4], bz = qz - s * c.motion[5];
      const double wx = s * c.motion[0], wy = s * c.motion[1], wz = s * c.motion[2];
      px = bx + wy * bz - wz * by;
      py = by + wz * bx - wx * bz;
      pz = bz + wx * by - wy * bx;
      if (pz <= 0)
        return;
    }
  }
  const double r = hypot(px, py), theta = atan2(r, pz), distance = sqrt(r * r + pz * pz);
  if (theta > MaxTheta || distance < .05 || distance > MaxDepth)
    return;
  const double tt = theta * theta;
  const auto* p = c.params;
  const double radial = theta * (1 + tt * (p[4] + tt * (p[5] + tt * (p[6] + tt * p[7])))),
               scale = r > 1e-15 ? radial / r : 0,
               u = p[0] * px * scale + p[2], v = p[1] * py * scale + p[3];
  if (u >= 0 && v >= 0 && u < width && v < height) {
    const size_t pixel = size_t(blockIdx.y) * width * height + int(v) * width + int(u);
    atomicMin(reinterpret_cast<unsigned*>(maps + pixel), __float_as_uint(float(distance)));
  }
}
class LasPointCache {
  std::unique_ptr<DeviceBuffer> resident_;
  std::array<std::unique_ptr<PointTransfer>, 2> transfers_;
  // Drain both streams before releasing any buffers, including constructor failure.
  Stream compute_, upload_;
  fs::path file_;
  uint64_t count_ = 0, excluded_ = 0, uploaded_bytes_ = 0;
  size_t budget_ = 0, batches_ = 0;
  double prepare_seconds_ = 0;

 public:
  LasPointCache(const fs::path& las, const LasHeader& header, const fs::path& output,
                int cache_mb) {
    const auto started = Clock::now();
    size_t free, total;
    CudaCheck(cudaMemGetInfo(&free, &total));
    budget_ = std::min(size_t(cache_mb) * 1024 * 1024, free / 4);
    const size_t maximum_bytes = size_t(header.count) * sizeof(int3);
    std::ofstream packed;
    if (maximum_bytes <= budget_)
      resident_ = std::make_unique<DeviceBuffer>(maximum_bytes);
    else {
      file_ = output / "las-coordinates.cache";
      packed.open(file_, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot create LAS coordinate cache");
    }
    for (auto& slot : transfers_)
      slot = std::make_unique<PointTransfer>(!resident_);
    size_t index = 0;
    ScanLas(las, header, [&](uint64_t, const std::vector<LasPoint>& chunk) {
      auto& slot = *transfers_[index++ % transfers_.size()];
      if (slot.used)
        CudaCheck(cudaEventSynchronize(slot.copied.value));
      size_t kept = 0;
      for (const auto& point : chunk) {
        if (KeepLasPoint(point))
          slot.host.data[kept++] = make_int3(point.raw[0], point.raw[1], point.raw[2]);
        else
          ++excluded_;
      }
      const size_t bytes = kept * sizeof(int3);
      if (resident_ && bytes) {
        auto* destination = static_cast<int3*>(resident_->data) + count_;
        CudaCheck(cudaMemcpyAsync(destination, slot.host.data, bytes, cudaMemcpyHostToDevice,
                                  upload_.value));
        CudaCheck(cudaEventRecord(slot.copied.value, upload_.value));
        slot.used = true;
        uploaded_bytes_ += bytes;
      } else if (!resident_) {
        packed.write(reinterpret_cast<const char*>(slot.host.data), std::streamsize(bytes));
        if (!packed)
          throw std::runtime_error("Cannot write LAS coordinate cache");
      }
      count_ += kept;
    });
    CudaCheck(cudaStreamSynchronize(upload_.value));
    if (resident_) {
      for (auto& slot : transfers_)
        slot.reset();
    } else {
      packed.close();
      if (!packed)
        throw std::runtime_error("Incomplete LAS coordinate cache write");
    }
    prepare_seconds_ = Seconds(started);
    Log("LAS coordinates: " + std::to_string(count_) + " points, " +
        (resident_ ? "GPU cache" : "compact file with double-buffered upload"));
  }
  cudaStream_t StreamHandle() const { return compute_.value; }
  uint64_t Count() const { return count_; }
  uint64_t Excluded() const { return excluded_; }
  void Project(const Projection* cameras, float* maps, int width, int height, size_t images) {
    std::ifstream packed;
    if (!resident_) {
      packed.open(file_, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS coordinate cache");
    }
    size_t index = 0;
    for (uint64_t first = 0; first < count_; first += PointChunk) {
      const size_t count = size_t(std::min<uint64_t>(PointChunk, count_ - first));
      const int3* points;
      PointTransfer* transfer = nullptr;
      if (resident_)
        points = static_cast<const int3*>(resident_->data) + first;
      else {
        transfer = transfers_[index++ % transfers_.size()].get();
        if (transfer->used)
          CudaCheck(cudaEventSynchronize(transfer->copied.value));
        const size_t bytes = count * sizeof(int3);
        if (!packed.read(reinterpret_cast<char*>(transfer->host.data), std::streamsize(bytes)))
          throw std::runtime_error("Truncated LAS coordinate cache");
        if (transfer->used)
          CudaCheck(cudaStreamWaitEvent(upload_.value, transfer->consumed.value, 0));
        CudaCheck(cudaMemcpyAsync(transfer->device.data, transfer->host.data, bytes,
                                  cudaMemcpyHostToDevice, upload_.value));
        CudaCheck(cudaEventRecord(transfer->copied.value, upload_.value));
        CudaCheck(cudaStreamWaitEvent(compute_.value, transfer->copied.value, 0));
        points = static_cast<const int3*>(transfer->device.data);
        uploaded_bytes_ += bytes;
      }
      ProjectLas<<<dim3(unsigned((count + 255) / 256), unsigned(images)), 256, 0, compute_.value>>>(
          points, int(count), cameras, maps, width, height);
      CudaCheck(cudaGetLastError());
      if (transfer) {
        CudaCheck(cudaEventRecord(transfer->consumed.value, compute_.value));
        transfer->used = true;
      }
    }
    ++batches_;
  }
  void Finish() {
    CudaCheck(cudaStreamSynchronize(compute_.value));
    CudaCheck(cudaStreamSynchronize(upload_.value));
    if (!file_.empty())
      fs::remove(file_);
  }
  Json Profile() const {
    return {{"storage", resident_ ? "gpu" : "file"},
            {"las_scans", 1},
            {"budget_bytes", budget_},
            {"coordinate_bytes", count_ * sizeof(int3)},
            {"gpu_cache_bytes", resident_ ? resident_->bytes : 0},
            {"transfer_buffer_bytes", resident_ ? 0 : 2 * PointChunk * sizeof(int3)},
            {"host_to_device_bytes", uploaded_bytes_},
            {"projection_batches", batches_},
            {"prepare_seconds", prepare_seconds_}};
  }
};
__device__ float At(const float* depth, int x, int y, int w, int h) {
  if (x < 0 || y < 0 || x >= w || y >= h)
    return 0;
  const float d = depth[y * w + x];
  return d > 0 && d <= MaxDepth ? d : 0;
}
__global__ void Cull(const float* raw, float* clean, const uint8_t* mask, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float near = float(MaxDepth), d = At(raw, x, y, w, h);
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const float v = At(raw, xx, yy, w, h);
      if (v > 0 && mask[yy * w + xx])
        near = fminf(near, v);
    }
  clean[i] = mask[i] && d <= near * 1.03f + .02f ? d : 0;
}
__global__ void Fill(const float* input, float* output, const uint8_t* mask, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float sum = 0, lo = float(MaxDepth), hi = 0;
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (!dx && !dy)
        continue;
      const float v = At(input, x + dx, y + dy, w, h);
      if (v > 0) {
        ++count;
        sum += v;
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
    }
  const float d = At(input, x, y, w, h);
  output[i] = d;
  if (mask[i] && !d && count >= 6 && hi - lo <= .02f * lo + .01f)
    output[i] = sum / count;
}
__global__ void Edges(const float* depth, const uint8_t* mask, uint8_t* edges, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float lo = float(MaxDepth), hi = 0;
  bool boundary = !mask[i];
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
        boundary = true;
        continue;
      }
      boundary = boundary || !mask[yy * w + xx];
      const float v = At(depth, xx, yy, w, h);
      if (v > 0) {
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
    }
  edges[i] = boundary || hi - lo > .025f * lo + .02f;
}
__global__ void Dilate(const uint8_t* input, uint8_t* output, int w, int h, int radius,
                         bool horizontal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  uint8_t bad = 0;
  for (int d = -radius; d <= radius; ++d) {
    const int x = i % w + (horizontal ? d : 0), y = i / w + (horizontal ? 0 : d);
    if (x >= 0 && y >= 0 && x < w && y < h)
      bad |= input[y * w + x];
  }
  output[i] = bad;
}
__device__ double3 Sub(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ double Dot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ bool DepthSupported(const float* depth, int x, int y, int w, int h, double center) {
  int count = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const float v = At(depth, x + dx, y + dy, w, h);
      if (v > 0 && fabs(v - center) <= .02 * center + .01 && ++count >= 4)
        return true;
    }
  return false;
}
__device__ double3 ReadoutInverse(double3 w, double3 p) {
  const double dot = Dot(w, p), den = 1 + Dot(w, w);
  return make_double3((p.x - w.y * p.z + w.z * p.y + w.x * dot) / den,
                      (p.y - w.z * p.x + w.x * p.z + w.y * dot) / den,
                      (p.z - w.x * p.y + w.y * p.x + w.z * dot) / den);
}
// Same readout fraction as ShutterMotion::Fraction, for the pixel centre of map pixel (x, y).
__device__ double ReadoutFraction(int x, int y, int w, int h, const Projection& c) {
  return c.axis ? (y + .5) / h - .5 : (x + .5) / w - .5;
}
__device__ double3 ReferencePoint(double3 point, int x, int y, int w, int h, const Projection& c) {
  if (!c.rolling)
    return point;
  const double s = ReadoutFraction(x, y, w, h, c);
  auto p = ReadoutInverse(make_double3(s * c.motion[0], s * c.motion[1], s * c.motion[2]), point);
  return make_double3(p.x + s * c.motion[3], p.y + s * c.motion[4], p.z + s * c.motion[5]);
}
__device__ double3 ReferenceRay(double3 ray, int x, int y, int w, int h, const Projection& c) {
  if (!c.rolling)
    return ray;
  const double s = ReadoutFraction(x, y, w, h, c);
  auto p = ReadoutInverse(make_double3(s * c.motion[0], s * c.motion[1], s * c.motion[2]), ray);
  const double length = sqrt(Dot(p, p));
  return make_double3(p.x / length, p.y / length, p.z / length);
}
__device__ bool SurfacePoint(const float* depth, const double3* rays, int x, int y, int w, int h,
                             double distance, double3 center, double3& delta,
                             const Projection& camera) {
  const float d = At(depth, x, y, w, h);
  if (!d || fabs(d - distance) > .02 * distance + .01)
    return false;
  const auto ray = rays[y * w + x];
  if (Dot(ray, ray) < .5)
    return false;
  delta = Sub(ReferencePoint(make_double3(d * ray.x, d * ray.y, d * ray.z), x, y, w, h, camera),
              center);
  return true;
}
__device__ bool Normal(const float* depth, const double3* rays, int x, int y, int w, int h,
                       int radius, double distance, double3& normal, double tolerance,
                       const Projection& camera) {
  const auto source_ray = rays[y * w + x];
  const auto ray = ReferenceRay(source_ray, x, y, w, h, camera);
  const double3 center = ReferencePoint(
      make_double3(distance * source_ray.x, distance * source_ray.y, distance * source_ray.z), x, y,
      w, h, camera);
  const double tangent_length = hypot(ray.x, ray.z);
  const double3 u = make_double3(ray.z / tangent_length, 0, -ray.x / tangent_length),
                v = make_double3(ray.y * u.z, ray.z * u.x - ray.x * u.z, -ray.y * u.x);
  // Fit ray displacement over the tangent plane. Missing samples are omitted rather than
  // requiring complete 3x3 patches at four particular positions.
  double sx = 0, sy = 0, sz = 0, sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
  int count = 0;
  for (int dy = -3; dy <= 3; ++dy)
    for (int dx = -3; dx <= 3; ++dx) {
      double3 delta;
      if (!SurfacePoint(depth, rays, x + dx * radius / 3, y + dy * radius / 3, w, h, distance,
                        center, delta, camera))
        continue;
      const double px = Dot(delta, u), py = Dot(delta, v), pz = Dot(delta, ray);
      sx += px;
      sy += py;
      sz += pz;
      sxx += px * px;
      sxy += px * py;
      syy += py * py;
      sxz += px * pz;
      syz += py * pz;
      ++count;
    }
  if (count < 8)
    return false;
  sxx -= sx * sx / count;
  sxy -= sx * sy / count;
  syy -= sy * sy / count;
  sxz -= sx * sz / count;
  syz -= sy * sz / count;
  const double trace = sxx + syy, det = sxx * syy - sxy * sxy;
  // Reject line-like support, including scan stripes with no surface extent across them.
  if (trace <= 1e-10 || det <= .02 * trace * trace)
    return false;
  const double a = (sxz * syy - syz * sxy) / det,
               b = (syz * sxx - sxz * sxy) / det,
               c = (sz - a * sx - b * sy) / count,
               length = sqrt(1 + a * a + b * b);
  if (fabs(c) > tolerance * length)
    return false;
  normal = make_double3((a * u.x + b * v.x - ray.x) / length,
                        (a * u.y + b * v.y - ray.y) / length,
                        (a * u.z + b * v.z - ray.z) / length);
  for (int dy = -3; dy <= 3; ++dy)
    for (int dx = -3; dx <= 3; ++dx) {
      double3 delta;
      if (SurfacePoint(depth, rays, x + dx * radius / 3, y + dy * radius / 3, w, h, distance,
                       center, delta, camera) &&
          fabs(Dot(normal, delta) + c / length) > tolerance)
        return false;
    }
  return true;
}
__global__ void Encode(const float* depth, const double3* rays, const uint8_t* edges,
                       uint16_t* millimetres, uchar3* normals, int w, int h, double focal,
                       Projection camera) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  millimetres[i] = 0;
  normals[i] = make_uchar3(0, 0, 0);
  const int x = i % w, y = i / w;
  const float d = At(depth, x, y, w, h);
  if (edges[i] || !d || Dot(rays[i], rays[i]) < .5 || !DepthSupported(depth, x, y, w, h, d))
    return;
  millimetres[i] = uint16_t(max(1, min(65535, int(d * 1000. + .5))));
  const int radius = max(3, min(24, int(focal * .04 / d + .5)));
  double3 normal, coarse;
  if (!Normal(depth, rays, x, y, w, h, radius, d, normal, .01, camera) ||
      -Dot(normal, ReferenceRay(rays[i], x, y, w, h, camera)) < MinIncidence)
    return;
  if (!Normal(depth, rays, x, y, w, h, radius * 2, d, coarse, .015, camera) ||
      Dot(normal, coarse) < .984807753012208)
    return;
  if (camera.rolling) {
    const double s = ReadoutFraction(x, y, w, h, camera);
    normal = ReadoutInverse(
        make_double3(-s * camera.motion[0], -s * camera.motion[1], -s * camera.motion[2]), normal);
    const double length = sqrt(Dot(normal, normal));
    normal = make_double3(normal.x / length, normal.y / length, normal.z / length);
  }
  normals[i] = make_uchar3(uint8_t(127.5 + 127.5 * normal.z + .5),
                           uint8_t(127.5 + 127.5 * normal.y + .5),
                           uint8_t(127.5 + 127.5 * normal.x + .5));
}
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
  cv::Mat source, resized;
  mask.convertTo(source, CV_32F, 1. / 255);
  cv::resize(source, resized, {camera.width, camera.height}, 0, 0, cv::INTER_AREA);
  mask = resized >= 1.f - 1e-6f;
  mask /= 255;
  return mask;
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
      if (has_normal) {
        if (!has_depth)
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
}  // namespace
SupervisionOptions ParseSupervisionOptions(const Arguments& args) {
  SupervisionOptions options;
  if (args.values.count("--las-supervision") && args.values.count("--no-las-supervision"))
    throw std::runtime_error("--las-supervision conflicts with --no-las-supervision");
  options.enabled = args.Get("--no-las-supervision") != "true";
  options.max_size = args.Integer("--supervision-max-size", 1600);
  options.boundary = args.Integer("--supervision-boundary", 6);
  options.cache_mb = args.Integer("--supervision-cache-mb", 1024);
  if (!args.Get("--supervision-mask-dir").empty())
    options.mask_dir = args.Path("--supervision-mask-dir");
  if (options.max_size < 128 || options.max_size > 4096 || options.boundary > 32)
    throw std::runtime_error("Supervision max size must be 128-4096 and boundary 1-32 pixels");
  if (options.cache_mb > 4096)
    throw std::runtime_error("Supervision cache budget must be 1-4096 MiB");
  if (!options.enabled &&
      (args.values.count("--supervision-max-size") || args.values.count("--supervision-boundary") ||
       args.values.count("--supervision-cache-mb") || !options.mask_dir.empty()))
    throw std::runtime_error("Supervision options cannot be used with --no-las-supervision");
  if (!options.mask_dir.empty() && !fs::is_directory(options.mask_dir))
    throw std::runtime_error("Geometry exclusion mask directory does not exist");
  return options;
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
  Json request = {{"version", Version},
                  {"max_size", options.max_size},
                  {"boundary_pixels", options.boundary},
                  {"las_file", FileMetadata(scene.las)},
                  {"exclusion_mask_dir", options.mask_dir.u8string()},
                  {"static_masks", static_masks},
                  {"exclusion_masks", masks}};
  if (rolling)
    request["readout_normals_version"] = 1;
  return request;
}
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model,
                          const fs::path& masks_dir, const fs::path& output,
                          const SupervisionOptions& options, const ShutterMotion* motion) {
  const auto started = Clock::now();
  const auto header = ReadLasHeader(scene.las);
  const auto request = SupervisionRequest(scene, model, masks_dir, options, motion != nullptr);
  std::vector<Json> rows(model.images.size());
  LasPointCache points(scene.las, header, output, options.cache_mb);
  const auto stream = points.StreamHandle();
  uint64_t depth_pixels = 0, normal_pixels = 0;
  size_t done = 0;
  struct Written {
    uint32_t id;
    Json row;
  };
  std::deque<std::future<Written>> writes;
  auto finish = [&] {
    auto written = writes.front().get();
    writes.pop_front();
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
  };
  for (uint32_t cid = 0; cid < model.cameras.size(); ++cid) {
    std::vector<uint32_t> images;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        images.push_back(i);
    if (images.empty())
      continue;
    const auto camera = MapCamera(model.cameras[cid], options.max_size);
    const int w = camera.width, h = camera.height, size = w * h, blocks = (size + 255) / 256;
    const size_t batch = std::min<size_t>(32, std::max<size_t>(1, DepthBudget / (size * sizeof(float))));
    std::vector<double3> ray_host(size);
    const double valid_radius = camera.Radius();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = (x + .5 - camera.params[2]) / camera.params[0],
                     dy = (y + .5 - camera.params[3]) / camera.params[1];
        const Eigen::Vector3d ray = dx * dx + dy * dy <= valid_radius * valid_radius
                                       ? camera.Ray({x + .5, y + .5}) : Eigen::Vector3d::Zero();
        ray_host[y * w + x] = make_double3(ray.x(), ray.y(), ray.z());
      }
    DeviceBuffer maps(batch * size * sizeof(float)), cameras(batch * sizeof(Projection)),
        rays(size * sizeof(double3)), clean(size * sizeof(float)), filled(size * sizeof(float)),
        mask(size), edge(size), dilated(size), millimetres(size * sizeof(uint16_t)), normals(size * sizeof(uchar3));
    CudaCheck(
        cudaMemcpyAsync(rays.data, ray_host.data(), rays.bytes, cudaMemcpyHostToDevice, stream));
    // Masks for the following images decode while the GPU projects the current batch.
    std::deque<std::future<LoadedMask>> masks;
    size_t requested = 0;
    auto request = [&] {
      for (; masks.size() < Lookahead && requested < images.size(); ++requested)
        masks.push_back(std::async(std::launch::async, [&, id = images[requested]] {
          auto pixels = StaticMask(scene, masks_dir, id, camera, options);
          const int count = cv::countNonZero(pixels);
          return LoadedMask{std::move(pixels), count};
        }));
    };
    for (size_t first = 0; first < images.size(); first += batch) {
      request();
      const size_t count = std::min(batch, images.size() - first);
      std::vector<Projection> projection(count);
      for (size_t j = 0; j < count; ++j) {
        auto& p = projection[j];
        const auto& pose = model.images[images[first + j]].pose;
        const Eigen::Matrix3d rotation = pose.Rotation().toRotationMatrix();
        const Eigen::Vector3d translation = rotation * header.origin + pose.Translation();
        for (int r = 0; r < 3; ++r) {
          p.translation[r] = translation[r];
          p.scales[r] = header.scale[r];
          for (int c = 0; c < 3; ++c)
            p.rotation[r * 3 + c] = rotation(r, c);
        }
        std::copy(camera.params.begin(), camera.params.end(), p.params);
        if (motion) {
          const uint32_t id = images[first + j];
          const Eigen::Vector3d moved = motion->readout[cid] * motion->velocity[id];
          for (int k = 0; k < 3; ++k) {
            p.motion[k] = motion->rotation[id][k];
            p.motion[3 + k] = moved[k];
          }
          p.rolling = 1;
          p.axis = motion->axis;
        }
      }
      CudaCheck(cudaMemcpyAsync(cameras.data, projection.data(), count * sizeof(Projection),
                                cudaMemcpyHostToDevice, stream));
      CudaCheck(cudaMemsetAsync(maps.data, 0x7f, count * size * sizeof(float), stream));
      points.Project(static_cast<const Projection*>(cameras.data), static_cast<float*>(maps.data),
                     w, h, count);
      for (size_t j = 0; j < count; ++j) {
        const uint32_t id = images[first + j];
        const auto static_mask = masks.front().get();
        masks.pop_front();
        request();
        CudaCheck(cudaMemcpyAsync(mask.data, static_mask.pixels.data, size, cudaMemcpyHostToDevice,
                                  stream));
        const auto* raw_depth = static_cast<float*>(maps.data) + j * size;
        auto* filtered = static_cast<float*>(clean.data);
        auto* scratch = static_cast<float*>(filled.data);
        const auto* keep = static_cast<const uint8_t*>(mask.data);
        Cull<<<blocks, 256, 0, stream>>>(raw_depth, filtered, keep, w, h);
        for (int pass = 0; pass < 2; ++pass) {
          Fill<<<blocks, 256, 0, stream>>>(filtered, scratch, keep, w, h);
          std::swap(filtered, scratch);
        }
        Edges<<<blocks, 256, 0, stream>>>(filtered, keep, static_cast<uint8_t*>(edge.data), w, h);
        Dilate<<<blocks, 256, 0, stream>>>(static_cast<const uint8_t*>(edge.data),
                                           static_cast<uint8_t*>(dilated.data), w, h,
                                           options.boundary, true);
        Dilate<<<blocks, 256, 0, stream>>>(static_cast<const uint8_t*>(dilated.data),
                                           static_cast<uint8_t*>(edge.data), w, h, options.boundary,
                                           false);
        Encode<<<blocks, 256, 0, stream>>>(
            filtered, static_cast<const double3*>(rays.data),
            static_cast<const uint8_t*>(edge.data), static_cast<uint16_t*>(millimetres.data),
            static_cast<uchar3*>(normals.data), w, h, camera.params[0], projection[j]);
        CudaCheck(cudaGetLastError());
        cv::Mat depth(h, w, CV_16UC1), normal(h, w, CV_8UC3);
        CudaCheck(cudaMemcpyAsync(depth.data, millimetres.data, millimetres.bytes,
                                  cudaMemcpyDeviceToHost, stream));
        CudaCheck(cudaMemcpyAsync(normal.data, normals.data, normals.bytes, cudaMemcpyDeviceToHost,
                                  stream));
        CudaCheck(cudaStreamSynchronize(stream));
        const auto& name = scene.images[id].name;
        const auto dp = output / "depths" / fs::u8path(name + ".png"),
                   np = output / "normals" / fs::u8path(name + ".png");
        fs::create_directories(dp.parent_path());
        fs::create_directories(np.parent_path());
        if (writes.size() >= Lookahead)
          finish();
        writes.push_back(std::async(std::launch::async, [=, static_pixels = static_mask.count] {
          const auto valid = Counts(depth, normal);
          WriteImage(dp, depth);
          WriteImage(np, normal);
          return Written{id,
                         {{"image_index", id}, {"name", name}, {"width", w}, {"height", h},
                          {"depth_size", fs::file_size(dp)}, {"normal_size", fs::file_size(np)},
                          {"static_pixels", static_pixels}, {"depth_pixels", valid.first},
                          {"normal_pixels", valid.second}}};
        }));
      }
    }
  }
  while (!writes.empty())
    finish();
  if (!depth_pixels || !normal_pixels)
    throw std::runtime_error("LAS supervision has no reliable depth or normal pixels");
  Json images = Json::array();
  for (const auto& row : rows)
    if (!row.is_null())
      images.push_back(row);
  if (request != SupervisionRequest(scene, model, masks_dir, options, motion != nullptr))
    throw std::runtime_error("LAS or geometry exclusion masks changed during supervision export");
  points.Finish();
  return {{"version", Version},
          {"request", request},
          {"source", "valid original LAS records"},
          {"projection", motion ? "rolling_shutter_readout_line" : "global_shutter"},
          {"accepted_records", points.Count()},
          {"excluded_records", points.Excluded()},
          {"point_cache", points.Profile()},
          {"depth_format", "uint16_png_mm"},
          {"depth_type", "ray_distance"},
          {"invalid_depth", 0},
          {"normal_format", "rgb8_png"},
          {"normal_frame", "opencv_camera_facing_camera"},
          {"normal_encoding", "round(127.5 * (n + 1))"},
          {"invalid_normal_rgb", {0, 0, 0}},
          {"depth_validity", "occlusion_boundary_and_local_depth_support"},
          {"normal_estimator", motion ? "readout_reference_two_scale_tangent_plane_least_squares"
                                      : "two_scale_tangent_plane_least_squares"},
          {"normal_grid_size", 7},
          {"normal_minimum_samples", 8},
          {"minimum_abs_normal_dot_ray", MinIncidence},
          {"surface_radius_m", .04},
          {"normal_angle_limit_degrees", 10},
          {"fine_plane_limit_m", .01},
          {"coarse_plane_limit_m", .015},
          {"depth_pixels", depth_pixels},
          {"normal_pixels", normal_pixels},
          {"images", images},
          {"seconds", Seconds(started)}};
}
void LasRangeMaps(const Scene& scene, const Reconstruction& model, int max_size,
                  const fs::path& scratch, const std::function<void(uint32_t, cv::Mat)>& use) {
  const auto header = ReadLasHeader(scene.las);
  LasPointCache points(scene.las, header, scratch, 1024);
  const auto stream = points.StreamHandle();
  for (uint32_t cid = 0; cid < model.cameras.size(); ++cid) {
    std::vector<uint32_t> images;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        images.push_back(i);
    if (images.empty())
      continue;
    const auto camera = MapCamera(model.cameras[cid], max_size);
    const int w = camera.width, h = camera.height, size = w * h, blocks = (size + 255) / 256;
    const size_t batch =
        std::min<size_t>(64, std::max<size_t>(1, DepthBudget / (size * sizeof(float))));
    DeviceBuffer maps(batch * size * sizeof(float)), cameras(batch * sizeof(Projection)),
        clean(size * sizeof(float)), mask(size);
    CudaCheck(cudaMemsetAsync(mask.data, 1, size, stream));
    for (size_t first = 0; first < images.size(); first += batch) {
      const size_t count = std::min(batch, images.size() - first);
      std::vector<Projection> projection(count);
      for (size_t j = 0; j < count; ++j) {
        auto& p = projection[j];
        const auto& pose = model.images[images[first + j]].pose;
        const Eigen::Matrix3d rotation = pose.Rotation().toRotationMatrix();
        const Eigen::Vector3d translation = rotation * header.origin + pose.Translation();
        for (int r = 0; r < 3; ++r) {
          p.translation[r] = translation[r];
          p.scales[r] = header.scale[r];
          for (int c = 0; c < 3; ++c)
            p.rotation[r * 3 + c] = rotation(r, c);
        }
        std::copy(camera.params.begin(), camera.params.end(), p.params);
      }
      CudaCheck(cudaMemcpyAsync(cameras.data, projection.data(), count * sizeof(Projection),
                                cudaMemcpyHostToDevice, stream));
      CudaCheck(cudaMemsetAsync(maps.data, 0x7f, count * size * sizeof(float), stream));
      points.Project(static_cast<const Projection*>(cameras.data), static_cast<float*>(maps.data),
                     w, h, count);
      for (size_t j = 0; j < count; ++j) {
        Cull<<<blocks, 256, 0, stream>>>(static_cast<const float*>(maps.data) + j * size,
                                         static_cast<float*>(clean.data),
                                         static_cast<const uint8_t*>(mask.data), w, h);
        CudaCheck(cudaGetLastError());
        cv::Mat range(h, w, CV_32FC1);
        CudaCheck(
            cudaMemcpyAsync(range.data, clean.data, clean.bytes, cudaMemcpyDeviceToHost, stream));
        CudaCheck(cudaStreamSynchronize(stream));
        use(images[first + j], std::move(range));
      }
    }
  }
  points.Finish();
}
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary) {
  if ((summary.at("version") != 1 && summary.at("version") != Version) ||
      summary.at("depth_type") != "ray_distance" ||
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
      const auto counts = Counts(depth, normal);
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

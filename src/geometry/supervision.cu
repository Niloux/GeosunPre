#define EIGEN_NO_CUDA
#include "geometry/supervision.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "io/las.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace geosun {
namespace {
constexpr int Version = 1;
constexpr double MaxDepth = 65.535, MinIncidence = .3;
constexpr size_t DepthBudget = 256 * 1024 * 1024;
struct Projection {
  double rotation[9], translation[3], params[8], scales[3];
};
__global__ void ProjectLas(const int3* points, int count, const Projection* cameras,
                            float* maps, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto& c = cameras[blockIdx.y];
  const auto raw = points[i];
  const double x = raw.x * c.scales[0], y = raw.y * c.scales[1], z = raw.z * c.scales[2];
  const double px = c.rotation[0] * x + c.rotation[1] * y + c.rotation[2] * z + c.translation[0],
               py = c.rotation[3] * x + c.rotation[4] * y + c.rotation[5] * z + c.translation[1],
               pz = c.rotation[6] * x + c.rotation[7] * y + c.rotation[8] * z + c.translation[2];
  if (pz <= 0)
    return;
  const double r = hypot(px, py), theta = atan2(r, pz), distance = sqrt(r * r + pz * pz);
  if (theta > 1.396263401595464 || distance < .05 || distance > MaxDepth)
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
__device__ bool Point(const float* depth, const double3* rays, int x, int y, int w, int h,
                        double3& point) {
  const float center = At(depth, x, y, w, h);
  if (!center)
    return false;
  point = make_double3(0, 0, 0);
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const float v = At(depth, xx, yy, w, h);
      if (v > 0 && fabs(v - center) <= .02 * center + .01) {
        const auto ray = rays[yy * w + xx];
        point.x += v * ray.x;
        point.y += v * ray.y;
        point.z += v * ray.z;
        ++count;
      }
    }
  if (count < 6)
    return false;
  point.x /= count;
  point.y /= count;
  point.z /= count;
  return true;
}
__device__ bool Normal(const float* depth, const double3* rays, int x, int y, int w, int h,
                         int radius, double3 center, double3& normal, double tolerance) {
  double3 p[4];
  if (!Point(depth, rays, x - radius, y, w, h, p[0]) ||
      !Point(depth, rays, x + radius, y, w, h, p[1]) ||
      !Point(depth, rays, x, y - radius, w, h, p[2]) ||
      !Point(depth, rays, x, y + radius, w, h, p[3]))
    return false;
  const auto dx = Sub(p[1], p[0]), dy = Sub(p[2], p[3]);
  normal = make_double3(dx.y * dy.z - dx.z * dy.y, dx.z * dy.x - dx.x * dy.z,
                        dx.x * dy.y - dx.y * dy.x);
  const double length = sqrt(Dot(normal, normal));
  if (length < 1e-12)
    return false;
  const double factor = (Dot(normal, center) > 0 ? -1. : 1.) / length;
  normal.x *= factor;
  normal.y *= factor;
  normal.z *= factor;
  for (int j = 0; j < 4; ++j)
    if (fabs(Dot(normal, Sub(p[j], center))) > tolerance)
      return false;
  return true;
}
__global__ void Encode(const float* depth, const double3* rays, const uint8_t* edges,
                         uint16_t* millimetres, uchar3* normals, int w, int h, double focal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  millimetres[i] = 0;
  normals[i] = make_uchar3(0, 0, 0);
  const int x = i % w, y = i / w;
  const float d = At(depth, x, y, w, h);
  if (edges[i] || !d)
    return;
  const int radius = max(2, min(24, int(focal * .04 / d + .5)));
  double3 center, normal, coarse;
  if (!Point(depth, rays, x, y, w, h, center) ||
      !Normal(depth, rays, x, y, w, h, radius, center, normal, .01) ||
      -Dot(normal, rays[i]) < MinIncidence)
    return;
  millimetres[i] = uint16_t(max(1, min(65535, int(d * 1000. + .5))));
  if (!Normal(depth, rays, x, y, w, h, radius * 2, center, coarse, .015) ||
      Dot(normal, coarse) < .984807753012208)
    return;
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
cv::Mat StaticMask(const Scene& scene, const fs::path& work, uint32_t image,
                    const Camera& camera, const SupervisionOptions& options) {
  auto mask = ReadImage(MaskPath(work, image), cv::IMREAD_GRAYSCALE);
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
  uint64_t depths = 0, normals = 0;
  for (int y = 0; y < depth.rows; ++y)
    for (int x = 0; x < depth.cols; ++x) {
      const bool has_depth = depth.at<uint16_t>(y, x) != 0;
      const auto n = normal.at<cv::Vec3b>(y, x);
      const bool has_normal = n != cv::Vec3b(0, 0, 0);
      if (has_normal) {
        if (!has_depth)
          throw std::runtime_error("Normal has no corresponding valid LAS depth");
        double norm = 0;
        for (int j = 0; j < 3; ++j)
          norm += std::pow(n[j] / 127.5 - 1, 2);
        if (std::abs(norm - 1) > .025)
          throw std::runtime_error("Nonunit encoded LAS normal");
      }
      depths += has_depth;
      normals += has_normal;
    }
  return {depths, normals};
}
}  // namespace
SupervisionOptions ParseSupervisionOptions(const Arguments& args) {
  SupervisionOptions options;
  options.enabled = args.Get("--las-supervision") == "true";
  options.max_size = args.Integer("--supervision-max-size", 1600);
  options.boundary = args.Integer("--supervision-boundary", 6);
  if (!args.Get("--supervision-mask-dir").empty())
    options.mask_dir = args.Path("--supervision-mask-dir");
  if (options.max_size < 128 || options.max_size > 4096 || options.boundary > 32)
    throw std::runtime_error("Supervision max size must be 128-4096 and boundary 1-32 pixels");
  if (!options.enabled && (args.values.count("--supervision-max-size") ||
                           args.values.count("--supervision-boundary") || !options.mask_dir.empty()))
    throw std::runtime_error("Supervision options require --las-supervision");
  if (!options.mask_dir.empty() && !fs::is_directory(options.mask_dir))
    throw std::runtime_error("Geometry exclusion mask directory does not exist");
  return options;
}
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& work,
                        const SupervisionOptions& options) {
  if (!options.enabled)
    return nullptr;
  if (FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed since reconstruction; choose a new work directory");
  Json masks = Json::object(), static_masks = Json::object();
  for (const auto& image : scene.images) {
    if (!model.images[image.id].registered)
      continue;
    static_masks[image.name] = FileMetadata(MaskPath(work, image.id));
    if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, image.name)))
      masks[image.name] = FileMetadata(ExtraMask(options, image.name));
  }
  return {{"version", Version}, {"max_size", options.max_size}, {"boundary_pixels", options.boundary},
          {"las_file", FileMetadata(scene.las)}, {"exclusion_mask_dir", options.mask_dir.u8string()},
          {"static_masks", static_masks}, {"exclusion_masks", masks}};
}
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model,
                          const fs::path& work, const fs::path& output,
                          const SupervisionOptions& options) {
  const auto started = Clock::now();
  const auto header = ReadLasHeader(scene.las);
  const auto request = SupervisionRequest(scene, model, work, options);
  std::vector<Json> rows(model.images.size());
  DeviceBuffer points(500000 * sizeof(int3));
  std::vector<int3> raw;
  raw.reserve(500000);
  uint64_t accepted = 0, rejected = 0, depth_pixels = 0, normal_pixels = 0;
  size_t done = 0;
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
    CudaCheck(cudaMemcpy(rays.data, ray_host.data(), rays.bytes, cudaMemcpyHostToDevice));
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
      CudaCheck(cudaMemcpy(cameras.data, projection.data(), count * sizeof(Projection), cudaMemcpyHostToDevice));
      CudaCheck(cudaMemset(maps.data, 0x7f, count * size * sizeof(float)));
      uint64_t kept = 0, skipped = 0;
      ScanLas(scene.las, header, [&](uint64_t, const std::vector<LasPoint>& chunk) {
        raw.clear();
        for (const auto& p : chunk) {
          if (KeepLasPoint(p))
            raw.push_back(make_int3(p.raw[0], p.raw[1], p.raw[2]));
          else
            ++skipped;
        }
        kept += raw.size();
        if (raw.empty())
          return;
        CudaCheck(cudaMemcpy(points.data, raw.data(), raw.size() * sizeof(int3), cudaMemcpyHostToDevice));
        ProjectLas<<<dim3(unsigned((raw.size() + 255) / 256), unsigned(count)), 256>>>(
            static_cast<const int3*>(points.data), int(raw.size()), static_cast<const Projection*>(cameras.data),
            static_cast<float*>(maps.data), w, h);
        CudaCheck(cudaGetLastError());
        CudaCheck(cudaDeviceSynchronize());
      });
      accepted = kept;
      rejected = skipped;
      for (size_t j = 0; j < count; ++j) {
        const uint32_t id = images[first + j];
        const auto static_mask = StaticMask(scene, work, id, camera, options);
        CudaCheck(cudaMemcpy(mask.data, static_mask.data, size, cudaMemcpyHostToDevice));
        const auto* raw_depth = static_cast<float*>(maps.data) + j * size;
        auto* filtered = static_cast<float*>(clean.data);
        auto* scratch = static_cast<float*>(filled.data);
        const auto* keep = static_cast<const uint8_t*>(mask.data);
        Cull<<<blocks, 256>>>(raw_depth, filtered, keep, w, h);
        for (int pass = 0; pass < 2; ++pass) {
          Fill<<<blocks, 256>>>(filtered, scratch, keep, w, h);
          std::swap(filtered, scratch);
        }
        Edges<<<blocks, 256>>>(filtered, keep, static_cast<uint8_t*>(edge.data), w, h);
        Dilate<<<blocks, 256>>>(static_cast<const uint8_t*>(edge.data), static_cast<uint8_t*>(dilated.data),
                                 w, h, options.boundary, true);
        Dilate<<<blocks, 256>>>(static_cast<const uint8_t*>(dilated.data), static_cast<uint8_t*>(edge.data),
                                 w, h, options.boundary, false);
        Encode<<<blocks, 256>>>(filtered, static_cast<const double3*>(rays.data), static_cast<const uint8_t*>(edge.data),
                                 static_cast<uint16_t*>(millimetres.data), static_cast<uchar3*>(normals.data),
                                 w, h, camera.params[0]);
        CudaCheck(cudaGetLastError());
        cv::Mat depth(h, w, CV_16UC1), normal(h, w, CV_8UC3);
        CudaCheck(cudaMemcpy(depth.data, millimetres.data, millimetres.bytes, cudaMemcpyDeviceToHost));
        CudaCheck(cudaMemcpy(normal.data, normals.data, normals.bytes, cudaMemcpyDeviceToHost));
        const auto valid = Counts(depth, normal);
        depth_pixels += valid.first;
        normal_pixels += valid.second;
        const auto name = scene.images[id].name + ".png";
        const auto dp = output / "depths" / fs::u8path(name), np = output / "normals" / fs::u8path(name);
        WriteImage(dp, depth);
        WriteImage(np, normal);
        rows[id] = {{"image_index", id}, {"name", scene.images[id].name}, {"width", w}, {"height", h},
                    {"depth_size", fs::file_size(dp)}, {"normal_size", fs::file_size(np)},
                    {"static_pixels", cv::countNonZero(static_mask)},
                    {"depth_pixels", valid.first}, {"normal_pixels", valid.second}};
        ++done;
        if (done % 16 == 0 || done == model.Registered())
          Log("LAS supervision: " + std::to_string(done) + "/" + std::to_string(model.Registered()) +
              " images, " + std::to_string(Seconds(started)) + " s");
      }
    }
  }
  if (!depth_pixels || !normal_pixels)
    throw std::runtime_error("LAS supervision has no reliable depth or normal pixels");
  Json images = Json::array();
  for (const auto& row : rows)
    if (!row.is_null())
      images.push_back(row);
  if (request != SupervisionRequest(scene, model, work, options))
    throw std::runtime_error("LAS or geometry exclusion masks changed during supervision export");
  return {{"version", Version}, {"request", request}, {"source", "valid original LAS records"},
          {"accepted_records", accepted}, {"excluded_records", rejected}, {"depth_format", "uint16_png_mm"},
          {"depth_type", "ray_distance"}, {"invalid_depth", 0}, {"normal_format", "rgb8_png"},
          {"normal_frame", "opencv_camera_facing_camera"}, {"normal_encoding", "round(127.5 * (n + 1))"},
          {"invalid_normal_rgb", {0, 0, 0}}, {"minimum_abs_normal_dot_ray", MinIncidence},
          {"surface_radius_m", .04}, {"normal_angle_limit_degrees", 10},
          {"fine_plane_limit_m", .01}, {"coarse_plane_limit_m", .015},
          {"depth_pixels", depth_pixels}, {"normal_pixels", normal_pixels}, {"images", images},
          {"seconds", Seconds(started)}};
}
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary) {
  if (summary.at("version") != Version || summary.at("depth_type") != "ray_distance" ||
      summary.at("depth_format") != "uint16_png_mm" || summary.at("normal_format") != "rgb8_png" ||
      summary.at("normal_frame") != "opencv_camera_facing_camera" ||
      summary.at("images").size() != photos.size())
    throw std::runtime_error("Unexpected LAS supervision contract");
  uint64_t depth_total = 0, normal_total = 0;
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
    const auto depth = ReadImage(dp, cv::IMREAD_UNCHANGED), normal = ReadImage(np, cv::IMREAD_UNCHANGED);
    if (depth.type() != CV_16UC1 || normal.type() != CV_8UC3 || depth.size() != normal.size() ||
        depth.cols != camera.width || depth.rows != camera.height ||
        row.at("width") != camera.width || row.at("height") != camera.height)
      throw std::runtime_error("LAS supervision map dimensions or pixel types differ");
    const auto counts = Counts(depth, normal);
    if (counts.first != row.at("depth_pixels").get<uint64_t>() ||
        counts.second != row.at("normal_pixels").get<uint64_t>())
      throw std::runtime_error("LAS supervision valid pixel counts differ");
    depth_total += counts.first;
    normal_total += counts.second;
  }
  if (!depth_total || !normal_total || depth_total != summary.at("depth_pixels").get<uint64_t>() ||
      normal_total != summary.at("normal_pixels").get<uint64_t>())
    throw std::runtime_error("LAS supervision totals differ");
}
}  // namespace geosun

#include "geometry/monocular.h"
#include "geometry/supervision_kernels.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace geosun {
namespace {
// The margin every planned ray keeps to the face edge.
constexpr double FaceMarginDegrees = 5;
// Rings of tilted faces step outwards by this share of the angle a face covers within its margin.
constexpr double RingStep = .85;
// The edge of an outermost ring's face lies this far beyond the widest ray of the lens.
constexpr double OuterRingDegrees = 15;
// Grid weights fall to zero over this outer band of the face, as a share of its half width.
constexpr double FeatherBand = .3;
// Fewest valid samples in MoGe's 64 x 64 shift grid for a face to be used.
constexpr size_t MinimumShiftSamples = 32;
// Fewest overlapping samples needed to align the relative scale of a tilted face.
constexpr size_t MinimumScaleOverlapSamples = 64;

double Radians(double degrees) { return degrees * Pi / 180; }
void AddRing(std::vector<Eigen::Matrix3d>& rotations, int ring, double tilt) {
  for (int k = 0; k < ring; ++k) {
    const double azimuth = 2 * Pi * k / ring;
    // Rotates the optical axis towards azimuth; a face's rows stay close to the lens rows.
    const Eigen::Matrix3d face_to_camera =
        Eigen::AngleAxisd(tilt, Eigen::Vector3d(-std::sin(azimuth), std::cos(azimuth), 0))
            .toRotationMatrix();
    rotations.push_back(face_to_camera.transpose());
  }
}
// Z shift turning MoGe's affine point map into camera space for the known face focal, as MoGe's
// own inference: min sum |xy / (z + shift) - uv|^2 over its 64 x 64 nearest-neighbour grid.
bool SolveShiftFromSamples(const float4* grid, int size, double& shift) {
  struct Sample {
    double x, y, z, u, v;
  };
  std::vector<Sample> samples;
  samples.reserve(4096);
  double nearest = INFINITY;
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 64; ++j) {
      const float4 s = grid[i * 64 + j];
      if (s.w <= 0.f)
        continue;
      const int row = i * size / 64, column = j * size / 64;
      samples.push_back({double(s.x), double(s.y), double(s.z), (2. * column + 1 - size) / size,
                         (2. * row + 1 - size) / size});
      nearest = std::min(nearest, double(s.z));
    }
  if (samples.size() < MinimumShiftSamples)
    return false;
  auto cost = [&](double s) {
    if (nearest + s <= 1e-6)
      return double(INFINITY);
    double sum = 0;
    for (const auto& p : samples) {
      const double ex = p.x / (p.z + s) - p.u, ey = p.y / (p.z + s) - p.v;
      sum += ex * ex + ey * ey;
    }
    return sum;
  };
  double s = 0, current = cost(s), damping = 1e-3;
  for (int iteration = 0; iteration < 100; ++iteration) {
    double gradient = 0, hessian = 0;
    for (const auto& p : samples) {
      const double inverse = 1 / (p.z + s), jx = -p.x * inverse * inverse,
                   jy = -p.y * inverse * inverse;
      gradient += jx * (p.x * inverse - p.u) + jy * (p.y * inverse - p.v);
      hessian += jx * jx + jy * jy;
    }
    if (!(hessian > 0))
      break;
    const double step = -gradient / (hessian * (1 + damping));
    const double trial = cost(s + step);
    if (trial < current) {
      const bool converged = std::abs(step) < 1e-9 * (1 + std::abs(s)) ||
                             current - trial < 1e-12 * current;
      s += step;
      current = trial;
      damping = std::max(damping / 10, 1e-9);
      if (converged)
        break;
    } else if ((damping *= 10) > 1e9) {
      break;
    }
  }
  if (!std::isfinite(current))
    return false;
  shift = s;
  return true;
}
}  // namespace
cv::Mat MonocularLens::Prefilter(const cv::Mat& rgb) const {
  if (rgb.type() != CV_8UC3 || rgb.cols != source.width || rgb.rows != source.height)
    throw std::runtime_error("Photo size differs from the lens calibration");
  if (prefilter >= 1)
    return rgb;
  cv::Mat small;
  cv::resize(rgb, small, cv::Size(prefilter_width, prefilter_height), 0, 0, cv::INTER_AREA);
  return small;
}
int MonocularLens::SamplesAtRay(const Eigen::Vector3d& ray_ref,
                                std::array<FaceSample, MaxSamples>& out) const {
  const double half_width = std::tan(Radians(half_degrees));
  std::array<FaceSample, 32> found{};
  int count = 0;
  for (size_t f = 0; f < faces.size(); ++f) {
    const Eigen::Vector3d q = faces[f].rotation * ray_ref;
    if (q.z() <= 0)
      continue;
    const double xn = q.x() / q.z(), yn = q.y() / q.z();
    const double weight = std::pow(
        std::clamp((1 - std::max(std::abs(xn), std::abs(yn)) / half_width) / FeatherBand, 0., 1.),
        2);
    const double x = focal * xn + size / 2. - .5, y = focal * yn + size / 2. - .5;
    if (weight > 0 && x >= 0 && y >= 0 && x < size - 1 && y < size - 1 &&
        count < int(found.size()))
      found[size_t(count++)] = {float(x), float(y), float(weight), uint8_t(f)};
  }
  std::sort(found.begin(), found.begin() + count,
            [](const FaceSample& a, const FaceSample& b) { return a.weight > b.weight; });
  const int keep = std::min(count, MaxSamples);
  for (int k = 0; k < keep; ++k)
    out[size_t(k)] = found[size_t(k)];
  return keep;
}
Json MonocularLens::ToJson() const {
  Json axes = Json::array();
  for (const auto& face : faces) {
    const Eigen::Vector3d axis = face.rotation.transpose().col(2);
    axes.push_back({axis.x(), axis.y(), axis.z()});
  }
  size_t covered = 0;
  for (const auto count : counts)
    covered += count > 0;
  return {{"faces", faces.size()},
          {"face_size", size},
          {"face_field_of_view_degrees", 2 * half_degrees},
          {"edge_margin_degrees", FaceMarginDegrees},
          {"ring_tilt_degrees", ring_tilts},
          {"face_axes", axes},
          {"prefilter_scale", prefilter},
          {"grid", {grid.width, grid.height}},
          {"covered_pixels", covered}};
}
Json FaceEstimate::ToJson() const {
  return {{"valid_pixels", valid}, {"shift", shift}, {"metric_scale", metric_scale}};
}
MonocularLens PlanMonocularLens(const Camera& source, const Camera& grid, int size,
                                double half_degrees) {
  if (size < 64)
    throw std::runtime_error("Monocular face size must be at least 64 pixels");
  MonocularLens lens;
  lens.source = source;
  lens.grid = grid;
  lens.size = size;
  lens.half_degrees = half_degrees;
  lens.focal = size / (2 * std::tan(Radians(half_degrees)));
  const int w = grid.width, h = grid.height;
  const double radius = grid.Radius();
  std::vector<Eigen::Vector3d> rays(size_t(w) * h, Eigen::Vector3d::Zero());
  double widest = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double dx = (x + .5 - grid.params[2]) / grid.params[0],
                   dy = (y + .5 - grid.params[3]) / grid.params[1];
      if (dx * dx + dy * dy > radius * radius)
        continue;
      const auto ray = grid.Ray({x + .5, y + .5});
      rays[size_t(y) * w + x] = ray;
      widest = std::max(widest, std::acos(std::clamp(ray.z(), -1., 1.)));
    }
  if (widest == 0)
    throw std::runtime_error("Lens has no valid rays");
  const double edge = std::tan(Radians(half_degrees - FaceMarginDegrees));
  auto covered = [&](const std::vector<Eigen::Matrix3d>& rotations) {
    for (const auto& ray : rays) {
      if (ray.isZero())
        continue;
      bool inside = false;
      for (const auto& rotation : rotations) {
        const Eigen::Vector3d q = rotation * ray;
        if (q.z() > 0 && std::max(std::abs(q.x()), std::abs(q.y())) <= edge * q.z()) {
          inside = true;
          break;
        }
      }
      if (!inside)
        return false;
    }
    return true;
  };
  std::vector<Eigen::Matrix3d> rotations{Eigen::Matrix3d::Identity()};
  if (!covered(rotations)) {
    // The rings overlap the front face and each other, and the last one reaches the lens edge.
    const double step = RingStep * Radians(2 * (half_degrees - FaceMarginDegrees)),
                 inside = Radians(half_degrees - OuterRingDegrees),
                 outer = std::max(widest - inside, inside);
    std::vector<double> tilts;
    for (double tilt = step; tilt < outer - step / 2; tilt += step)
      tilts.push_back(tilt);
    tilts.push_back(outer);
    for (const double tilt : tilts)
      lens.ring_tilts.push_back(tilt * 180 / Pi);
    for (int extra = 0;; ++extra) {
      if (extra > 8)
        throw std::runtime_error("Monocular faces cannot cover the lens field of view");
      rotations.assign(1, Eigen::Matrix3d::Identity());
      for (const double tilt : tilts)
        AddRing(rotations, int(std::ceil(2 * Pi * std::sin(tilt) / step)) + extra, tilt);
      if (covered(rotations))
        break;
    }
  }
  lens.prefilter = std::min(1., 2 * lens.focal / source.params[0]);
  lens.prefilter_width =
      lens.prefilter >= 1 ? source.width : int(std::lround(source.width * lens.prefilter));
  lens.prefilter_height =
      lens.prefilter >= 1 ? source.height : int(std::lround(source.height * lens.prefilter));
  lens.prefilter_sx = double(lens.prefilter_width) / double(source.width);
  lens.prefilter_sy = double(lens.prefilter_height) / double(source.height);
  for (const auto& rotation : rotations)
    lens.faces.push_back({rotation});
  lens.samples.assign(rays.size(), {});
  lens.counts.assign(rays.size(), 0);
  lens.rays = cv::Mat::zeros(h, w, CV_32FC3);
  for (size_t i = 0; i < rays.size(); ++i) {
    if (rays[i].isZero())
      continue;
    lens.rays.at<cv::Vec3f>(int(i / w), int(i % w)) =
        cv::Vec3f(float(rays[i].x()), float(rays[i].y()), float(rays[i].z()));
    lens.counts[i] = uint8_t(lens.SamplesAtRay(rays[i], lens.samples[i]));
  }
  return lens;
}
struct MonocularEstimator::Workspace {
  cudaStream_t stream = nullptr;
  int faces = 0;
  size_t prefiltered_capacity = 0;
  const MonocularLens* cached_lens = nullptr;
  std::unique_ptr<DeviceBuffer> d_prefiltered, d_source_params, d_all_face_inputs, d_points,
      d_normals, d_masks, d_face_valid, d_metric_scales, d_shift_samples, d_shifts, d_shift_ok,
      d_to_camera, d_out_distance, d_out_normal, d_valid_counts;
  std::shared_ptr<DeviceBuffer> d_face_input;
  std::unique_ptr<HostBuffer> h_prefiltered, h_shift_samples, h_metric_scales, h_shifts, h_shift_ok,
      h_to_camera, h_out_distance, h_out_normal, h_valid_counts;
  explicit Workspace(cudaStream_t s) : stream(s) {}
  ~Workspace() {
    if (stream)
      cudaStreamSynchronize(stream);
  }
  void ReservePrefiltered(size_t bytes) {
    if (bytes <= prefiltered_capacity)
      return;
    d_prefiltered = std::make_unique<DeviceBuffer>(bytes);
    h_prefiltered = std::make_unique<HostBuffer>(bytes);
    prefiltered_capacity = bytes;
  }
  void Reserve(int count, int size) {
    if (count <= faces)
      return;
    const size_t face_pixels = size_t(size) * size, total_pixels = size_t(count) * face_pixels;
    d_source_params = std::make_unique<DeviceBuffer>(8 * sizeof(double));
    d_all_face_inputs = std::make_unique<DeviceBuffer>(total_pixels * 3 * sizeof(float));
    d_face_input = std::make_shared<DeviceBuffer>(face_pixels * 3 * sizeof(float));
    d_points = std::make_unique<DeviceBuffer>(total_pixels * 3 * sizeof(float));
    d_normals = std::make_unique<DeviceBuffer>(total_pixels * 3 * sizeof(float));
    d_masks = std::make_unique<DeviceBuffer>(total_pixels * sizeof(float));
    d_face_valid = std::make_unique<DeviceBuffer>(total_pixels);
    d_metric_scales = std::make_unique<DeviceBuffer>(size_t(count) * sizeof(float));
    d_shift_samples = std::make_unique<DeviceBuffer>(size_t(count) * 4096 * sizeof(float4));
    d_shifts = std::make_unique<DeviceBuffer>(size_t(count) * sizeof(double));
    d_shift_ok = std::make_unique<DeviceBuffer>(size_t(count));
    d_to_camera = std::make_unique<DeviceBuffer>(size_t(count) * 9 * sizeof(double));
    d_out_distance = std::make_unique<DeviceBuffer>(total_pixels * sizeof(float));
    d_out_normal = std::make_unique<DeviceBuffer>(total_pixels * 3 * sizeof(float));
    d_valid_counts = std::make_unique<DeviceBuffer>(size_t(count) * sizeof(uint32_t));
    h_shift_samples = std::make_unique<HostBuffer>(size_t(count) * 4096 * sizeof(float4));
    h_metric_scales = std::make_unique<HostBuffer>(size_t(count) * sizeof(float));
    h_shifts = std::make_unique<HostBuffer>(size_t(count) * sizeof(double));
    h_shift_ok = std::make_unique<HostBuffer>(size_t(count));
    h_to_camera = std::make_unique<HostBuffer>(size_t(count) * 9 * sizeof(double));
    h_out_distance = std::make_unique<HostBuffer>(total_pixels * sizeof(float));
    h_out_normal = std::make_unique<HostBuffer>(total_pixels * 3 * sizeof(float));
    h_valid_counts = std::make_unique<HostBuffer>(size_t(count) * sizeof(uint32_t));
    faces = count;
    cached_lens = nullptr;
  }
};
MonocularEstimator::MonocularEstimator(Models& models)
    : models_(models),
      workspace_(std::make_unique<Workspace>(models.Stream())) {
  const auto& manifest = models.Manifest();
  if (!manifest.at("models").contains("geometry"))
    throw std::runtime_error("Model package has no geometry engine; use run.ps1 prepare-models");
  size_ = manifest.at("geometry_size");
  // Load the engine now, so device memory budgets measured afterwards see it.
  const std::vector<float> blank(size_t(3) * size_ * size_, 0.f);
  models_.Predict("geometry",
                  {Tensor::Float({1, 3, size_, size_}, blank.data(), blank.size())}, true);
}
MonocularEstimator::~MonocularEstimator() = default;
std::vector<FaceEstimate> MonocularEstimator::Estimate(const MonocularLens& lens,
                                                       const cv::Mat& prefiltered,
                                                       const Eigen::Vector3d& omega, int axis,
                                                       bool rolling) {
  if (lens.size != size_ || prefiltered.type() != CV_8UC3 ||
      prefiltered.cols != lens.prefilter_width || prefiltered.rows != lens.prefilter_height)
    throw std::runtime_error("Monocular photo differs from the geometry engine or lens plan");
  const int size = size_, faces = int(lens.faces.size());
  if (!faces)
    return {};
  const size_t face_pixels = size_t(size) * size;
  const auto stream = models_.Stream();
  workspace_->Reserve(faces, size);
  const cv::Mat contiguous = prefiltered.isContinuous() ? prefiltered : prefiltered.clone();
  const size_t pref_bytes = contiguous.total() * 3;
  workspace_->ReservePrefiltered(pref_bytes);
  std::memcpy(workspace_->h_prefiltered->data, contiguous.data, pref_bytes);
  CudaCheck(cudaMemcpyAsync(workspace_->d_prefiltered->data, workspace_->h_prefiltered->data,
                            pref_bytes, cudaMemcpyHostToDevice, stream));
  if (workspace_->cached_lens != &lens) {
    auto* to_camera = static_cast<double*>(workspace_->h_to_camera->data);
    for (int f = 0; f < faces; ++f) {
      const Eigen::Matrix3d tc = lens.faces[f].rotation.transpose();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          to_camera[f * 9 + r * 3 + c] = tc(r, c);
    }
    CudaCheck(cudaMemcpyAsync(workspace_->d_to_camera->data, to_camera,
                              size_t(faces) * 9 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CudaCheck(cudaMemcpyAsync(workspace_->d_source_params->data, lens.source.params.data(),
                              8 * sizeof(double), cudaMemcpyHostToDevice, stream));
    workspace_->cached_lens = &lens;
  }
  const double3 gpu_omega = make_double3(omega.x(), omega.y(), omega.z());
  const bool use_rolling = rolling && !omega.isZero();
  supervision_gpu::SampleMogeFaces(
      static_cast<const uint8_t*>(workspace_->d_prefiltered->data), lens.prefilter_width,
      lens.prefilter_height, static_cast<const double*>(workspace_->d_source_params->data),
      lens.source.width, lens.source.height, lens.source.Radius(), lens.prefilter_sx,
      lens.prefilter_sy, static_cast<const double*>(workspace_->d_to_camera->data), gpu_omega,
      use_rolling ? 1 : 0, axis, faces, size, lens.focal,
      static_cast<float*>(workspace_->d_all_face_inputs->data),
      static_cast<uint8_t*>(workspace_->d_face_valid->data), stream);
  const std::vector<int64_t> map{1, size, size, 3};
  for (int f = 0; f < faces; ++f) {
    CudaCheck(cudaMemcpyAsync(workspace_->d_face_input->data,
                              static_cast<const float*>(workspace_->d_all_face_inputs->data) +
                                  size_t(f) * 3 * face_pixels,
                              3 * face_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    Tensor input;
    input.shape = {1, 3, size, size};
    input.device = workspace_->d_face_input;
    const auto outputs = models_.Predict("geometry", {input}, true);
    if (outputs.size() != 4 || outputs[0].shape != map || outputs[1].shape != map ||
        outputs[2].shape != std::vector<int64_t>{1, size, size} ||
        outputs[3].shape != std::vector<int64_t>{1} || !outputs[0].device ||
        !outputs[1].device || !outputs[2].device || !outputs[3].device)
      throw std::runtime_error("Geometry engine output contract differs");
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_points->data) + size_t(f) * face_pixels * 3,
        outputs[0].device->data, face_pixels * 3 * sizeof(float), cudaMemcpyDeviceToDevice,
        stream));
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_normals->data) + size_t(f) * face_pixels * 3,
        outputs[1].device->data, face_pixels * 3 * sizeof(float), cudaMemcpyDeviceToDevice,
        stream));
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_masks->data) + size_t(f) * face_pixels,
        outputs[2].device->data, face_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CudaCheck(cudaMemcpyAsync(static_cast<float*>(workspace_->d_metric_scales->data) + f,
                              outputs[3].device->data, sizeof(float), cudaMemcpyDeviceToDevice,
                              stream));
    supervision_gpu::SampleMogeShiftGrid(
        static_cast<const float*>(workspace_->d_points->data) + size_t(f) * face_pixels * 3,
        static_cast<const float*>(workspace_->d_masks->data) + size_t(f) * face_pixels,
        static_cast<const uint8_t*>(workspace_->d_face_valid->data) + size_t(f) * face_pixels,
        size, static_cast<float4*>(workspace_->d_shift_samples->data) + size_t(f) * 4096, stream);
  }
  CudaCheck(cudaMemcpyAsync(workspace_->h_shift_samples->data, workspace_->d_shift_samples->data,
                            size_t(faces) * 4096 * sizeof(float4), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_metric_scales->data, workspace_->d_metric_scales->data,
                            size_t(faces) * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  const auto* metric_scales = static_cast<const float*>(workspace_->h_metric_scales->data);
  const auto* samples = static_cast<const float4*>(workspace_->h_shift_samples->data);
  auto* shifts = static_cast<double*>(workspace_->h_shifts->data);
  auto* shift_ok = static_cast<uint8_t*>(workspace_->h_shift_ok->data);
  for (int f = 0; f < faces; ++f) {
    if (!std::isfinite(metric_scales[f]) || metric_scales[f] <= 0)
      throw std::runtime_error("Geometry engine returned an invalid metric scale");
    double shift = 0;
    shift_ok[f] = SolveShiftFromSamples(samples + size_t(f) * 4096, size, shift) ? 1 : 0;
    shifts[f] = shift;
  }
  CudaCheck(cudaMemcpyAsync(workspace_->d_shifts->data, shifts, size_t(faces) * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
  CudaCheck(cudaMemcpyAsync(workspace_->d_shift_ok->data, shift_ok, size_t(faces),
                            cudaMemcpyHostToDevice, stream));
  CudaCheck(cudaMemsetAsync(workspace_->d_valid_counts->data, 0, size_t(faces) * sizeof(uint32_t),
                            stream));
  supervision_gpu::BuildMogeFaceEstimates(
      static_cast<const float*>(workspace_->d_points->data),
      static_cast<const float*>(workspace_->d_normals->data),
      static_cast<const float*>(workspace_->d_masks->data),
      static_cast<const uint8_t*>(workspace_->d_face_valid->data),
      static_cast<const double*>(workspace_->d_shifts->data),
      static_cast<const uint8_t*>(workspace_->d_shift_ok->data),
      static_cast<const float*>(workspace_->d_metric_scales->data),
      static_cast<const double*>(workspace_->d_to_camera->data), faces, size, lens.focal,
      static_cast<float*>(workspace_->d_out_distance->data),
      static_cast<float*>(workspace_->d_out_normal->data),
      static_cast<uint32_t*>(workspace_->d_valid_counts->data), stream);
  CudaCheck(cudaMemcpyAsync(workspace_->h_out_distance->data, workspace_->d_out_distance->data,
                            size_t(faces) * face_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_out_normal->data, workspace_->d_out_normal->data,
                            size_t(faces) * face_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_valid_counts->data, workspace_->d_valid_counts->data,
                            size_t(faces) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  const auto* out_dist = static_cast<const float*>(workspace_->h_out_distance->data);
  const auto* out_norm = static_cast<const float*>(workspace_->h_out_normal->data);
  const auto* valid_counts = static_cast<const uint32_t*>(workspace_->h_valid_counts->data);
  std::vector<FaceEstimate> estimates(faces);
  for (int f = 0; f < faces; ++f) {
    auto& e = estimates[f];
    e.metric_scale = metric_scales[f];
    e.shift = shift_ok[f] ? shifts[f] : 0.0;
    e.valid = valid_counts[f];
    e.distance = cv::Mat(size, size, CV_32FC1);
    std::memcpy(e.distance.data, out_dist + size_t(f) * face_pixels, face_pixels * sizeof(float));
    e.normal = cv::Mat(size, size, CV_32FC3);
    std::memcpy(e.normal.data, out_norm + size_t(f) * face_pixels * 3,
                face_pixels * 3 * sizeof(float));
  }
  return estimates;
}
namespace {
bool SampleFace(const FaceEstimate& face, const FaceSample& sample, double& distance,
                Eigen::Vector3d& normal) {
  const int x = int(sample.x), y = int(sample.y);
  const double fx = sample.x - x, fy = sample.y - y;
  const double corner[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
  distance = 0;
  normal.setZero();
  for (int c = 0; c < 4; ++c) {
    const float d = face.distance.at<float>(y + c / 2, x + c % 2);
    if (!(d > 0.f) || !std::isfinite(d))
      return false;
    const auto& value = face.normal.at<cv::Vec3f>(y + c / 2, x + c % 2);
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
      return false;
    distance += corner[c] * d;
    normal += corner[c] * Eigen::Vector3d(value[0], value[1], value[2]);
  }
  return std::isfinite(distance) && distance > 0.0;
}
struct FacePixelSample {
  uint32_t pixel = 0;
  float distance = 0.f;
  float weight = 0.f;
};
}  // namespace

MogeSsiResult EncodeMogeSsiMaps(const MonocularLens& lens,
                                const std::vector<FaceEstimate>& faces,
                                const cv::Mat& allowed,
                                const Eigen::Vector3d* omega,
                                int axis) {
  const int w = lens.grid.width, h = lens.grid.height, fsize = lens.size;
  if (faces.size() != lens.faces.size() || allowed.type() != CV_8UC1 || allowed.cols != w ||
      allowed.rows != h)
    throw std::runtime_error("Static mask or face count differs from the lens plan");

  const size_t face_count = faces.size();
  for (size_t f = 0; f < face_count; ++f) {
    const auto& face = faces[f];
    if (face.distance.type() != CV_32FC1 || face.normal.type() != CV_32FC3 ||
        face.distance.rows != fsize || face.distance.cols != fsize ||
        face.normal.size() != face.distance.size())
      throw std::runtime_error("Monocular face estimate differs from the lens plan");
  }

  const cv::Mat allowed_contig = allowed.isContinuous() ? allowed : allowed.clone();
  const uint8_t* allowed_ptr = allowed_contig.ptr<uint8_t>();
  const size_t pixels = size_t(w) * h;
  const bool rolling = omega && !omega->isZero();

  std::vector<std::vector<FacePixelSample>> face_samples(face_count);
  for (auto& list : face_samples)
    list.reserve(pixels / std::max<size_t>(1, face_count / 2));

  std::vector<cv::Vec3f> blended_normals(pixels, cv::Vec3f(0.f, 0.f, 0.f));
  std::vector<uint8_t> has_normal(pixels, 0);

  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const auto& r_cv = lens.rays.at<cv::Vec3f>(y, x);
      if (r_cv[0] == 0.f && r_cv[1] == 0.f && r_cv[2] == 0.f)
        continue;
      const Eigen::Vector3d r_exp(r_cv[0], r_cv[1], r_cv[2]);

      std::array<FaceSample, MonocularLens::MaxSamples> samples;
      int count = 0;
      Eigen::Vector3d w_rot = Eigen::Vector3d::Zero();
      if (rolling) {
        const double s = axis ? (y + .5 - .5 * h) / h : (x + .5 - .5 * w) / w;
        w_rot = s * (*omega);
        const Eigen::Vector3d r_ref =
            (r_exp - w_rot.cross(r_exp) + w_rot.dot(r_exp) * w_rot).normalized();
        count = lens.SamplesAtRay(r_ref, samples);
      } else {
        samples = lens.samples[i];
        count = lens.counts[i];
      }

      double w_sum = 0.0;
      Eigen::Vector3d n_sum = Eigen::Vector3d::Zero();
      for (int k = 0; k < count; ++k) {
        const auto& sample = samples[k];
        double distance = 0.0;
        Eigen::Vector3d n;
        if (!SampleFace(faces[sample.face], sample, distance, n) || !(sample.weight > 0.f))
          continue;
        face_samples[sample.face].push_back({uint32_t(i), float(distance), sample.weight});
        w_sum += sample.weight;
        n_sum += sample.weight * n;
      }

      if (!allowed_ptr[i] || !(w_sum > 0.0))
        continue;
      const double n_norm = n_sum.norm();
      if (!(n_norm > 1e-6) || !std::isfinite(n_norm))
        continue;
      Eigen::Vector3d normal = n_sum / n_norm;
      if (rolling)
        normal = (normal + w_rot.cross(normal)).normalized();
      if (normal.dot(r_exp) > 0.0)
        normal = -normal;
      blended_normals[i] = cv::Vec3f(float(normal.x()), float(normal.y()), float(normal.z()));
      has_normal[i] = 1;
    }

  // Align face scales across overlapping faces using median ratio (matching LFS align_face_scales).
  std::vector<double> scales(face_count, 1.0);
  std::vector<double> merged(pixels, 0.0);
  std::vector<double> merged_weight(pixels, 0.0);
  std::vector<float> ratios;
  for (size_t f = 0; f < face_count; ++f) {
    const auto& list = face_samples[f];
    if (f > 0) {
      ratios.clear();
      for (const auto& item : list) {
        if (allowed_ptr[item.pixel] && merged_weight[item.pixel] > 0.0 && item.distance > 0.f &&
            std::isfinite(item.distance))
          ratios.push_back(float(merged[item.pixel] / double(item.distance)));
      }
      if (ratios.size() < MinimumScaleOverlapSamples) {
        ratios.clear();
        for (const auto& item : list) {
          if (merged_weight[item.pixel] > 0.0 && item.distance > 0.f &&
              std::isfinite(item.distance))
            ratios.push_back(float(merged[item.pixel] / double(item.distance)));
        }
      }
      if (ratios.size() >= MinimumScaleOverlapSamples) {
        const auto mid = ratios.begin() + ptrdiff_t(ratios.size() / 2);
        std::nth_element(ratios.begin(), mid, ratios.end());
        scales[f] = double(*mid);
      }
    }
    for (const auto& item : list) {
      if (!(item.distance > 0.f) || !std::isfinite(item.distance) || !(item.weight > 0.f))
        continue;
      const double total = merged_weight[item.pixel] + double(item.weight);
      merged[item.pixel] = (merged[item.pixel] * merged_weight[item.pixel] +
                            scales[f] * double(item.distance) * double(item.weight)) /
                           total;
      merged_weight[item.pixel] = total;
    }
  }

  double min_d = std::numeric_limits<double>::infinity();
  double max_d = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < pixels; ++i) {
    if (!allowed_ptr[i] || !(merged_weight[i] > 0.0) || !std::isfinite(merged[i]) ||
        !(merged[i] > 0.0))
      continue;
    min_d = std::min(min_d, merged[i]);
    max_d = std::max(max_d, merged[i]);
  }

  MogeSsiResult out;
  out.depth_u16 = cv::Mat::zeros(h, w, CV_16UC1);
  out.normal_rgb = cv::Mat::zeros(h, w, CV_8UC3);
  out.face_scales = scales;

  if (std::isfinite(min_d) && std::isfinite(max_d)) {
    out.min_distance = min_d;
    out.max_distance = max_d;
    const double denom = std::max(max_d - min_d, 1e-6);
    const auto encode_norm = [](double v) -> uint8_t {
      return uint8_t(std::clamp(std::lround(127.5 * (v + 1.0)), 0L, 255L));
    };
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (!allowed_ptr[i] || !(merged_weight[i] > 0.0) || !std::isfinite(merged[i]) ||
            !(merged[i] > 0.0))
          continue;
        const double normalized = 0.02 + 0.98 * ((merged[i] - min_d) / denom);
        out.depth_u16.at<uint16_t>(y, x) =
            uint16_t(std::clamp(std::lround(normalized * 65535.0), 1L, 65535L));
        if (has_normal[i]) {
          const auto& n = blended_normals[i];
          out.normal_rgb.at<cv::Vec3b>(y, x) =
              cv::Vec3b(encode_norm(n[2]), encode_norm(n[1]), encode_norm(n[0]));
        }
      }
  }

  Json face_rows = Json::array();
  for (size_t f = 0; f < face_count; ++f) {
    face_rows.push_back({{"valid_pixels", faces[f].valid},
                         {"shift", faces[f].shift},
                         {"metric_scale", faces[f].metric_scale},
                         {"aligned_scale", scales[f]}});
  }
  out.report = {{"min_distance_m", out.min_distance},
                {"max_distance_m", out.max_distance},
                {"face_scales", scales},
                {"faces", std::move(face_rows)}};
  return out;
}
}  // namespace geosun

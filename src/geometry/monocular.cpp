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
// MoGe face edge detection and LAS reverse cleaning parameters.
constexpr double EdgeDepthLog = .05;
constexpr double EdgeNormalDeg = 35.;
constexpr int FitLattice = 2;
constexpr size_t MinFaceRansacSamples = 64, MinFaceRansacInliers = 32;
constexpr int RansacIterations = 256;
constexpr size_t RansacScoreSubset = 4096;
constexpr double RansacScoreLog = .30, RansacMadScale = 3.5, RansacLogMin = .25,
                 RansacLogMax = .45, MinRansacInlierFraction = .40,
                 PenetrationMinDiffMetres = .25, NormalDisagreeMaxDeg = 35.,
                 FarRescueNormalAgreeMaxDeg = 25., FarRescueMaxLogRatio = 1.1;

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
    if (!(d > 0))
      return false;
    const auto& value = face.normal.at<cv::Vec3f>(y + c / 2, x + c % 2);
    distance += corner[c] * d;
    normal += corner[c] * Eigen::Vector3d(value[0], value[1], value[2]);
  }
  return true;
}
struct PixelFaces {
  float distance[MonocularLens::MaxSamples]{};
  uint8_t face[MonocularLens::MaxSamples]{};
  uint8_t count = 0;
};
struct RansacSample {
  double q_moge;
  double r_las;
  double q_las;
};
struct FaceRansac {
  bool accepted = false;
  double s = 1.;
  double t = 0.;
  double mad_log = 0.;
  double tau = RansacLogMax;
  size_t samples = 0;
  size_t inliers = 0;
};
FaceRansac FitFaceRansac(const std::vector<RansacSample>& pts, size_t face_index) {
  FaceRansac fit;
  const size_t n = pts.size();
  fit.samples = n;
  if (n < MinFaceRansacSamples)
    return fit;

  const size_t stride = std::max<size_t>(1, n / RansacScoreSubset);
  auto score_model = [&](double s, double t) -> size_t {
    if (!(s > 0.) || !std::isfinite(s) || !std::isfinite(t))
      return 0;
    size_t count = 0;
    for (size_t i = 0; i < n; i += stride) {
      const double q_hat = s * pts[i].q_moge + t;
      if (q_hat > 0. && std::abs(std::log(pts[i].r_las * q_hat)) <= RansacScoreLog)
        ++count;
    }
    return count;
  };

  std::vector<double> init_ratios;
  init_ratios.reserve((n + stride - 1) / stride);
  for (size_t i = 0; i < n; i += stride)
    init_ratios.push_back(std::log(pts[i].q_las / pts[i].q_moge));
  double best_s = std::exp(Quantile(init_ratios, .5));
  double best_t = 0.;
  size_t best_score = score_model(best_s, best_t);

  uint32_t state = 0x9e3779b9u ^ uint32_t(face_index * 2654435761u) ^ uint32_t(n);
  if (!state)
    state = 1u;
  auto next_rand = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  for (int it = 0; it < RansacIterations; ++it) {
    const size_t i0 = next_rand() % n;
    const size_t i1 = next_rand() % n;
    const double dq = pts[i1].q_moge - pts[i0].q_moge;
    if (std::abs(dq) < 1e-6)
      continue;
    const double s = (pts[i1].q_las - pts[i0].q_las) / dq;
    if (!(s > 0.) || !std::isfinite(s))
      continue;
    const double t = pts[i0].q_las - s * pts[i0].q_moge;
    if (!std::isfinite(t))
      continue;
    const size_t score = score_model(s, t);
    if (score > best_score) {
      best_score = score;
      best_s = s;
      best_t = t;
    }
  }

  std::vector<size_t> consensus;
  consensus.reserve(n);
  std::vector<double> consensus_ratios;
  consensus_ratios.reserve(n);
  double m00 = 0., m01 = 0., m11 = 0., rhs0 = 0., rhs1 = 0.;
  for (size_t i = 0; i < n; ++i) {
    const double q_hat = best_s * pts[i].q_moge + best_t;
    if (q_hat > 0. && std::abs(std::log(pts[i].r_las * q_hat)) <= RansacScoreLog) {
      consensus.push_back(i);
      consensus_ratios.push_back(std::log(pts[i].q_las / pts[i].q_moge));
      const double a = pts[i].r_las * pts[i].q_moge;
      const double b = pts[i].r_las;
      m00 += a * a;
      m01 += a * b;
      m11 += b * b;
      rhs0 += a;
      rhs1 += b;
    }
  }
  if (consensus.size() < MinFaceRansacInliers)
    return fit;

  double s_final = best_s, t_final = best_t;
  bool ls_ok = false;
  const double det = m00 * m11 - m01 * m01;
  if (det > 1e-12 * (m00 * m11 + 1e-12)) {
    const double s_ls = (m11 * rhs0 - m01 * rhs1) / det;
    const double t_ls = (-m01 * rhs0 + m00 * rhs1) / det;
    if (s_ls > 0. && std::isfinite(s_ls) && std::isfinite(t_ls)) {
      size_t positive = 0;
      for (const size_t idx : consensus)
        if (s_ls * pts[idx].q_moge + t_ls > 0.)
          ++positive;
      if (positive >= MinFaceRansacInliers) {
        s_final = s_ls;
        t_final = t_ls;
        ls_ok = true;
      }
    }
  }
  if (!ls_ok) {
    s_final = std::exp(Quantile(consensus_ratios, .5));
    t_final = 0.;
  }

  std::vector<double> residuals;
  residuals.reserve(consensus.size());
  for (const size_t idx : consensus) {
    const double q_hat = s_final * pts[idx].q_moge + t_final;
    if (q_hat > 0.)
      residuals.push_back(std::log(pts[idx].r_las * q_hat));
  }
  if (residuals.size() < MinFaceRansacInliers)
    return fit;

  const double med = Quantile(residuals, .5);
  std::vector<double> abs_dev(residuals.size());
  for (size_t i = 0; i < residuals.size(); ++i)
    abs_dev[i] = std::abs(residuals[i] - med);
  const double mad_log = 1.4826 * Quantile(abs_dev, .5);
  const double tau = std::clamp(RansacMadScale * mad_log, RansacLogMin, RansacLogMax);

  size_t inliers = 0;
  for (const auto& p : pts) {
    const double q_hat = s_final * p.q_moge + t_final;
    if (q_hat > 0. && std::abs(std::log(p.r_las * q_hat)) <= tau)
      ++inliers;
  }

  fit.s = s_final;
  fit.t = t_final;
  fit.mad_log = mad_log;
  fit.tau = tau;
  fit.inliers = inliers;
  fit.accepted =
      inliers >= MinFaceRansacInliers && double(inliers) / double(n) >= MinRansacInlierFraction;
  return fit;
}
}  // namespace

Json CleanLasAndEncodeNormals(const MonocularLens& lens,
                              const std::vector<FaceEstimate>& faces,
                              const cv::Mat& las,
                              const cv::Mat& las_normal,
                              const cv::Mat& allowed,
                              const Eigen::Vector3d* omega,
                              int axis,
                              double min_incidence,
                              cv::Mat& depth_mm,
                              cv::Mat& normal_rgb) {
  if (faces.size() != lens.faces.size() || las.type() != CV_16UC1 ||
      las.cols != lens.grid.width || las.rows != lens.grid.height ||
      las_normal.type() != CV_8UC3 || las_normal.size() != las.size() ||
      allowed.type() != CV_8UC1 || allowed.size() != las.size())
    throw std::runtime_error("LAS depth, normal or static mask differs from the lens plan");

  const size_t face_count = faces.size();
  const int fsize = lens.size;
  const double edge_normal_cos = std::cos(Radians(EdgeNormalDeg));
  std::vector<cv::Mat> face_edges(face_count);
  for (size_t f = 0; f < face_count; ++f) {
    const auto& face = faces[f];
    if (face.distance.type() != CV_32FC1 || face.normal.type() != CV_32FC3 ||
        face.distance.rows != fsize || face.distance.cols != fsize ||
        face.normal.size() != face.distance.size())
      throw std::runtime_error("Monocular face estimate differs from the lens plan");
    cv::Mat edge = cv::Mat::zeros(fsize, fsize, CV_8UC1);
    for (int y = 0; y < fsize; ++y)
      for (int x = 0; x < fsize; ++x) {
        const float d = face.distance.at<float>(y, x);
        if (!(d > 0.f))
          continue;
        const auto& n1 = face.normal.at<cv::Vec3f>(y, x);
        const double n1_norm =
            std::sqrt(double(n1[0]) * n1[0] + double(n1[1]) * n1[1] + double(n1[2]) * n1[2]);
        bool is_edge = false;
        for (int dy = -1; dy <= 1 && !is_edge; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy)
              continue;
            const int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= fsize || yy >= fsize)
              continue;
            const float d2 = face.distance.at<float>(yy, xx);
            if (!(d2 > 0.f))
              continue;
            if (std::abs(std::log(double(d2) / double(d))) > EdgeDepthLog) {
              is_edge = true;
              break;
            }
            const auto& n2 = face.normal.at<cv::Vec3f>(yy, xx);
            const double n2_norm =
                std::sqrt(double(n2[0]) * n2[0] + double(n2[1]) * n2[1] + double(n2[2]) * n2[2]);
            if (n1_norm > 0. && n2_norm > 0.) {
              const double dot =
                  double(n1[0]) * n2[0] + double(n1[1]) * n2[1] + double(n1[2]) * n2[2];
              if (dot < edge_normal_cos * n1_norm * n2_norm) {
                is_edge = true;
                break;
              }
            }
          }
        if (is_edge)
          edge.at<uint8_t>(y, x) = 1;
      }
    face_edges[f] = std::move(edge);
  }

  const int w = lens.grid.width, h = lens.grid.height;
  const bool rolling = omega && !omega->isZero();
  cv::Mat gathered_normal = cv::Mat::zeros(h, w, CV_32FC3);
  cv::Mat grid_edge = cv::Mat::zeros(h, w, CV_8UC1);
  std::vector<PixelFaces> pixel_faces(size_t(w) * h);

  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      std::array<FaceSample, MonocularLens::MaxSamples> samples;
      int count = 0;
      Eigen::Vector3d w_rot = Eigen::Vector3d::Zero();
      if (rolling) {
        const auto& r_cv = lens.rays.at<cv::Vec3f>(y, x);
        if (r_cv[0] == 0.f && r_cv[1] == 0.f && r_cv[2] == 0.f)
          continue;
        const double s = axis ? (y + .5 - .5 * h) / h : (x + .5 - .5 * w) / w;
        w_rot = s * (*omega);
        const Eigen::Vector3d r_exp(r_cv[0], r_cv[1], r_cv[2]);
        const Eigen::Vector3d r_ref =
            (r_exp - w_rot.cross(r_exp) + w_rot.dot(r_exp) * w_rot).normalized();
        count = lens.SamplesAtRay(r_ref, samples);
      } else {
        samples = lens.samples[i];
        count = lens.counts[i];
      }
      double weights = 0.;
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      bool is_edge = false;
      auto& pf = pixel_faces[i];
      for (int k = 0; k < count; ++k) {
        const auto& sample = samples[k];
        double distance = 0.;
        Eigen::Vector3d n;
        if (!SampleFace(faces[sample.face], sample, distance, n))
          continue;
        weights += sample.weight;
        normal += sample.weight * n;
        if (pf.count < MonocularLens::MaxSamples) {
          pf.distance[pf.count] = float(distance);
          pf.face[pf.count] = sample.face;
          ++pf.count;
        }
        const int fx0 = int(sample.x), fy0 = int(sample.y);
        const auto& fe = face_edges[sample.face];
        if (fe.at<uint8_t>(fy0, fx0) || fe.at<uint8_t>(fy0, fx0 + 1) ||
            fe.at<uint8_t>(fy0 + 1, fx0) || fe.at<uint8_t>(fy0 + 1, fx0 + 1))
          is_edge = true;
      }
      if (is_edge)
        grid_edge.at<uint8_t>(y, x) = 1;
      if (weights <= 0. || normal.norm() < 1e-6)
        continue;
      normal.normalize();
      if (rolling)
        normal = (normal + w_rot.cross(normal)).normalized();
      gathered_normal.at<cv::Vec3f>(y, x) =
          cv::Vec3f(float(normal.x()), float(normal.y()), float(normal.z()));
    }

  cv::Mat moge_edge_band;
  cv::dilate(grid_edge, moge_edge_band,
             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

  std::vector<std::vector<RansacSample>> face_samples(face_count);
  for (int y = 0; y < h; y += FitLattice)
    for (int x = 0; x < w; x += FitLattice) {
      if (!allowed.at<uint8_t>(y, x) || moge_edge_band.at<uint8_t>(y, x))
        continue;
      const uint16_t raw_mm = las.at<uint16_t>(y, x);
      if (!raw_mm)
        continue;
      const double r_las = double(raw_mm) / 1000.;
      const double q_las = 1. / r_las;
      const auto& pf = pixel_faces[size_t(y) * w + x];
      for (uint8_t k = 0; k < pf.count; ++k)
        if (pf.distance[k] > 1e-9f)
          face_samples[pf.face[k]].push_back({1. / double(pf.distance[k]), r_las, q_las});
    }

  std::vector<FaceRansac> fits(face_count);
  uint64_t total_samples = 0, total_inliers = 0;
  Json face_rows = Json::array();
  for (size_t f = 0; f < face_count; ++f) {
    fits[f] = FitFaceRansac(face_samples[f], f);
    total_samples += fits[f].samples;
    total_inliers += fits[f].inliers;
    face_rows.push_back({{"accepted", fits[f].accepted},
                         {"samples", fits[f].samples},
                         {"inliers", fits[f].inliers},
                         {"scale", fits[f].s},
                         {"shift", fits[f].t},
                         {"mad_log", fits[f].mad_log},
                         {"tau", fits[f].tau},
                         {"metric_scale", faces[f].metric_scale},
                         {"face_shift", faces[f].shift},
                         {"valid_pixels", faces[f].valid}});
  }

  depth_mm = cv::Mat::zeros(h, w, CV_16UC1);
  normal_rgb = cv::Mat::zeros(h, w, CV_8UC3);
  const double normal_disagree_cos = std::cos(Radians(NormalDisagreeMaxDeg));
  const double far_rescue_cos = std::cos(Radians(FarRescueNormalAgreeMaxDeg));
  uint64_t las_input = 0, moge_edge_dropped = 0, ransac_dropped_near = 0,
           ransac_dropped_far = 0, far_normal_rescued = 0,
           penetration_normal_dropped = 0, normal_disagree_dropped = 0;

  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (!allowed.at<uint8_t>(y, x))
        continue;
      const auto& n = gathered_normal.at<cv::Vec3f>(y, x);
      const auto& ray = lens.rays.at<cv::Vec3f>(y, x);
      bool has_moge_normal = false;
      Eigen::Vector3d n_moge = Eigen::Vector3d::Zero();
      if ((n[0] || n[1] || n[2]) &&
          -(double(n[0]) * ray[0] + double(n[1]) * ray[1] + double(n[2]) * ray[2]) >= min_incidence) {
        n_moge = Eigen::Vector3d(double(n[0]), double(n[1]), double(n[2])).normalized();
        has_moge_normal = true;
        normal_rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(uint8_t(127.5f + 127.5f * n[2] + .5f),
                                                   uint8_t(127.5f + 127.5f * n[1] + .5f),
                                                   uint8_t(127.5f + 127.5f * n[0] + .5f));
      }

      const auto& ln = las_normal.at<cv::Vec3b>(y, x);
      bool has_las_normal = false;
      Eigen::Vector3d n_las = Eigen::Vector3d::Zero();
      if (ln[0] || ln[1] || ln[2]) {
        n_las = Eigen::Vector3d(double(ln[2]) / 127.5 - 1.,
                                double(ln[1]) / 127.5 - 1.,
                                double(ln[0]) / 127.5 - 1.);
        const double ln_norm = n_las.norm();
        if (ln_norm > 1e-3) {
          n_las /= ln_norm;
          has_las_normal = true;
        }
      }

      const bool on_moge_edge = moge_edge_band.at<uint8_t>(y, x) != 0;
      const double normal_dot =
          (has_moge_normal && has_las_normal) ? n_moge.dot(n_las) : -2.;
      if (!on_moge_edge && has_moge_normal && has_las_normal && normal_dot < normal_disagree_cos) {
        normal_rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        ++normal_disagree_dropped;
      }

      const uint16_t raw_mm = las.at<uint16_t>(y, x);
      if (!raw_mm)
        continue;
      ++las_input;
      if (on_moge_edge) {
        ++moge_edge_dropped;
        continue;
      }

      const auto& pf = pixel_faces[size_t(y) * w + x];
      const double r_las = double(raw_mm) / 1000.;
      bool has_accepted = false;
      bool any_inlier = false;
      bool has_best_fit = false;
      double best_abs_log = std::numeric_limits<double>::infinity();
      double best_log_ratio = 0.;
      double best_r_fit = 0.;
      double best_tau = 0.;
      for (uint8_t k = 0; k < pf.count; ++k) {
        const auto& fit = fits[pf.face[k]];
        if (!fit.accepted)
          continue;
        has_accepted = true;
        const double q_hat = fit.s / double(pf.distance[k]) + fit.t;
        if (!(q_hat > 0.))
          continue;
        const double r_fit = 1. / q_hat;
        const double log_ratio = std::log(r_las / r_fit);
        const double abs_log = std::abs(log_ratio);
        if (abs_log <= fit.tau) {
          any_inlier = true;
          break;
        }
        if (abs_log < best_abs_log) {
          has_best_fit = true;
          best_abs_log = abs_log;
          best_log_ratio = log_ratio;
          best_r_fit = r_fit;
          best_tau = fit.tau;
        }
      }

      if (!has_accepted || any_inlier) {
        depth_mm.at<uint16_t>(y, x) = raw_mm;
        continue;
      }
      if (!has_best_fit) {
        ++ransac_dropped_far;
        continue;
      }
      if (best_log_ratio < -best_tau) {
        ++ransac_dropped_near;
      } else if (has_las_normal && has_moge_normal && normal_dot >= far_rescue_cos &&
                 best_log_ratio <= FarRescueMaxLogRatio) {
        depth_mm.at<uint16_t>(y, x) = raw_mm;
        ++far_normal_rescued;
      } else {
        ++ransac_dropped_far;
        if (r_las - best_r_fit > PenetrationMinDiffMetres) {
          auto& code = normal_rgb.at<cv::Vec3b>(y, x);
          if (code[0] || code[1] || code[2]) {
            code = cv::Vec3b(0, 0, 0);
            ++penetration_normal_dropped;
          }
        }
      }
    }

  return {{"las_input_pixels", las_input},
          {"moge_edge_dropped_pixels", moge_edge_dropped},
          {"ransac_dropped_near_pixels", ransac_dropped_near},
          {"ransac_dropped_far_pixels", ransac_dropped_far},
          {"far_normal_rescued_pixels", far_normal_rescued},
          {"penetration_normal_dropped_pixels", penetration_normal_dropped},
          {"normal_disagree_dropped_pixels", normal_disagree_dropped},
          {"ransac_samples", total_samples},
          {"ransac_inliers", total_inliers},
          {"faces", face_rows}};
}
}  // namespace geosun

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
  int max_batch = 1;
  size_t prefiltered_capacity = 0;
  const MonocularLens* cached_lens = nullptr;
  std::unique_ptr<DeviceBuffer> d_prefiltered, d_source_params, d_points,
      d_normals, d_masks, d_face_valid, d_metric_scales, d_shift_samples, d_shifts, d_shift_ok,
      d_to_camera, d_out_distance, d_out_normal, d_valid_counts;
  std::shared_ptr<DeviceBuffer> d_all_face_inputs, d_face_input;
  std::unique_ptr<HostBuffer> h_prefiltered, h_metric_scales, h_shifts, h_shift_ok,
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
  void Reserve(int count, int size, int batch_cap) {
    if (count <= faces && batch_cap <= max_batch)
      return;
    const size_t face_pixels = size_t(size) * size, total_pixels = size_t(count) * face_pixels;
    d_source_params = std::make_unique<DeviceBuffer>(8 * sizeof(double));
    d_all_face_inputs = std::make_shared<DeviceBuffer>(total_pixels * 3 * sizeof(float));
    d_face_input = std::make_shared<DeviceBuffer>(size_t(batch_cap) * face_pixels * 3 * sizeof(float));
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
    h_metric_scales = std::make_unique<HostBuffer>(size_t(count) * sizeof(float));
    h_shifts = std::make_unique<HostBuffer>(size_t(count) * sizeof(double));
    h_shift_ok = std::make_unique<HostBuffer>(size_t(count));
    h_to_camera = std::make_unique<HostBuffer>(size_t(count) * 9 * sizeof(double));
    h_out_distance = std::make_unique<HostBuffer>(total_pixels * sizeof(float));
    h_out_normal = std::make_unique<HostBuffer>(total_pixels * 3 * sizeof(float));
    h_valid_counts = std::make_unique<HostBuffer>(size_t(count) * sizeof(uint32_t));
    faces = count;
    max_batch = batch_cap;
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
  int max_batch = models_.MaxBatch("geometry");
  if (max_batch > 4) {
    size_t free_bytes = 0, total_bytes = 0;
    CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < size_t(5) * 1024 * 1024 * 1024)
      max_batch = 4;
  }
  max_batch = std::max(1, std::min(max_batch, faces));
  workspace_->Reserve(faces, size, max_batch);
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
  for (int start = 0; start < faces; start += max_batch) {
    const int batch = std::min(max_batch, faces - start);
    const std::vector<int64_t> map{batch, size, size, 3};
    Tensor input;
    input.shape = {batch, 3, size, size};
    if (start == 0 && batch == faces) {
      input.device = workspace_->d_all_face_inputs;
    } else {
      CudaCheck(cudaMemcpyAsync(workspace_->d_face_input->data,
                                static_cast<const float*>(workspace_->d_all_face_inputs->data) +
                                    size_t(start) * 3 * face_pixels,
                                size_t(batch) * 3 * face_pixels * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
      input.device = workspace_->d_face_input;
    }
    const auto outputs = models_.Predict("geometry", {input}, true);
    if (outputs.size() != 4 || outputs[0].shape != map || outputs[1].shape != map ||
        outputs[2].shape != std::vector<int64_t>{batch, size, size} ||
        outputs[3].shape != std::vector<int64_t>{batch} || !outputs[0].device ||
        !outputs[1].device || !outputs[2].device || !outputs[3].device)
      throw std::runtime_error("Geometry engine output contract differs");
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_points->data) + size_t(start) * face_pixels * 3,
        outputs[0].device->data, size_t(batch) * face_pixels * 3 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_normals->data) + size_t(start) * face_pixels * 3,
        outputs[1].device->data, size_t(batch) * face_pixels * 3 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CudaCheck(cudaMemcpyAsync(
        static_cast<float*>(workspace_->d_masks->data) + size_t(start) * face_pixels,
        outputs[2].device->data, size_t(batch) * face_pixels * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CudaCheck(cudaMemcpyAsync(static_cast<float*>(workspace_->d_metric_scales->data) + start,
                              outputs[3].device->data, size_t(batch) * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));
  }
  supervision_gpu::SampleMogeShiftGrid(
      static_cast<const float*>(workspace_->d_points->data),
      static_cast<const float*>(workspace_->d_masks->data),
      static_cast<const uint8_t*>(workspace_->d_face_valid->data), faces, size,
      static_cast<float4*>(workspace_->d_shift_samples->data), stream);
  supervision_gpu::SolveMogeShifts(static_cast<const float4*>(workspace_->d_shift_samples->data),
                                   faces, size, static_cast<double*>(workspace_->d_shifts->data),
                                   static_cast<uint8_t*>(workspace_->d_shift_ok->data), stream);
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
  CudaCheck(cudaMemcpyAsync(workspace_->h_metric_scales->data, workspace_->d_metric_scales->data,
                            size_t(faces) * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_shifts->data, workspace_->d_shifts->data,
                            size_t(faces) * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_shift_ok->data, workspace_->d_shift_ok->data,
                            size_t(faces), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_out_distance->data, workspace_->d_out_distance->data,
                            size_t(faces) * face_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_out_normal->data, workspace_->d_out_normal->data,
                            size_t(faces) * face_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CudaCheck(cudaMemcpyAsync(workspace_->h_valid_counts->data, workspace_->d_valid_counts->data,
                            size_t(faces) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  const auto* metric_scales = static_cast<const float*>(workspace_->h_metric_scales->data);
  const auto* shifts = static_cast<const double*>(workspace_->h_shifts->data);
  const auto* shift_ok = static_cast<const uint8_t*>(workspace_->h_shift_ok->data);
  for (int f = 0; f < faces; ++f) {
    if (!std::isfinite(metric_scales[f]) || metric_scales[f] <= 0)
      throw std::runtime_error("Geometry engine returned an invalid metric scale");
  }
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
constexpr size_t MinimumFaceLasSamples = 256;
constexpr float MinimumFaceFitWeight = 0.1f;
constexpr size_t MinimumBilateralLasSamples = 256;
constexpr double BilateralLogGate = 0.45;
constexpr double BilateralCauchyScale = 0.18;
constexpr double BilateralSigmaR = 0.25;
constexpr double BilateralDamping = 2.0;
constexpr double BilateralMaxLogShift = 0.45;

cv::Mat CleanLasDepth(const cv::Mat& las_depth, const cv::Mat& allowed) {
  if (las_depth.empty())
    return {};
  if (las_depth.type() != CV_32FC1 || las_depth.size() != allowed.size())
    throw std::runtime_error("LAS depth map differs from the supervision grid");
  const int h = las_depth.rows, w = las_depth.cols;
  cv::Mat las_inf(h, w, CV_32FC1, cv::Scalar(1e9f));
  for (int y = 0; y < h; ++y) {
    const float* src = las_depth.ptr<float>(y);
    const uint8_t* allow = allowed.ptr<uint8_t>(y);
    float* dst = las_inf.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const float d = src[x];
      if (allow[x] && std::isfinite(d) && d >= 0.20f && d <= 60.0f)
        dst[x] = d;
    }
  }
  cv::Mat las_min;
  cv::erode(las_inf, las_min, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
  cv::Mat las_clean = cv::Mat::zeros(h, w, CV_32FC1);
  for (int y = 0; y < h; ++y) {
    const float* src = las_inf.ptr<float>(y);
    const float* mmin = las_min.ptr<float>(y);
    float* dst = las_clean.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const float d = src[x];
      if (d < 1e8f && d <= mmin[x] * 1.08f + 0.08f)
        dst[x] = d;
    }
  }
  return las_clean;
}

struct LasPair {
  double d_face = 0.0;
  double d_las = 0.0;
};

void FitFaceToLas(const std::vector<FacePixelSample>& list, const uint8_t* allowed_ptr,
                  const float* las_clean_ptr, double& out_a, double& out_b, bool& out_aligned) {
  out_a = 1.0;
  out_b = 0.0;
  out_aligned = false;
  if (!las_clean_ptr)
    return;

  std::vector<LasPair> pairs;
  pairs.reserve(list.size() / 4);
  for (const auto& item : list) {
    if (!allowed_ptr[item.pixel] || item.weight < MinimumFaceFitWeight ||
        !(item.distance > 0.05f) || !std::isfinite(item.distance))
      continue;
    const float d_las = las_clean_ptr[item.pixel];
    if (d_las > 0.f)
      pairs.push_back({double(item.distance), double(d_las)});
  }
  if (pairs.size() < MinimumFaceLasSamples)
    return;

  std::vector<double> buf(pairs.size());
  for (size_t k = 0; k < pairs.size(); ++k)
    buf[k] = pairs[k].d_las / pairs[k].d_face;
  const double a_init = Quantile(buf, 0.5);
  if (!std::isfinite(a_init) || a_init <= 0.05 || a_init >= 20.0)
    return;

  std::vector<LasPair> inliers;
  inliers.reserve(pairs.size());
  for (const auto& p : pairs) {
    const double r = p.d_las / p.d_face;
    if (r >= 0.35 * a_init && r <= 2.8 * a_init)
      inliers.push_back(p);
  }
  if (inliers.size() < MinimumFaceLasSamples)
    return;

  std::vector<double> ratios(inliers.size());
  for (size_t k = 0; k < inliers.size(); ++k)
    ratios[k] = inliers[k].d_las / inliers[k].d_face;
  const double a_scale = Quantile(ratios, 0.5);
  if (!std::isfinite(a_scale) || a_scale <= 0.05 || a_scale >= 20.0)
    return;

  std::vector<double> errs(inliers.size()), las_vals(inliers.size()), df_vals(inliers.size());
  for (size_t k = 0; k < inliers.size(); ++k) {
    errs[k] = std::abs(a_scale * inliers[k].d_face - inliers[k].d_las) / inliers[k].d_las;
    las_vals[k] = inliers[k].d_las;
    df_vals[k] = inliers[k].d_face;
  }
  const double rel_scale = Quantile(errs, 0.5);
  out_a = a_scale;
  out_b = 0.0;
  out_aligned = true;

  const double las_p10 = Quantile(las_vals, 0.10);
  const double las_p50 = Quantile(las_vals, 0.50);
  const double las_p90 = Quantile(las_vals, 0.90);
  const double df_p01 = Quantile(df_vals, 0.01);
  if (!(las_p90 / std::max(las_p10, 0.1) >= 1.25 && las_p90 - las_p10 >= 0.40))
    return;

  double a = a_scale, b = 0.0;
  std::vector<double> abs_err(inliers.size());
  for (int iter = 0; iter < 8; ++iter) {
    for (size_t k = 0; k < inliers.size(); ++k)
      abs_err[k] = std::abs(a * inliers[k].d_face + b - inliers[k].d_las);
    const double scale = std::max(0.08, 2.0 * Quantile(abs_err, 0.5));
    double sxx = 0.0, sx1 = 0.0, s11 = 0.0, sxy = 0.0, sy = 0.0;
    for (size_t k = 0; k < inliers.size(); ++k) {
      const double u = abs_err[k] / scale;
      const double w = 1.0 / (1.0 + u * u);
      const double x = inliers[k].d_face, y = inliers[k].d_las;
      sxx += w * x * x;
      sx1 += w * x;
      s11 += w;
      sxy += w * x * y;
      sy += w * y;
    }
    const double det = sxx * s11 - sx1 * sx1;
    if (!(det > 1e-12))
      return;
    a = (sxy * s11 - sy * sx1) / det;
    b = (sxx * sy - sx1 * sxy) / det;
  }

  for (size_t k = 0; k < inliers.size(); ++k)
    errs[k] = std::abs(a * inliers[k].d_face + b - inliers[k].d_las) / inliers[k].d_las;
  const double rel_affine = Quantile(errs, 0.5);
  if (std::isfinite(a) && std::isfinite(b) && a > 0.25 * a_scale && a < 2.5 * a_scale &&
      std::abs(b) <= std::min(1.5, 0.5 * las_p50) && (a * df_p01 + b) > 0.05 &&
      rel_affine < 0.98 * rel_scale) {
    out_a = a;
    out_b = b;
  }
}

void BlurGrid3D(std::vector<double>& grid, int gy, int gx, int gz, int passes) {
  std::vector<double> tmp(grid.size());
  const size_t stride_y = size_t(gx) * size_t(gz);
  const size_t stride_x = size_t(gz);
  for (int pass = 0; pass < passes; ++pass) {
    for (int y = 0; y < gy; ++y) {
      const int y0 = std::max(0, y - 1), y2 = std::min(gy - 1, y + 1);
      for (int x = 0; x < gx; ++x)
        for (int z = 0; z < gz; ++z) {
          const size_t off = size_t(x) * stride_x + size_t(z);
          tmp[size_t(y) * stride_y + off] = 0.25 * grid[size_t(y0) * stride_y + off] +
                                            0.50 * grid[size_t(y) * stride_y + off] +
                                            0.25 * grid[size_t(y2) * stride_y + off];
        }
    }
    for (int y = 0; y < gy; ++y)
      for (int x = 0; x < gx; ++x) {
        const int x0 = std::max(0, x - 1), x2 = std::min(gx - 1, x + 1);
        const size_t row = size_t(y) * stride_y;
        for (int z = 0; z < gz; ++z)
          grid[row + size_t(x) * stride_x + size_t(z)] =
              0.25 * tmp[row + size_t(x0) * stride_x + size_t(z)] +
              0.50 * tmp[row + size_t(x) * stride_x + size_t(z)] +
              0.25 * tmp[row + size_t(x2) * stride_x + size_t(z)];
      }
    for (int y = 0; y < gy; ++y)
      for (int x = 0; x < gx; ++x) {
        const size_t base = size_t(y) * stride_y + size_t(x) * stride_x;
        for (int z = 0; z < gz; ++z) {
          const int z0 = std::max(0, z - 1), z2 = std::min(gz - 1, z + 1);
          tmp[base + size_t(z)] = 0.25 * grid[base + size_t(z0)] +
                                  0.50 * grid[base + size_t(z)] +
                                  0.25 * grid[base + size_t(z2)];
        }
      }
    grid.swap(tmp);
  }
}

struct BilateralStats {
  size_t las_valid_pixels = 0;
  size_t las_inlier_pixels = 0;
  bool bilateral_applied = false;
  double las_median_abs_err_m = 0.0;
};

BilateralStats ApplyBilateralLasRefinement(int w, int h, const uint8_t* allowed_ptr,
                                           const float* las_clean_ptr,
                                           const std::vector<double>& merged_weight,
                                           std::vector<double>& merged) {
  BilateralStats stats;
  if (!las_clean_ptr)
    return stats;

  const size_t pixels = size_t(w) * size_t(h);
  std::vector<double> valid_log_d;
  std::vector<size_t> las_pixels;
  valid_log_d.reserve(pixels);
  for (size_t i = 0; i < pixels; ++i) {
    if (!allowed_ptr[i] || !(merged_weight[i] > 0.0) || !std::isfinite(merged[i]) ||
        !(merged[i] > 0.05))
      continue;
    valid_log_d.push_back(std::log(merged[i]));
    if (las_clean_ptr[i] > 0.f)
      las_pixels.push_back(i);
  }
  stats.las_valid_pixels = las_pixels.size();
  if (las_pixels.size() < MinimumBilateralLasSamples || valid_log_d.size() < MinimumBilateralLasSamples)
    return stats;

  const double ld_min = Quantile(valid_log_d, 0.005);
  const double ld_max = std::max(ld_min + 0.25, Quantile(valid_log_d, 0.995));
  const double sigma_s = std::max(16.0, double(std::max(w, h)) / 32.0);
  const int gy = int(std::ceil(double(h) / sigma_s)) + 2;
  const int gx = int(std::ceil(double(w) / sigma_s)) + 2;
  const int gz = std::clamp(int(std::ceil((ld_max - ld_min) / BilateralSigmaR)) + 3, 4, 48);
  const double sigma_r = (ld_max - ld_min) / double(std::max(1, gz - 3));
  const size_t stride_y = size_t(gx) * size_t(gz);
  const size_t stride_x = size_t(gz);

  std::vector<double> grid_num(size_t(gy) * stride_y, 0.0);
  std::vector<double> grid_den(size_t(gy) * stride_y, 0.0);
  std::vector<size_t> inlier_pixels;
  inlier_pixels.reserve(las_pixels.size());

  for (const size_t i : las_pixels) {
    const double d_base = merged[i];
    const double r = std::log(double(las_clean_ptr[i]) / d_base);
    if (std::abs(r) > BilateralLogGate)
      continue;
    inlier_pixels.push_back(i);
    const double u = r / BilateralCauchyScale;
    const double w_pix = 1.0 / (1.0 + u * u);
    const int y = int(i / size_t(w)), x = int(i % size_t(w));
    const double fy = std::clamp(double(y) / sigma_s, 0.0, double(gy) - 1.001);
    const double fx = std::clamp(double(x) / sigma_s, 0.0, double(gx) - 1.001);
    const double fz = std::clamp((std::log(d_base) - ld_min) / sigma_r, 0.0, double(gz) - 1.001);
    const int iy = int(fy), ix = int(fx), iz = int(fz);
    const double dy = fy - iy, dx = fx - ix, dz = fz - iz;
    const double wy[2] = {1.0 - dy, dy};
    const double wx[2] = {1.0 - dx, dx};
    const double wz[2] = {1.0 - dz, dz};
    for (int oy = 0; oy < 2; ++oy)
      for (int ox = 0; ox < 2; ++ox) {
        const size_t base = size_t(iy + oy) * stride_y + size_t(ix + ox) * stride_x + size_t(iz);
        const double wxy = w_pix * wy[oy] * wx[ox];
        for (int oz = 0; oz < 2; ++oz) {
          const double ww = wxy * wz[oz];
          grid_num[base + size_t(oz)] += ww * r;
          grid_den[base + size_t(oz)] += ww;
        }
      }
  }
  stats.las_inlier_pixels = inlier_pixels.size();

  if (inlier_pixels.size() >= MinimumBilateralLasSamples) {
    BlurGrid3D(grid_num, gy, gx, gz, 2);
    BlurGrid3D(grid_den, gy, gx, gz, 2);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * size_t(w) + size_t(x);
        if (!allowed_ptr[i] || !(merged_weight[i] > 0.0) || !std::isfinite(merged[i]) ||
            !(merged[i] > 0.05))
          continue;
        const double fy = std::clamp(double(y) / sigma_s, 0.0, double(gy) - 1.001);
        const double fx = std::clamp(double(x) / sigma_s, 0.0, double(gx) - 1.001);
        const double fz =
            std::clamp((std::log(merged[i]) - ld_min) / sigma_r, 0.0, double(gz) - 1.001);
        const int iy = int(fy), ix = int(fx), iz = int(fz);
        const double dy = fy - iy, dx = fx - ix, dz = fz - iz;
        const double wy[2] = {1.0 - dy, dy};
        const double wx[2] = {1.0 - dx, dx};
        const double wz[2] = {1.0 - dz, dz};
        double s_num = 0.0, s_den = 0.0;
        for (int oy = 0; oy < 2; ++oy)
          for (int ox = 0; ox < 2; ++ox) {
            const size_t base =
                size_t(iy + oy) * stride_y + size_t(ix + ox) * stride_x + size_t(iz);
            const double wxy = wy[oy] * wx[ox];
            for (int oz = 0; oz < 2; ++oz) {
              const double ww = wxy * wz[oz];
              s_num += ww * grid_num[base + size_t(oz)];
              s_den += ww * grid_den[base + size_t(oz)];
            }
          }
        const double s =
            s_den > 1e-6
                ? std::clamp(s_num / (s_den + BilateralDamping), -BilateralMaxLogShift,
                             BilateralMaxLogShift)
                : 0.0;
        merged[i] *= std::exp(s);
      }
    stats.bilateral_applied = true;
  }

  if (!inlier_pixels.empty()) {
    std::vector<double> abs_errs(inlier_pixels.size());
    for (size_t k = 0; k < inlier_pixels.size(); ++k) {
      const size_t i = inlier_pixels[k];
      abs_errs[k] = std::abs(merged[i] - double(las_clean_ptr[i]));
    }
    stats.las_median_abs_err_m = Quantile(abs_errs, 0.5);
  }
  return stats;
}
}  // namespace

MogeSsiResult EncodeMogeSsiMaps(const MonocularLens& lens,
                                const std::vector<FaceEstimate>& faces,
                                const cv::Mat& allowed,
                                const Eigen::Vector3d* omega,
                                int axis,
                                const cv::Mat* las_depth) {
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
  const cv::Mat las_clean =
      las_depth && !las_depth->empty() ? CleanLasDepth(*las_depth, allowed_contig) : cv::Mat();
  const float* las_clean_ptr = las_clean.empty() ? nullptr : las_clean.ptr<float>();
  const size_t pixels = size_t(w) * h;
  const bool rolling = omega && !omega->isZero();

  std::vector<std::vector<FacePixelSample>> face_samples(face_count);
  for (auto& list : face_samples)
    list.reserve(pixels / std::max<size_t>(1, face_count / 2));

  std::vector<cv::Vec3f> blended_normals(pixels, cv::Vec3f(0.f, 0.f, 0.f));
  std::vector<uint8_t> has_normal(pixels, 0);

  std::vector<Eigen::Matrix3d> r_neg, r_pos;
  if (rolling) {
    const int axis_len = axis ? h : w;
    r_neg.resize(size_t(axis_len));
    r_pos.resize(size_t(axis_len));
    for (int c = 0; c < axis_len; ++c) {
      const double s = (c + 0.5 - 0.5 * axis_len) / axis_len;
      const Eigen::Vector3d w_rot = s * (*omega);
      r_neg[size_t(c)] = RodriguesMatrix(-w_rot);
      r_pos[size_t(c)] = RodriguesMatrix(w_rot);
    }
  }

  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const auto& r_cv = lens.rays.at<cv::Vec3f>(y, x);
      if (r_cv[0] == 0.f && r_cv[1] == 0.f && r_cv[2] == 0.f)
        continue;
      const Eigen::Vector3d r_exp(r_cv[0], r_cv[1], r_cv[2]);

      std::array<FaceSample, MonocularLens::MaxSamples> samples;
      int count = 0;
      const int line = axis ? y : x;
      if (rolling) {
        const Eigen::Vector3d r_ref = (r_neg[size_t(line)] * r_exp).normalized();
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
        normal = (r_pos[size_t(line)] * normal).normalized();
      if (normal.dot(r_exp) > 0.0)
        normal = -normal;
      blended_normals[i] = cv::Vec3f(float(normal.x()), float(normal.y()), float(normal.z()));
      has_normal[i] = 1;
    }

  // Align each face to cleaned LAS metric ray distance when supported, then merge remaining faces
  // via overlapping median scale ratios against already merged faces.
  std::vector<double> face_a(face_count, 1.0);
  std::vector<double> face_b(face_count, 0.0);
  std::vector<uint8_t> face_las_aligned(face_count, 0);
  size_t las_aligned_faces = 0;
  for (size_t f = 0; f < face_count; ++f) {
    bool aligned = false;
    FitFaceToLas(face_samples[f], allowed_ptr, las_clean_ptr, face_a[f], face_b[f], aligned);
    if (aligned) {
      face_las_aligned[f] = 1;
      ++las_aligned_faces;
    }
  }

  std::vector<double> merged(pixels, 0.0);
  std::vector<double> merged_weight(pixels, 0.0);
  auto merge_face = [&](size_t f, bool use_shift) {
    const double a = face_a[f], b = face_b[f];
    for (const auto& item : face_samples[f]) {
      if (!(item.distance > 0.f) || !std::isfinite(item.distance) || !(item.weight > 0.f))
        continue;
      const double d = use_shift ? std::max(0.05, a * double(item.distance) + b)
                                 : a * double(item.distance);
      const double total = merged_weight[item.pixel] + double(item.weight);
      merged[item.pixel] =
          (merged[item.pixel] * merged_weight[item.pixel] + d * double(item.weight)) / total;
      merged_weight[item.pixel] = total;
    }
  };

  for (size_t f = 0; f < face_count; ++f)
    if (face_las_aligned[f])
      merge_face(f, true);

  std::vector<float> ratios;
  for (size_t f = 0; f < face_count; ++f) {
    if (face_las_aligned[f])
      continue;
    const auto& list = face_samples[f];
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
      face_a[f] = double(*mid);
    }
    merge_face(f, false);
  }

  const BilateralStats bilateral =
      ApplyBilateralLasRefinement(w, h, allowed_ptr, las_clean_ptr, merged_weight, merged);

  std::vector<double> valid_depths;
  valid_depths.reserve(pixels);
  for (size_t i = 0; i < pixels; ++i) {
    if (!allowed_ptr[i] || !(merged_weight[i] > 0.0) || !std::isfinite(merged[i]) ||
        !(merged[i] > 0.0))
      continue;
    valid_depths.push_back(merged[i]);
  }

  MogeSsiResult out;
  out.depth_u16 = cv::Mat::zeros(h, w, CV_16UC1);
  out.normal_rgb = cv::Mat::zeros(h, w, CV_8UC3);
  out.face_scales = face_a;

  if (!valid_depths.empty()) {
    const double min_d = std::max(0.05, Quantile(valid_depths, 0.001));
    const double max_d = std::max(min_d + 1e-3, Quantile(valid_depths, 0.999));
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
        const double clamped = std::clamp(merged[i], min_d, max_d);
        const double normalized = 0.02 + 0.98 * ((clamped - min_d) / denom);
        out.depth_u16.at<uint16_t>(y, x) =
            uint16_t(std::clamp(std::lround(normalized * 65535.0), 1311L, 65535L));
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
                         {"aligned_scale", face_a[f]},
                         {"aligned_shift", face_b[f]},
                         {"las_aligned", face_las_aligned[f] != 0}});
  }
  out.report = {{"min_distance_m", out.min_distance},
                {"max_distance_m", out.max_distance},
                {"face_scales", face_a},
                {"faces", std::move(face_rows)},
                {"las_alignment",
                 {{"las_valid_pixels", bilateral.las_valid_pixels},
                  {"las_inlier_pixels", bilateral.las_inlier_pixels},
                  {"las_aligned_faces", las_aligned_faces},
                  {"bilateral_applied", bilateral.bilateral_applied},
                  {"las_median_abs_err_m", bilateral.las_median_abs_err_m}}}};
  return out;
}
}  // namespace geosun

#include "geometry/monocular.h"
#include "geometry/supervision.h"
#include "geometry/supervision_kernels.h"
#include "io/scene.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include <tuple>

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
// LAS scale fit: lattice step in grid pixels, Cauchy scale and inlier bound in log distance,
// overlap residual weight against a LAS residual, and the photo acceptance.
constexpr int FitLattice = 2;
constexpr double CauchyScale = .03, InlierLog = .05, OverlapWeight = .25;
// Node first-difference weight per mean LAS anchor of a field node.
constexpr double FieldSmoothing = .1;
constexpr size_t MinimumImageAnchors = 2000;
constexpr double MinimumInlierFraction = .5;
// Encoding: depth step between neighbours, LAS agreement and the delivered depth range.
constexpr double StepLog = .04, AgreementFraction = .03, AgreementMetres = .03;
// LAS depth is trusted only where its surfel normal is within this angle of MoGe's normal: two
// independent estimates of the same surface. Gaps showing the background, stray or transient
// returns and silhouette spread rarely agree, so they neither shape the fit nor veto MoGe.
constexpr double TrustedNormalDegrees = 15;
constexpr double MinimumDepth = .05, MaximumDepth = 65.535;
// Cross-view free-space check: views sharing most tracks, lattice step in grid pixels, and the log
// distance by which a point must lie in front of the other view's depth to contradict it.
constexpr size_t ConflictNeighbours = 12, ConflictSharedTracks = 30;
constexpr double ConflictBaselineMetres = .05, ConflictFieldOfViewDegrees = 160;
constexpr int ConflictLattice = 4;
constexpr double ConflictLog = .05;
constexpr int ConflictMinimumViews = 2;

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
}  // namespace
MonocularMaps GatherFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                          const std::function<double(const FaceSample&)>& log_scale,
                          const Eigen::Vector3d* omega, int axis) {
  if (faces.size() != lens.faces.size())
    throw std::runtime_error("Monocular faces differ from the lens plan");
  const int w = lens.grid.width, h = lens.grid.height;
  const bool rolling = omega && !omega->isZero();
  MonocularMaps maps{cv::Mat::zeros(h, w, CV_32FC1), cv::Mat::zeros(h, w, CV_32FC3)};
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
      double weights = 0, log_distance = 0;
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      for (int k = 0; k < count; ++k) {
        const auto& sample = samples[k];
        double distance;
        Eigen::Vector3d n;
        if (!SampleFace(faces[sample.face], sample, distance, n))
          continue;
        weights += sample.weight;
        log_distance += sample.weight * (std::log(distance) + log_scale(sample));
        normal += sample.weight * n;
      }
      if (weights <= 0 || normal.norm() < 1e-6)
        continue;
      maps.distance.at<float>(y, x) = float(std::exp(log_distance / weights));
      normal.normalize();
      if (rolling)
        normal = (normal + w_rot.cross(normal)).normalized();
      maps.normal.at<cv::Vec3f>(y, x) =
          cv::Vec3f(float(normal.x()), float(normal.y()), float(normal.z()));
    }
  return maps;
}
namespace {
// Bilinear field node indices and weights of a face sample.
struct FieldCorners {
  int node[4];
  double weight[4];
};
// las_normal is the LAS supervision encoding (B, G, R = z, y, x; black where none).
bool Trusted(const cv::Mat& las_normal, int x, int y, const Eigen::Vector3d& normal) {
  const auto& code = las_normal.at<cv::Vec3b>(y, x);
  if (!code[0] && !code[1] && !code[2])
    return false;
  const Eigen::Vector3d las(code[2] / 127.5 - 1, code[1] / 127.5 - 1, code[0] / 127.5 - 1);
  return normal.dot(las) >=
         std::cos(TrustedNormalDegrees * Pi / 180) * normal.norm() * las.norm();
}
FieldCorners Corners(const FaceSample& sample, int size) {
  constexpr int n = FaceFit::Cells + 1;
  const double u = sample.x / (size - 1) * FaceFit::Cells, v = sample.y / (size - 1) * FaceFit::Cells;
  const int c = std::clamp(int(u), 0, FaceFit::Cells - 1), r = std::clamp(int(v), 0, FaceFit::Cells - 1);
  const double fu = u - c, fv = v - r;
  const int base = sample.face * n * n + r * n + c;
  return {{base, base + 1, base + n, base + n + 1},
          {(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv}};
}
}  // namespace
double FaceFit::LogScale(const FaceSample& sample, int size) const {
  constexpr int n = Cells + 1;
  const auto corners = Corners(sample, size);
  double value = 0;
  for (int k = 0; k < 4; ++k)
    value += corners.weight[k] * fields[sample.face][corners.node[k] - sample.face * n * n];
  return value;
}
Json FaceFit::ToJson() const {
  return {{"accepted", accepted}, {"las_samples", las_samples}, {"anchors", anchors},
          {"inliers", inliers},
          {"field_cells", Cells}, {"scales", scales}, {"faces", faces}};
}
FaceFit FitFaceScales(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                      const cv::Mat& las, const cv::Mat& las_normal,
                      const Eigen::Vector3d* omega, int axis) {
  if (faces.size() != lens.faces.size() || las.type() != CV_16UC1 ||
      las.cols != lens.grid.width || las.rows != lens.grid.height ||
      las_normal.type() != CV_8UC3 || las_normal.size() != las.size())
    throw std::runtime_error("LAS depth or monocular faces differ from the lens plan");
  struct Anchor {
    FaceSample sample;
    float value;  // log(LAS distance / face distance)
  };
  struct Tie {
    FaceSample a, b;
    float value;  // log(face b distance / face a distance)
  };
  std::vector<Anchor> anchors;
  std::vector<Tie> ties;
  FaceFit fit;
  const int w = lens.grid.width, h = lens.grid.height;
  const bool rolling = omega && !omega->isZero();
  for (int y = 0; y < h; y += FitLattice)
    for (int x = 0; x < w; x += FitLattice) {
      const size_t i = size_t(y) * w + x;
      std::array<FaceSample, MonocularLens::MaxSamples> samples;
      int sample_count = 0;
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
        sample_count = lens.SamplesAtRay(r_ref, samples);
      } else {
        samples = lens.samples[i];
        sample_count = lens.counts[i];
      }
      double distance[MonocularLens::MaxSamples];
      bool valid[MonocularLens::MaxSamples];
      Eigen::Vector3d normal[MonocularLens::MaxSamples];
      for (int k = 0; k < sample_count; ++k) {
        Eigen::Vector3d n_ref;
        valid[k] = SampleFace(faces[samples[k].face], samples[k], distance[k], n_ref);
        if (valid[k])
          normal[k] = rolling ? (n_ref + w_rot.cross(n_ref)).normalized() : n_ref;
      }
      const double target = las.at<uint16_t>(y, x) / 1000.;
      for (int a = 0; a < sample_count; ++a) {
        if (!valid[a])
          continue;
        if (target > 0) {
          ++fit.las_samples;
          if (Trusted(las_normal, x, y, normal[a]))
            anchors.push_back({samples[a], float(std::log(target / distance[a]))});
        }
        for (int b = a + 1; b < sample_count; ++b)
          if (valid[b])
            ties.push_back({samples[a], samples[b], float(std::log(distance[b] / distance[a]))});
      }
    }
  const size_t count = faces.size();
  constexpr int n = FaceFit::Cells + 1;
  fit.scales.assign(count, 1.);
  fit.fields.assign(count, std::vector<double>(n * n, 0.));
  fit.anchors = anchors.size();
  fit.faces = Json::array();
  const size_t min_anchors = std::max<size_t>(
      128, size_t(std::lround(MinimumImageAnchors * (double(lens.grid.width) * lens.grid.height) /
                              (1600. * 1600.))));
  if (anchors.size() < min_anchors)
    return fit;
  auto weight = [](double residual) {
    const double r = residual / CauchyScale;
    return 1 / (1 + r * r);
  };
  // One scale per face first; a small pull to the current value keeps every face defined.
  std::vector<double> start;
  for (const auto& anchor : anchors)
    start.push_back(anchor.value);
  Eigen::VectorXd logs = Eigen::VectorXd::Constant(count, Quantile(start, .5));
  for (int iteration = 0; iteration < 50; ++iteration) {
    Eigen::MatrixXd normal = Eigen::MatrixXd::Identity(count, count) * 1e-6;
    Eigen::VectorXd right = logs * 1e-6;
    for (const auto& anchor : anchors) {
      const int f = anchor.sample.face;
      const double wi = weight(logs[f] - anchor.value);
      normal(f, f) += wi;
      right[f] += wi * anchor.value;
    }
    for (const auto& tie : ties) {
      const int a = tie.a.face, b = tie.b.face;
      const double wi = OverlapWeight * weight(logs[a] - logs[b] - tie.value);
      normal(a, a) += wi;
      normal(b, b) += wi;
      normal(a, b) -= wi;
      normal(b, a) -= wi;
      right[a] += wi * tie.value;
      right[b] -= wi * tie.value;
    }
    const Eigen::VectorXd next = normal.ldlt().solve(right);
    const double change = (next - logs).cwiseAbs().maxCoeff();
    logs = next;
    if (!(change > 1e-7))
      break;
  }
  // The fields refine those scales. Node neighbours are tied with a weight relative to the mean
  // number of anchors per node, so LAS-rich cells follow the LAS and empty cells follow the face.
  const int size = lens.size, nodes = int(count) * n * n;
  Eigen::VectorXd field(nodes);
  for (size_t f = 0; f < count; ++f)
    field.segment(f * n * n, n * n).setConstant(logs[f]);
  const double smooth = FieldSmoothing * double(anchors.size()) / nodes;
  std::vector<FieldCorners> anchor_corners, tie_a, tie_b;
  for (const auto& anchor : anchors)
    anchor_corners.push_back(Corners(anchor.sample, size));
  for (const auto& tie : ties) {
    tie_a.push_back(Corners(tie.a, size));
    tie_b.push_back(Corners(tie.b, size));
  }
  auto evaluate = [&](const FieldCorners& c) {
    double value = 0;
    for (int k = 0; k < 4; ++k)
      value += c.weight[k] * field[c.node[k]];
    return value;
  };
  // The normal matrix is accumulated densely (a few MB) and solved sparse: nodes couple only
  // within a cell, to their neighbours and across overlapping faces.
  Eigen::MatrixXd dense(nodes, nodes);
  for (int iteration = 0; iteration < 10; ++iteration) {
    dense.setZero();
    dense.diagonal().setConstant(1e-6);
    Eigen::VectorXd right = field * 1e-6;
    // Accumulate a row J (with right-hand side t) of weight wi into the normal equations.
    auto add = [&](const int* index, const double* value, int length, double t, double wi) {
      for (int p = 0; p < length; ++p) {
        right[index[p]] += wi * value[p] * t;
        for (int q = 0; q < length; ++q)
          dense(index[p], index[q]) += wi * value[p] * value[q];
      }
    };
    for (size_t i = 0; i < anchors.size(); ++i) {
      const auto& c = anchor_corners[i];
      add(c.node, c.weight, 4, anchors[i].value, weight(evaluate(c) - anchors[i].value));
    }
    for (size_t i = 0; i < ties.size(); ++i) {
      int index[8];
      double value[8];
      for (int k = 0; k < 4; ++k) {
        index[k] = tie_a[i].node[k];
        value[k] = tie_a[i].weight[k];
        index[4 + k] = tie_b[i].node[k];
        value[4 + k] = -tie_b[i].weight[k];
      }
      const double residual = evaluate(tie_a[i]) - evaluate(tie_b[i]) - ties[i].value;
      add(index, value, 8, ties[i].value, OverlapWeight * weight(residual));
    }
    for (size_t f = 0; f < count; ++f)
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
          const int node = int(f) * n * n + r * n + c;
          for (const int other : {c + 1 < n ? node + 1 : -1, r + 1 < n ? node + n : -1}) {
            if (other < 0)
              continue;
            const int index[2] = {node, other};
            const double value[2] = {1, -1};
            add(index, value, 2, 0, smooth);
          }
        }
    const Eigen::SparseMatrix<double> matrix = dense.sparseView();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(matrix);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Monocular scale field fit is singular");
    const Eigen::VectorXd next = solver.solve(right);
    const double change = (next - field).cwiseAbs().maxCoeff();
    field = next;
    if (!(change > 1e-4))
      break;
  }
  if (!field.allFinite())
    throw std::runtime_error("Monocular face scale fit diverged");
  std::vector<std::vector<double>> residuals(count);
  for (size_t i = 0; i < anchors.size(); ++i) {
    const double r = anchors[i].value - evaluate(anchor_corners[i]);
    residuals[anchors[i].sample.face].push_back(r);
    fit.inliers += std::abs(r) <= InlierLog;
  }
  for (size_t f = 0; f < count; ++f) {
    const Eigen::VectorXd nodes_f = field.segment(f * n * n, n * n);
    fit.fields[f].assign(nodes_f.data(), nodes_f.data() + n * n);
    fit.scales[f] = std::exp(nodes_f.mean());
    auto& values = residuals[f];
    Json row = {{"scale", fit.scales[f]},
                {"field_log_range", {nodes_f.minCoeff(), nodes_f.maxCoeff()}},
                {"anchors", values.size()}};
    if (!values.empty()) {
      const double median = Quantile(values, .5);
      size_t inside = 0;
      for (auto& value : values) {
        inside += std::abs(value) <= InlierLog;
        value = std::abs(value - median);
      }
      row["inlier_fraction"] = double(inside) / values.size();
      row["log_residual_median"] = median;
      row["log_residual_mad"] = Quantile(values, .5);
    }
    fit.faces.push_back(row);
  }
  fit.accepted = double(fit.inliers) >= MinimumInlierFraction * fit.anchors;
  return fit;
}
Json EncodeMonocular(const MonocularLens& lens, const MonocularMaps& maps, const cv::Mat& las,
                     const cv::Mat& las_normal, const cv::Mat& allowed, bool depth,
                     double min_incidence, cv::Mat& depth_mm, cv::Mat& normal_rgb,
                     cv::Mat& verified) {
  const int w = lens.grid.width, h = lens.grid.height;
  if (las.size() != cv::Size(w, h) || allowed.size() != las.size() || allowed.type() != CV_8UC1 ||
      maps.distance.size() != las.size() || maps.normal.size() != las.size() ||
      las_normal.size() != las.size() || las_normal.type() != CV_8UC3)
    throw std::runtime_error("Monocular maps differ from the supervision grid");
  // A pixel next to a jump of more than StepLog in log distance holds a blend of both sides.
  cv::Mat step = cv::Mat::zeros(h, w, CV_8UC1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float d = maps.distance.at<float>(y, x);
      if (d <= 0)
        continue;
      for (int dy = -1; dy <= 1 && !step.at<uint8_t>(y, x); ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h)
            continue;
          const float other = maps.distance.at<float>(yy, xx);
          if (other > 0 && std::abs(std::log(other / d)) > StepLog) {
            step.at<uint8_t>(y, x) = 1;
            break;
          }
        }
    }
  cv::Mat blended;
  cv::dilate(step, blended, cv::Mat::ones(3, 3, CV_8UC1));
  depth_mm = cv::Mat::zeros(h, w, CV_16UC1);
  normal_rgb = cv::Mat::zeros(h, w, CV_8UC3);
  verified = cv::Mat::zeros(h, w, CV_8UC1);
  std::vector<double> agreement;
  size_t disagreeing = 0, untrusted = 0, untrusted_disagreeing = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (!allowed.at<uint8_t>(y, x))
        continue;
      const auto& n = maps.normal.at<cv::Vec3f>(y, x);
      const auto& ray = lens.rays.at<cv::Vec3f>(y, x);
      if ((n[0] || n[1] || n[2]) &&
          -(n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2]) >= min_incidence)
        normal_rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(uint8_t(127.5f + 127.5f * n[2] + .5f),
                                                   uint8_t(127.5f + 127.5f * n[1] + .5f),
                                                   uint8_t(127.5f + 127.5f * n[0] + .5f));
      // A depth between both sides of a step is a point in mid air, whereas thin structures keep
      // their normals, so only depth drops the steps.
      const float d = maps.distance.at<float>(y, x);
      if (d <= 0 || blended.at<uint8_t>(y, x))
        continue;
      // Only trusted LAS depth vetoes MoGe; elsewhere the disagreement is only counted.
      const double target = las.at<uint16_t>(y, x) / 1000.;
      bool agrees = true, confirmed = false;
      if (target > 0) {
        const double ratio = std::abs(std::log(d / target));
        const bool within = ratio <= std::log1p(std::max(AgreementFraction, AgreementMetres / target));
        if (Trusted(las_normal, x, y, Eigen::Vector3d(n[0], n[1], n[2]))) {
          agreement.push_back(ratio);
          agrees = confirmed = within;
          disagreeing += !within;
        } else {
          ++untrusted;
          untrusted_disagreeing += !within;
        }
      }
      if (depth && agrees && d >= MinimumDepth && d <= MaximumDepth) {
        depth_mm.at<uint16_t>(y, x) = uint16_t(std::clamp<long>(std::lround(d * 1000.), 1, 65535));
        verified.at<uint8_t>(y, x) = confirmed ? 255 : 0;
      }
    }
  Json stats = {{"las_compared_pixels", agreement.size()},
                {"las_disagreeing_pixels", disagreeing},
                {"las_untrusted_pixels", untrusted},
                {"las_untrusted_disagreeing_pixels", untrusted_disagreeing}};
  if (!agreement.empty())
    stats["las_log_ratio"] = {{"p50", Quantile(agreement, .5)}, {"p90", Quantile(agreement, .9)}};
  return stats;
}
cv::Mat ConflictCells(const cv::Mat& pixels) {
  cv::Mat cells = cv::Mat::zeros((pixels.rows + ConflictLattice - 1) / ConflictLattice,
                                 (pixels.cols + ConflictLattice - 1) / ConflictLattice, CV_8UC1);
  for (int y = 0; y < pixels.rows; ++y)
    for (int x = 0; x < pixels.cols; ++x)
      if (pixels.at<uint8_t>(y, x))
        cells.at<uint8_t>(y / ConflictLattice, x / ConflictLattice) = 255;
  return cells;
}
namespace {
class DepthLruCache {
  const fs::path& output_;
  const Scene& scene_;
  const Reconstruction& model_;
  const std::array<Camera, 2>& grids_;
  std::list<uint32_t> order_;
  struct Item {
    cv::Mat image;
    std::list<uint32_t>::iterator pos;
  };
  std::map<uint32_t, Item> items_;

 public:
  DepthLruCache(const fs::path& output, const Scene& scene, const Reconstruction& model,
                const std::array<Camera, 2>& grids)
      : output_(output), scene_(scene), model_(model), grids_(grids) {}
  const cv::Mat& Get(uint32_t i) {
    auto it = items_.find(i);
    if (it != items_.end()) {
      order_.splice(order_.begin(), order_, it->second.pos);
      return it->second.image;
    }
    if (items_.size() >= 96) {
      items_.erase(order_.back());
      order_.pop_back();
    }
    auto depth = ReadImage(output_ / "depths" / fs::u8path(scene_.images[i].name + ".png"),
                           cv::IMREAD_UNCHANGED);
    const auto& grid = grids_[model_.images[i].camera];
    if (depth.type() != CV_16UC1 || depth.cols != grid.width || depth.rows != grid.height)
      throw std::runtime_error("Unexpected supervision depth map: " + scene_.images[i].name);
    order_.push_front(i);
    return items_.emplace(i, Item{std::move(depth), order_.begin()}).first->second.image;
  }
};
struct ConflictStream {
  cudaStream_t value = nullptr;
  ConflictStream() { CudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~ConflictStream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
};
struct StageEvents {
  static constexpr int Count = 4;
  std::array<cudaEvent_t, Count> events{};
  std::array<bool, Count> recorded{};
  StageEvents() {
    for (auto& e : events)
      CudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  ~StageEvents() {
    for (auto& e : events)
      if (e)
        cudaEventDestroy(e);
  }
};
}  // namespace
Json FilterViewConflicts(const fs::path& output, const Scene& scene, const Reconstruction& model,
                         const ShutterMotion* motion, int max_size,
                         const std::vector<cv::Mat>& verified, std::vector<Json>& rows,
                         std::vector<cv::Mat>* held_depths) {
  const auto started = std::chrono::steady_clock::now();
  const size_t n = model.images.size();
  const std::array<Camera, 2> grids{MapCamera(model.cameras[0], max_size),
                                    MapCamera(model.cameras[1], max_size)};
  std::vector<Eigen::Matrix3d> rotations(n);
  std::vector<Eigen::Vector3d> translations(n), centers(n);
  for (size_t i = 0; i < n; ++i) {
    rotations[i] = model.images[i].pose.Rotation().toRotationMatrix();
    translations[i] = model.images[i].pose.Translation();
    centers[i] = model.images[i].pose.Center();
  }
  std::vector<uint32_t> shared(n * n);
  for (const auto& point : model.points)
    for (const auto& a : point.track)
      for (const auto& b : point.track)
        if (a.image != b.image)
          ++shared[a.image * n + b.image];
  std::vector<std::vector<uint8_t>> paired(n, std::vector<uint8_t>(n));
  for (uint32_t i = 0; i < n; ++i) {
    if (!model.images[i].registered)
      continue;
    std::vector<uint32_t> others;
    for (uint32_t j = 0; j < n; ++j)
      if (i != j && model.images[j].registered && shared[i * n + j] >= ConflictSharedTracks &&
          (centers[i] - centers[j]).norm() >= ConflictBaselineMetres)
        others.push_back(j);
    std::sort(others.begin(), others.end(), [&](uint32_t a, uint32_t b) {
      return std::make_pair(shared[i * n + a], b) > std::make_pair(shared[i * n + b], a);
    });
    others.resize(std::min(others.size(), ConflictNeighbours));
    for (const uint32_t j : others)
      paired[i][j] = paired[j][i] = 1;
  }
  auto cells_of = [&](uint32_t i) {
    const auto& grid = grids[model.images[i].camera];
    return std::make_pair((grid.width + ConflictLattice - 1) / ConflictLattice,
                          (grid.height + ConflictLattice - 1) / ConflictLattice);
  };
  // Per cell, the number of views contradicting it.
  std::vector<std::vector<uint8_t>> conflicts(n);
  for (uint32_t i = 0; i < n; ++i)
    if (model.images[i].registered) {
      const auto [cw, ch] = cells_of(i);
      conflicts[i].assign(size_t(cw) * ch, 0);
      if (verified.at(i).type() != CV_8UC1 || verified[i].cols != cw || verified[i].rows != ch)
        throw std::runtime_error("LAS-verified cells differ from the conflict lattice");
      if (held_depths) {
        const auto& grid = grids[model.images[i].camera];
        const auto& depth = held_depths->at(i);
        if (depth.type() != CV_16UC1 || depth.cols != grid.width || depth.rows != grid.height)
          throw std::runtime_error("Unexpected held supervision depth map: " + scene.images[i].name);
      }
    }
  constexpr uint32_t Tile = 32;
  struct TileBatch {
    uint32_t ti, tj;
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
  };
  std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> tile_map;
  size_t total_pairs = 0;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = i + 1; j < n; ++j)
      if (paired[i][j]) {
        tile_map[{i / Tile, j / Tile}].emplace_back(i, j);
        ++total_pairs;
      }
  std::vector<TileBatch> tiles;
  tiles.reserve(tile_map.size());
  for (auto& [key, list] : tile_map)
    tiles.push_back({key.first, key.second, std::move(list)});
  for (size_t i = 0; i < tiles.size();) {
    size_t end = i + 1;
    while (end < tiles.size() && tiles[end].ti == tiles[i].ti)
      ++end;
    if (tiles[i].ti & 1u)
      std::reverse(tiles.begin() + std::ptrdiff_t(i), tiles.begin() + std::ptrdiff_t(end));
    i = end;
  }
  const double view_limit = std::cos(ConflictFieldOfViewDegrees * Pi / 360),
               tolerance = std::exp(ConflictLog);
  uint64_t tested_total = 0, conflicting_total = 0;
  if (!tiles.empty()) {
    std::array<std::vector<double3>, 2> lattice_rays;
    std::array<int, 2> cam_cw{}, cam_ch{};
    for (int c = 0; c < 2; ++c) {
      const auto& grid = grids[c];
      const double radius = grid.Radius();
      cam_cw[c] = (grid.width + ConflictLattice - 1) / ConflictLattice;
      cam_ch[c] = (grid.height + ConflictLattice - 1) / ConflictLattice;
      lattice_rays[c].assign(size_t(cam_cw[c]) * cam_ch[c], make_double3(0., 0., 0.));
      for (int cy = 0; cy < cam_ch[c]; ++cy)
        for (int cx = 0; cx < cam_cw[c]; ++cx) {
          const int x = cx * ConflictLattice, y = cy * ConflictLattice;
          const double dx = (x + .5 - grid.params[2]) / grid.params[0],
                       dy = (y + .5 - grid.params[3]) / grid.params[1];
          if (dx * dx + dy * dy <= radius * radius) {
            const auto ray = grid.Ray({x + .5, y + .5});
            lattice_rays[c][size_t(cy) * cam_cw[c] + cx] = make_double3(ray.x(), ray.y(), ray.z());
          }
        }
    }
    std::vector<supervision_gpu::ConflictView> h_views(n);
    for (size_t i = 0; i < n; ++i) {
      if (!model.images[i].registered)
        continue;
      const uint32_t cid = model.images[i].camera;
      const auto& grid = grids[cid];
      auto& v = h_views[i];
      for (int r = 0; r < 3; ++r) {
        v.translation[r] = translations[i][r];
        for (int c = 0; c < 3; ++c)
          v.rotation[r * 3 + c] = rotations[i](r, c);
      }
      std::copy(grid.params.begin(), grid.params.end(), v.params);
      if (motion) {
        const Eigen::Vector3d m_trans = motion->readout[cid] * motion->velocity[i];
        for (int k = 0; k < 3; ++k) {
          v.motion_rot[k] = motion->rotation[i][k];
          v.motion_trans[k] = m_trans[k];
        }
        v.rolling = 1;
        v.axis = motion->axis;
      }
      v.camera = int(cid);
      v.width = grid.width;
      v.height = grid.height;
      v.cw = cam_cw[cid];
      v.ch = cam_ch[cid];
    }
    const size_t max_pixels = std::max(size_t(grids[0].width) * grids[0].height,
                                       size_t(grids[1].width) * grids[1].height);
    const int max_cells = std::max(cam_cw[0] * cam_ch[0], cam_cw[1] * cam_ch[1]);
    constexpr int GpuSlots = 64, BatchPairs = 256;
    const size_t max_tile_pairs = Tile * Tile;
    DeviceBuffer d_depths(size_t(GpuSlots) * max_pixels * sizeof(uint16_t)),
        d_tile_conflicts(size_t(GpuSlots) * max_cells * sizeof(unsigned int)),
        d_pair_own(size_t(BatchPairs) * max_cells),
        d_pair_other(size_t(BatchPairs) * max_cells),
        d_pairs(max_tile_pairs * sizeof(supervision_gpu::ConflictPair)),
        d_views(n * sizeof(supervision_gpu::ConflictView)),
        d_rays0(lattice_rays[0].size() * sizeof(double3)),
        d_rays1(lattice_rays[1].size() * sizeof(double3)),
        d_stats(2 * sizeof(unsigned long long));
    HostBuffer h_depth_stage(size_t(StageEvents::Count) * max_pixels * sizeof(uint16_t)),
        h_pairs(max_tile_pairs * sizeof(supervision_gpu::ConflictPair)),
        h_tile_conflicts(size_t(GpuSlots) * max_cells * sizeof(unsigned int));
    StageEvents stage_events;
    ConflictStream stream;
    CudaCheck(cudaMemcpyAsync(d_views.data, h_views.data(), d_views.bytes, cudaMemcpyHostToDevice,
                              stream.value));
    CudaCheck(cudaMemcpyAsync(d_rays0.data, lattice_rays[0].data(), d_rays0.bytes,
                              cudaMemcpyHostToDevice, stream.value));
    CudaCheck(cudaMemcpyAsync(d_rays1.data, lattice_rays[1].data(), d_rays1.bytes,
                              cudaMemcpyHostToDevice, stream.value));
    CudaCheck(cudaMemsetAsync(d_stats.data, 0, d_stats.bytes, stream.value));
    DepthLruCache lru(output, scene, model, grids);
    std::array<int, GpuSlots> slot_image;
    slot_image.fill(-1);
    std::array<uint64_t, GpuSlots> slot_stamp{};
    uint64_t stamp = 0, stage_seq = 0;
    std::vector<int> slot_of(n, -1);
    for (const auto& tile : tiles) {
      std::vector<uint32_t> tile_images;
      tile_images.reserve(GpuSlots);
      for (const auto& [a, b] : tile.pairs) {
        if (slot_of[a] == -1) {
          slot_of[a] = -2;
          tile_images.push_back(a);
        }
        if (slot_of[b] == -1) {
          slot_of[b] = -2;
          tile_images.push_back(b);
        }
      }
      std::array<bool, GpuSlots> in_tile{};
      for (uint32_t id : tile_images)
        for (int s = 0; s < GpuSlots; ++s)
          if (slot_image[s] == int(id)) {
            slot_of[id] = s;
            in_tile[s] = true;
            break;
          }
      for (uint32_t id : tile_images) {
        if (slot_of[id] >= 0)
          continue;
        int chosen = -1;
        for (int s = 0; s < GpuSlots; ++s)
          if (!in_tile[s] && (chosen < 0 || slot_stamp[s] < slot_stamp[chosen]))
            chosen = s;
        if (chosen < 0)
          throw std::runtime_error("Conflict GPU depth cache exhausted");
        slot_image[chosen] = int(id);
        in_tile[chosen] = true;
        slot_of[id] = chosen;
        const cv::Mat& depth = held_depths ? (*held_depths)[id] : lru.Get(id);
        const int stage_idx = int(stage_seq % StageEvents::Count);
        if (stage_events.recorded[stage_idx])
          CudaCheck(cudaEventSynchronize(stage_events.events[stage_idx]));
        const size_t pixels = size_t(depth.rows) * depth.cols;
        auto* dst = static_cast<uint16_t*>(h_depth_stage.data) + size_t(stage_idx) * max_pixels;
        if (depth.isContinuous())
          std::memcpy(dst, depth.data, pixels * sizeof(uint16_t));
        else
          for (int r = 0; r < depth.rows; ++r)
            std::memcpy(dst + size_t(r) * depth.cols, depth.ptr<uint16_t>(r),
                        size_t(depth.cols) * sizeof(uint16_t));
        CudaCheck(cudaMemcpyAsync(
            static_cast<uint16_t*>(d_depths.data) + size_t(chosen) * max_pixels, dst,
            pixels * sizeof(uint16_t), cudaMemcpyHostToDevice, stream.value));
        CudaCheck(cudaEventRecord(stage_events.events[stage_idx], stream.value));
        stage_events.recorded[stage_idx] = true;
        ++stage_seq;
      }
      for (uint32_t id : tile_images)
        slot_stamp[slot_of[id]] = ++stamp;
      auto* h_pair_ptr = static_cast<supervision_gpu::ConflictPair*>(h_pairs.data);
      for (size_t k = 0; k < tile.pairs.size(); ++k) {
        const auto [a, b] = tile.pairs[k];
        h_pair_ptr[k] = {uint32_t(k), uint16_t(slot_of[a]), uint16_t(slot_of[b]), a, b};
      }
      CudaCheck(cudaMemcpyAsync(d_pairs.data, h_pair_ptr,
                                tile.pairs.size() * sizeof(supervision_gpu::ConflictPair),
                                cudaMemcpyHostToDevice, stream.value));
      CudaCheck(cudaMemsetAsync(d_tile_conflicts.data, 0,
                                size_t(GpuSlots) * max_cells * sizeof(unsigned int), stream.value));
      for (size_t base = 0; base < tile.pairs.size(); base += BatchPairs) {
        const int count = int(std::min<size_t>(BatchPairs, tile.pairs.size() - base));
        CudaCheck(cudaMemsetAsync(d_pair_own.data, 0, size_t(count) * max_cells, stream.value));
        CudaCheck(cudaMemsetAsync(d_pair_other.data, 0, size_t(count) * max_cells, stream.value));
        supervision_gpu::CheckConflictPairs(
            static_cast<const uint16_t*>(d_depths.data), max_pixels,
            static_cast<const supervision_gpu::ConflictView*>(d_views.data),
            static_cast<const double3*>(d_rays0.data), static_cast<const double3*>(d_rays1.data),
            static_cast<const supervision_gpu::ConflictPair*>(d_pairs.data) + base, count,
            max_cells, ConflictLattice, view_limit, tolerance,
            static_cast<uint8_t*>(d_pair_own.data), static_cast<uint8_t*>(d_pair_other.data),
            static_cast<unsigned int*>(d_tile_conflicts.data),
            static_cast<unsigned long long*>(d_stats.data), stream.value);
      }
      CudaCheck(cudaMemcpyAsync(h_tile_conflicts.data, d_tile_conflicts.data,
                                size_t(GpuSlots) * max_cells * sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, stream.value));
      CudaCheck(cudaStreamSynchronize(stream.value));
      const auto* tile_votes = static_cast<const unsigned int*>(h_tile_conflicts.data);
      for (uint32_t id : tile_images) {
        const int s = slot_of[id];
        const unsigned int* src = tile_votes + size_t(s) * max_cells;
        auto& dst = conflicts[id];
        for (size_t k = 0; k < dst.size(); ++k) {
          const unsigned int sum = unsigned(dst[k]) + src[k];
          dst[k] = uint8_t(std::min(255u, sum));
        }
        slot_of[id] = -1;
      }
    }
    unsigned long long stats_host[2]{};
    CudaCheck(cudaMemcpyAsync(stats_host, d_stats.data, sizeof(stats_host), cudaMemcpyDeviceToHost,
                              stream.value));
    CudaCheck(cudaStreamSynchronize(stream.value));
    tested_total = stats_host[0];
    conflicting_total = stats_host[1];
  }
  // A cell loses its depth when it is not LAS-verified and at least ConflictMinimumViews views
  // contradict it. Agreeing views do not outvote them: in a LAS hole two groups of views can each
  // agree among themselves, and nothing tells which group is right.
  const auto workers = std::min(n, size_t(std::clamp(std::thread::hardware_concurrency(), 1u, 8u)));
  std::atomic<uint32_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_lock;
  std::atomic<uint64_t> dropped_total{0}, dropped_normal_total{0};
  auto rewrite = [&] {
    try {
      for (uint32_t i; (i = next++) < n;) {
        if (!model.images[i].registered)
          continue;
        const auto [cw, ch] = cells_of(i);
        bool any_candidate = false;
        const size_t cell_count = size_t(cw) * ch;
        for (size_t k = 0; k < cell_count; ++k)
          if (!verified[i].data[k] && conflicts[i][k] >= ConflictMinimumViews) {
            any_candidate = true;
            break;
          }
        auto& row = rows.at(i);
        const auto path = output / "depths" / fs::u8path(scene.images[i].name + ".png");
        uint64_t dropped = 0, dropped_normals = 0;
        if (held_depths) {
          auto depth = std::move((*held_depths)[i]);
          if (any_candidate)
            for (int y = 0; y < depth.rows; ++y) {
              auto* d = depth.ptr<uint16_t>(y);
              for (int x = 0; x < depth.cols; ++x) {
                if (!d[x])
                  continue;
                const size_t k = size_t(y / ConflictLattice) * cw + x / ConflictLattice;
                if (!verified[i].data[k] && conflicts[i][k] >= ConflictMinimumViews) {
                  d[x] = 0;
                  ++dropped;
                }
              }
            }
          WriteImage(path, depth);
          row["depth_size"] = fs::file_size(path);
        } else if (any_candidate) {
          auto depth = ReadImage(path, cv::IMREAD_UNCHANGED);
          const auto& grid = grids[model.images[i].camera];
          if (depth.type() != CV_16UC1 || depth.cols != grid.width || depth.rows != grid.height)
            throw std::runtime_error("Unexpected supervision depth map: " + scene.images[i].name);
          for (int y = 0; y < depth.rows; ++y) {
            auto* d = depth.ptr<uint16_t>(y);
            for (int x = 0; x < depth.cols; ++x) {
              if (!d[x])
                continue;
              const size_t k = size_t(y / ConflictLattice) * cw + x / ConflictLattice;
              if (!verified[i].data[k] && conflicts[i][k] >= ConflictMinimumViews) {
                d[x] = 0;
                ++dropped;
              }
            }
          }
          if (dropped) {
            WriteImage(path, depth);
            row["depth_size"] = fs::file_size(path);
          }
        }
        if (any_candidate) {
          const auto normal_path = output / "normals" / fs::u8path(scene.images[i].name + ".png");
          auto normal = ReadImage(normal_path, cv::IMREAD_UNCHANGED);
          const auto& grid = grids[model.images[i].camera];
          if (normal.type() != CV_8UC3 || normal.cols != grid.width || normal.rows != grid.height)
            throw std::runtime_error("Unexpected supervision normal map: " + scene.images[i].name);
          for (int y = 0; y < normal.rows; ++y) {
            auto* np = normal.ptr<cv::Vec3b>(y);
            for (int x = 0; x < normal.cols; ++x) {
              if (!np[x][0] && !np[x][1] && !np[x][2])
                continue;
              const size_t k = size_t(y / ConflictLattice) * cw + x / ConflictLattice;
              if (!verified[i].data[k] && conflicts[i][k] >= ConflictMinimumViews) {
                np[x] = cv::Vec3b(0, 0, 0);
                ++dropped_normals;
              }
            }
          }
          if (dropped_normals) {
            WriteImage(normal_path, normal);
            row["normal_size"] = fs::file_size(normal_path);
          }
        }
        row["depth_pixels"] = row.at("depth_pixels").get<uint64_t>() - dropped;
        row["normal_pixels"] = row.at("normal_pixels").get<uint64_t>() - dropped_normals;
        row["monocular"]["view_conflict_pixels"] = dropped;
        row["monocular"]["view_conflict_normal_pixels"] = dropped_normals;
        dropped_total += dropped;
        dropped_normal_total += dropped_normals;
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(failure_lock);
      if (!failure)
        failure = std::current_exception();
      next = uint32_t(n);
    }
  };
  {
    std::vector<std::thread> threads;
    for (size_t k = 0; k < workers; ++k)
      threads.emplace_back(rewrite);
    for (auto& thread : threads)
      thread.join();
  }
  if (failure)
    std::rethrow_exception(failure);
  return {{"neighbours", ConflictNeighbours},
          {"shared_tracks", ConflictSharedTracks},
          {"baseline_m", ConflictBaselineMetres},
          {"lattice_pixels", ConflictLattice},
          {"in_front_log", ConflictLog},
          {"drop", "cell not LAS-verified with at least 2 contradicting views drops both depth and normal"},
          {"neighbourhood", "3 x 3 nearest depth of the other view, all valid"},
          {"pairs", total_pairs},
          {"tested_samples", tested_total},
          {"conflicting_samples", conflicting_total},
          {"dropped_depth_pixels", dropped_total.load()},
          {"dropped_normal_pixels", dropped_normal_total.load()},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
}
}  // namespace geosun

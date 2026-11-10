#include "geometry/monocular.h"
#include "geometry/supervision.h"
#include "io/scene.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

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
// Fewest grid pixels two faces must share to constrain their relative scale.
constexpr size_t MinimumOverlapPixels = 500;
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
bool SolveShift(const float* points, const cv::Mat& valid, int size, double& shift) {
  struct Sample {
    double x, y, z, u, v;
  };
  std::vector<Sample> samples;
  double nearest = INFINITY;
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 64; ++j) {
      const int row = i * size / 64, column = j * size / 64;
      if (!valid.at<uint8_t>(row, column))
        continue;
      const float* p = points + 3 * (size_t(row) * size + column);
      samples.push_back({p[0], p[1], p[2], (2. * column + 1 - size) / size,
                         (2. * row + 1 - size) / size});
      nearest = std::min(nearest, double(p[2]));
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
cv::Mat Prefiltered(const MonocularLens& lens, const cv::Mat& rgb) {
  if (rgb.type() != CV_8UC3 || rgb.cols != lens.source.width || rgb.rows != lens.source.height)
    throw std::runtime_error("Photo size differs from the lens calibration");
  if (lens.prefilter >= 1)
    return rgb;
  cv::Mat small;
  cv::resize(rgb, small,
             cv::Size(int(std::lround(rgb.cols * lens.prefilter)),
                      int(std::lround(rgb.rows * lens.prefilter))),
             0, 0, cv::INTER_AREA);
  return small;
}
}  // namespace
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
  const double sx = std::lround(source.width * lens.prefilter) / double(source.width),
               sy = std::lround(source.height * lens.prefilter) / double(source.height);
  const double source_radius = source.Radius(), limit = std::cos(Radians(FieldOfViewDegrees / 2));
  for (const auto& rotation : rotations) {
    MonocularFace face{rotation, cv::Mat(size, size, CV_32FC2), cv::Mat::zeros(size, size, CV_8UC1)};
    const Eigen::Matrix3d to_camera = rotation.transpose();
    for (int v = 0; v < size; ++v)
      for (int u = 0; u < size; ++u) {
        const Eigen::Vector3d ray =
            (to_camera * Eigen::Vector3d((u + .5 - size / 2.) / lens.focal,
                                         (v + .5 - size / 2.) / lens.focal, 1))
                .normalized();
        auto& target = face.source.at<cv::Vec2f>(v, u);
        target = cv::Vec2f(-10, -10);
        if (ray.z() < limit)
          continue;
        const auto pixel = source.Project(ray);
        const double dx = (pixel.x() - source.params[2]) / source.params[0],
                     dy = (pixel.y() - source.params[3]) / source.params[1];
        if (pixel.x() < 0 || pixel.y() < 0 || pixel.x() >= source.width ||
            pixel.y() >= source.height || dx * dx + dy * dy > source_radius * source_radius)
          continue;
        target = cv::Vec2f(float(pixel.x() * sx - .5), float(pixel.y() * sy - .5));
        face.valid.at<uint8_t>(v, u) = 255;
      }
    lens.faces.push_back(std::move(face));
  }
  lens.samples.assign(rays.size(), {});
  lens.counts.assign(rays.size(), 0);
  lens.rays = cv::Mat::zeros(h, w, CV_32FC3);
  const double half_width = std::tan(Radians(half_degrees));
  for (size_t i = 0; i < rays.size(); ++i) {
    if (rays[i].isZero())
      continue;
    lens.rays.at<cv::Vec3f>(int(i / w), int(i % w)) =
        cv::Vec3f(float(rays[i].x()), float(rays[i].y()), float(rays[i].z()));
    std::vector<FaceSample> found;
    for (size_t f = 0; f < rotations.size(); ++f) {
      const Eigen::Vector3d q = rotations[f] * rays[i];
      if (q.z() <= 0)
        continue;
      const double xn = q.x() / q.z(), yn = q.y() / q.z();
      const double weight = std::pow(
          std::clamp((1 - std::max(std::abs(xn), std::abs(yn)) / half_width) / FeatherBand, 0., 1.),
          2);
      const double x = lens.focal * xn + size / 2. - .5, y = lens.focal * yn + size / 2. - .5;
      if (weight > 0 && x >= 0 && y >= 0 && x < size - 1 && y < size - 1)
        found.push_back({float(x), float(y), float(weight), uint8_t(f)});
    }
    std::sort(found.begin(), found.end(),
              [](const FaceSample& a, const FaceSample& b) { return a.weight > b.weight; });
    const size_t count = std::min<size_t>(found.size(), MonocularLens::MaxSamples);
    std::copy(found.begin(), found.begin() + count, lens.samples[i].begin());
    lens.counts[i] = uint8_t(count);
  }
  return lens;
}
MonocularEstimator::MonocularEstimator(Models& models)
    : models_(models), pixels_(models.Stream()) {
  const auto& manifest = models.Manifest();
  if (!manifest.at("models").contains("geometry"))
    throw std::runtime_error(
        "Model package has no geometry engine; prepare a version 3 package with "
        "prepare-models.ps1");
  size_ = manifest.at("geometry_size");
  // Load the engine now, so device memory budgets measured afterwards see it.
  pixels_.Upload(cv::Mat::zeros(size_, size_, CV_8UC3));
  models_.Predict("geometry", {pixels_.Normalize(size_, size_)});
}
std::vector<cv::Mat> MonocularEstimator::Faces(const MonocularLens& lens, const cv::Mat& rgb) {
  const auto source = Prefiltered(lens, rgb);
  std::vector<cv::Mat> faces;
  for (const auto& face : lens.faces) {
    cv::Mat sampled;
    cv::remap(source, sampled, face.source, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
              cv::Scalar::all(0));
    faces.push_back(std::move(sampled));
  }
  return faces;
}
std::vector<FaceEstimate> MonocularEstimator::Estimate(const MonocularLens& lens,
                                                       const std::vector<cv::Mat>& images) {
  if (lens.size != size_ || images.size() != lens.faces.size())
    throw std::runtime_error("Monocular faces differ from the geometry engine or lens plan");
  const int size = size_;
  // The CPU half of a face (shift solve, distance and normal maps) runs beside the GPU inference
  // of the following faces; a few in flight bound the memory of their host-side outputs.
  auto process = [&lens, size](size_t f, const std::vector<Tensor>& outputs) {
    const std::vector<int64_t> map{1, size, size, 3};
    if (outputs.size() != 4 || outputs[0].shape != map || outputs[1].shape != map ||
        outputs[2].shape != std::vector<int64_t>{1, size, size} ||
        outputs[3].shape != std::vector<int64_t>{1})
      throw std::runtime_error("Geometry engine output contract differs");
    const float *points = outputs[0].Floats(), *normals = outputs[1].Floats(),
                *mask = outputs[2].Floats();
    const auto& face = lens.faces[f];
    cv::Mat valid(size, size, CV_8UC1);
    for (int i = 0; i < size * size; ++i)
      valid.data[i] = face.valid.data[i] && mask[i] > .5f && std::isfinite(points[3 * i + 2]) ? 255 : 0;
    FaceEstimate estimate;
    estimate.distance = cv::Mat::zeros(size, size, CV_32FC1);
    estimate.normal = cv::Mat::zeros(size, size, CV_32FC3);
    estimate.metric_scale = outputs[3].Floats()[0];
    if (!std::isfinite(estimate.metric_scale) || estimate.metric_scale <= 0)
      throw std::runtime_error("Geometry engine returned an invalid metric scale");
    if (SolveShift(points, valid, size, estimate.shift)) {
      const Eigen::Matrix3d to_camera = face.rotation.transpose();
      for (int v = 0; v < size; ++v)
        for (int u = 0; u < size; ++u) {
          const size_t i = size_t(v) * size + u;
          const double depth = points[3 * i + 2] + estimate.shift;
          if (!valid.data[i] || !(depth > 0))
            continue;
          const Eigen::Vector3d ray((u + .5 - size / 2.) / lens.focal,
                                    (v + .5 - size / 2.) / lens.focal, 1);
          Eigen::Vector3d n =
              to_camera * Eigen::Vector3d(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
          if (n.dot(to_camera * ray) > 0)
            n = -n;
          estimate.distance.at<float>(v, u) = float(depth * ray.norm() * estimate.metric_scale);
          estimate.normal.at<cv::Vec3f>(v, u) = cv::Vec3f(float(n.x()), float(n.y()), float(n.z()));
          ++estimate.valid;
        }
    }
    return estimate;
  };
  constexpr size_t InFlight = 4;
  std::vector<FaceEstimate> estimates;
  std::deque<std::future<FaceEstimate>> pending;
  for (size_t f = 0; f < images.size(); ++f) {
    pixels_.Upload(images[f]);
    auto outputs = models_.Predict("geometry", {pixels_.Normalize(size, size)});
    pending.push_back(std::async(std::launch::async, [process, f, outputs = std::move(outputs)] {
      return process(f, outputs);
    }));
    if (pending.size() >= InFlight) {
      estimates.push_back(pending.front().get());
      pending.pop_front();
    }
  }
  while (!pending.empty()) {
    estimates.push_back(pending.front().get());
    pending.pop_front();
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
FaceAlignment AlignFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces) {
  if (faces.size() != lens.faces.size())
    throw std::runtime_error("Monocular faces differ from the lens plan");
  const size_t count = faces.size();
  std::vector<std::vector<double>> differences(count * count);
  for (size_t i = 0; i < lens.counts.size(); ++i) {
    double distance[MonocularLens::MaxSamples];
    bool valid[MonocularLens::MaxSamples];
    Eigen::Vector3d normal;
    for (int k = 0; k < lens.counts[i]; ++k)
      valid[k] = SampleFace(faces[lens.samples[i][k].face], lens.samples[i][k], distance[k], normal);
    for (int a = 0; a < lens.counts[i]; ++a)
      for (int b = 0; b < lens.counts[i]; ++b) {
        const size_t fa = lens.samples[i][a].face, fb = lens.samples[i][b].face;
        if (valid[a] && valid[b] && fa < fb)
          differences[fa * count + fb].push_back(std::log(distance[b] / distance[a]));
      }
  }
  // s_a - s_b = median(log d_b - log d_a) per overlapping pair; face 0 keeps log scale 0.
  FaceAlignment alignment{std::vector<double>(count, 1.), Json::array()};
  Eigen::MatrixXd normal = Eigen::MatrixXd::Zero(count, count);
  Eigen::VectorXd right = Eigen::VectorXd::Zero(count);
  std::vector<std::array<double, 3>> pairs(count * count);
  for (size_t a = 0; a < count; ++a)
    for (size_t b = a + 1; b < count; ++b) {
      auto& values = differences[a * count + b];
      if (values.size() < MinimumOverlapPixels)
        continue;
      const double median = Quantile(values, .5);
      for (auto& value : values)
        value = std::abs(value - median);
      const double mad = Quantile(values, .5), weight = std::sqrt(double(values.size()));
      pairs[a * count + b] = {median, mad, double(values.size())};
      normal(a, a) += weight;
      normal(b, b) += weight;
      normal(a, b) -= weight;
      normal(b, a) -= weight;
      right[a] += weight * median;
      right[b] -= weight * median;
    }
  normal(0, 0) += 1e6;
  const Eigen::VectorXd logs = normal.ldlt().solve(right);
  for (size_t f = 0; f < count; ++f) {
    if (f && normal(f, f) == 0)
      throw std::runtime_error("Monocular face " + std::to_string(f) + " overlaps no other face");
    alignment.scales[f] = std::exp(logs[f]);
  }
  for (size_t a = 0; a < count; ++a)
    for (size_t b = a + 1; b < count; ++b)
      if (const auto& p = pairs[a * count + b]; p[2] > 0)
        alignment.pairs.push_back({{"faces", {a, b}},
                                   {"pixels", size_t(p[2])},
                                   {"log_ratio_median", p[0]},
                                   {"log_ratio_mad", p[1]},
                                   {"aligned_residual", p[0] - logs[a] + logs[b]}});
  return alignment;
}
MonocularMaps GatherFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                          const std::function<double(const FaceSample&)>& log_scale) {
  if (faces.size() != lens.faces.size())
    throw std::runtime_error("Monocular faces differ from the lens plan");
  const int w = lens.grid.width, h = lens.grid.height;
  MonocularMaps maps{cv::Mat::zeros(h, w, CV_32FC1), cv::Mat::zeros(h, w, CV_32FC3)};
  for (size_t i = 0; i < lens.counts.size(); ++i) {
    double weights = 0, log_distance = 0;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (int k = 0; k < lens.counts[i]; ++k) {
      const auto& sample = lens.samples[i][k];
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
    maps.distance.at<float>(int(i / w), int(i % w)) = float(std::exp(log_distance / weights));
    normal.normalize();
    maps.normal.at<cv::Vec3f>(int(i / w), int(i % w)) =
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
                      const cv::Mat& las, const cv::Mat& las_normal) {
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
  const int w = lens.grid.width;
  for (int y = 0; y < lens.grid.height; y += FitLattice)
    for (int x = 0; x < w; x += FitLattice) {
      const size_t i = size_t(y) * w + x;
      double distance[MonocularLens::MaxSamples];
      bool valid[MonocularLens::MaxSamples];
      Eigen::Vector3d normal[MonocularLens::MaxSamples];
      for (int k = 0; k < lens.counts[i]; ++k)
        valid[k] =
            SampleFace(faces[lens.samples[i][k].face], lens.samples[i][k], distance[k], normal[k]);
      const double target = las.at<uint16_t>(y, x) / 1000.;
      for (int a = 0; a < lens.counts[i]; ++a) {
        if (!valid[a])
          continue;
        if (target > 0) {
          ++fit.las_samples;
          if (Trusted(las_normal, x, y, normal[a]))
            anchors.push_back({lens.samples[i][a], float(std::log(target / distance[a]))});
        }
        for (int b = a + 1; b < lens.counts[i]; ++b)
          if (valid[b])
            ties.push_back({lens.samples[i][a], lens.samples[i][b],
                            float(std::log(distance[b] / distance[a]))});
      }
    }
  const size_t count = faces.size();
  constexpr int n = FaceFit::Cells + 1;
  fit.scales.assign(count, 1.);
  fit.fields.assign(count, std::vector<double>(n * n, 0.));
  fit.anchors = anchors.size();
  fit.faces = Json::array();
  if (anchors.size() < MinimumImageAnchors)
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
Json EncodeMonocular(const MonocularLens& lens, const MonocularMaps& maps,
                     const cv::Mat& detail_normal, const cv::Mat& las, const cv::Mat& las_normal,
                     const cv::Mat& allowed, bool depth, double min_incidence, cv::Mat& depth_mm,
                     cv::Mat& normal_rgb, cv::Mat& verified) {
  const int w = lens.grid.width, h = lens.grid.height;
  if (las.size() != cv::Size(w, h) || allowed.size() != las.size() || allowed.type() != CV_8UC1 ||
      maps.distance.size() != las.size() || las_normal.size() != las.size() ||
      las_normal.type() != CV_8UC3 || detail_normal.size() != las.size())
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
      const auto& detail = detail_normal.at<cv::Vec3f>(y, x);
      const auto& ray = lens.rays.at<cv::Vec3f>(y, x);
      if ((detail[0] || detail[1] || detail[2]) &&
          -(detail[0] * ray[0] + detail[1] * ray[1] + detail[2] * ray[2]) >= min_incidence)
        normal_rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(uint8_t(127.5f + 127.5f * detail[2] + .5f),
                                                   uint8_t(127.5f + 127.5f * detail[1] + .5f),
                                                   uint8_t(127.5f + 127.5f * detail[0] + .5f));
      // A depth between both sides of a step is a point in mid air, whereas thin structures keep
      // their normals, so only depth drops the steps.
      const float d = maps.distance.at<float>(y, x);
      if (d <= 0 || blended.at<uint8_t>(y, x))
        continue;
      // Only trusted LAS depth vetoes MoGe; elsewhere the disagreement is only counted.
      const auto& n = maps.normal.at<cv::Vec3f>(y, x);
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
Json CheckMonocular(const fs::path& input, const fs::path& root, const std::string& image,
                    const fs::path& output, int max_size) {
  if (fs::exists(output))
    throw std::runtime_error("Monocular check output already exists; choose a new directory");
  const auto scene = ReadScene(input);
  const auto exposure = std::find_if(scene.images.begin(), scene.images.end(),
                                     [&](const Exposure& e) { return e.name == image; });
  if (exposure == scene.images.end())
    throw std::runtime_error("Photo is not part of the scene: " + image);
  const auto& camera = scene.cameras[exposure->camera];
  Models models(root);
  MonocularEstimator estimator(models);
  auto started = Clock::now();
  const auto lens = PlanMonocularLens(camera, MapCamera(camera, max_size), estimator.FaceSize(),
                                      DepthFaceHalfDegrees);
  const auto normal_lens = PlanMonocularLens(camera, MapCamera(camera, max_size),
                                             estimator.FaceSize(), NormalFaceHalfDegrees);
  const double plan_seconds = Seconds(started);
  auto bgr = ReadImage(exposure->source, cv::IMREAD_COLOR);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  started = Clock::now();
  const auto estimates = estimator.Estimate(lens, MonocularEstimator::Faces(lens, rgb));
  const auto normal_estimates =
      estimator.Estimate(normal_lens, MonocularEstimator::Faces(normal_lens, rgb));
  const double estimate_seconds = Seconds(started);
  started = Clock::now();
  const auto alignment = AlignFaces(lens, estimates);
  const auto maps = GatherFaces(lens, estimates, [&](const FaceSample& sample) {
    return std::log(alignment.scales[sample.face]);
  });
  const auto normal_maps =
      GatherFaces(normal_lens, normal_estimates, [](const FaceSample&) { return 0.; });
  const double gather_seconds = Seconds(started);

  fs::create_directories(output / "faces");
  const auto faces = MonocularEstimator::Faces(lens, rgb);
  Json face_reports = Json::array();
  for (size_t f = 0; f < faces.size(); ++f) {
    cv::Mat face;
    cv::cvtColor(faces[f], face, cv::COLOR_RGB2BGR);
    WriteImage(output / "faces" / (std::to_string(f) + ".jpg"), face);
    face_reports.push_back(estimates[f].ToJson());
  }
  const int w = lens.grid.width, h = lens.grid.height;
  cv::Mat depth = cv::Mat::zeros(h, w, CV_16UC1), normal = cv::Mat::zeros(h, w, CV_8UC3),
          preview = cv::Mat::zeros(h, w, CV_8UC1), valid = cv::Mat::zeros(h, w, CV_8UC1);
  std::vector<float> logs;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (const float d = maps.distance.at<float>(y, x); d > 0)
        logs.push_back(std::log(d));
  if (logs.empty())
    throw std::runtime_error("Monocular estimate has no valid pixels");
  std::vector<double> sorted(logs.begin(), logs.end());
  const double low = Quantile(sorted, .01), high = Quantile(sorted, .99);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float d = maps.distance.at<float>(y, x);
      if (d <= 0)
        continue;
      const auto& n = normal_maps.normal.at<cv::Vec3f>(y, x);
      depth.at<uint16_t>(y, x) = uint16_t(std::clamp<long>(std::lround(d * 1000), 1, 65535));
      if (n[0] || n[1] || n[2])
        normal.at<cv::Vec3b>(y, x) =
            cv::Vec3b(uint8_t(127.5f + 127.5f * n[2] + .5f), uint8_t(127.5f + 127.5f * n[1] + .5f),
                      uint8_t(127.5f + 127.5f * n[0] + .5f));
      preview.at<uint8_t>(y, x) = uint8_t(
          std::clamp(255 * (std::log(d) - low) / std::max(high - low, 1e-6), 1., 255.));
      valid.at<uint8_t>(y, x) = 255;
    }
  cv::Mat colour;
  cv::applyColorMap(preview, colour, cv::COLORMAP_TURBO);
  colour.setTo(cv::Scalar::all(0), ~valid);
  WriteImage(output / "depth.png", depth);
  WriteImage(output / "normal.png", normal);
  WriteImage(output / "depth_preview.jpg", colour);
  Json report = {{"image", image},
                 {"camera", camera.label},
                 {"lens", lens.ToJson()},
                 {"normal_lens", normal_lens.ToJson()},
                 {"faces", face_reports},
                 {"overlap_scales", alignment.scales},
                 {"overlaps", alignment.pairs},
                 {"valid_pixels", logs.size()},
                 {"distance_m", {{"p01", std::exp(low)}, {"p50", std::exp(Quantile(sorted, .5))},
                                 {"p99", std::exp(high)}}},
                 {"seconds", {{"plan", plan_seconds},
                              {"estimate", estimate_seconds},
                              {"align_and_gather", gather_seconds}}},
                 {"inference", models.Profile()}};
  WriteJson(output / "monocular.json", report);
  return report;
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
Json FilterViewConflicts(const fs::path& output, const Scene& scene, const Reconstruction& model,
                         const ShutterMotion* motion, int max_size,
                         const std::vector<cv::Mat>& verified, std::vector<Json>& rows) {
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
  std::array<std::vector<Eigen::Vector3d>, 2> rays;
  for (int c = 0; c < 2; ++c) {
    const auto& grid = grids[c];
    const double radius = grid.Radius();
    rays[c].assign(size_t(grid.width) * grid.height, Eigen::Vector3d::Zero());
    for (int y = 0; y < grid.height; y += ConflictLattice)
      for (int x = 0; x < grid.width; x += ConflictLattice) {
        const double dx = (x + .5 - grid.params[2]) / grid.params[0],
                     dy = (y + .5 - grid.params[3]) / grid.params[1];
        if (dx * dx + dy * dy <= radius * radius)
          rays[c][size_t(y) * grid.width + x] = grid.Ray({x + .5, y + .5});
      }
  }
  auto cells_of = [&](uint32_t i) {
    const auto& grid = grids[model.images[i].camera];
    return std::make_pair((grid.width + ConflictLattice - 1) / ConflictLattice,
                          (grid.height + ConflictLattice - 1) / ConflictLattice);
  };
  // Per cell, the number of views contradicting it.
  std::vector<std::vector<uint8_t>> conflicts(n);
  std::vector<std::mutex> locks(n);
  for (uint32_t i = 0; i < n; ++i)
    if (model.images[i].registered) {
      const auto [cw, ch] = cells_of(i);
      conflicts[i].assign(size_t(cw) * ch, 0);
      if (verified.at(i).type() != CV_8UC1 || verified[i].cols != cw || verified[i].rows != ch)
        throw std::runtime_error("LAS-verified cells differ from the conflict lattice");
    }
  auto read_depth = [&](uint32_t i) {
    const auto depth = ReadImage(output / "depths" / fs::u8path(scene.images[i].name + ".png"),
                                 cv::IMREAD_UNCHANGED);
    const auto& grid = grids[model.images[i].camera];
    if (depth.type() != CV_16UC1 || depth.cols != grid.width || depth.rows != grid.height)
      throw std::runtime_error("Unexpected supervision depth map: " + scene.images[i].name);
    return depth;
  };
  const double view_limit = std::cos(ConflictFieldOfViewDegrees * Pi / 360),
               tolerance = std::exp(ConflictLog);
  std::atomic<uint64_t> tested{0}, conflicting{0};
  // Flags the cells of a, and the cells of b its points land on, where a's depth lies in front of
  // b's.
  auto check = [&](uint32_t a, uint32_t b, const cv::Mat& da, const cv::Mat& db,
                   std::vector<uint8_t>& own, std::vector<uint8_t>& other) {
    const uint32_t ca = model.images[a].camera, cb = model.images[b].camera;
    const auto& ga = grids[ca];
    const auto& gb = grids[cb];
    const int cells_a = (ga.width + ConflictLattice - 1) / ConflictLattice,
              cells_b = (gb.width + ConflictLattice - 1) / ConflictLattice;
    uint64_t local_tested = 0, local_conflicting = 0;
    for (int y = 0; y < ga.height; y += ConflictLattice)
      for (int x = 0; x < ga.width; x += ConflictLattice) {
        const auto millimetres = da.at<uint16_t>(y, x);
        const Eigen::Vector3d& ray = rays[ca][size_t(y) * ga.width + x];
        if (!millimetres || ray.isZero())
          continue;
        const Eigen::Vector3d seen = millimetres / 1000. * ray;
        const Eigen::Vector3d reference =
            motion ? motion->Reference(a, ca, motion->Fraction(ga, {x + .5, y + .5}), seen) : seen;
        const Eigen::Vector3d world = rotations[a].transpose() * (reference - translations[a]);
        const Eigen::Vector3d there = rotations[b] * world + translations[b];
        Eigen::Vector3d moved = there;
        Eigen::Vector2d pixel;
        double s = 0;
        bool visible = true;
        for (int k = 0; k < (motion ? 4 : 1); ++k) {
          if (motion)
            moved = motion->Apply(b, cb, s, there);
          if (moved.z() < view_limit * moved.norm()) {
            visible = false;
            break;
          }
          pixel = gb.Project(moved);
          if (motion)
            s = motion->Fraction(gb, pixel);
        }
        if (!visible)
          continue;
        const int u = int(std::floor(pixel.x())), v = int(std::floor(pixel.y()));
        if (u < 1 || v < 1 || u >= gb.width - 1 || v >= gb.height - 1)
          continue;
        // The nearest depth of b's 3 x 3 neighbourhood, all of which must hold depth: a point near
        // a depth edge of b or off its valid depth is no evidence.
        uint16_t nearest = 65535;
        for (int k = 0; k < 9 && nearest; ++k) {
          const auto value = db.at<uint16_t>(v + k / 3 - 1, u + k % 3 - 1);
          nearest = value ? std::min(nearest, value) : uint16_t(0);
        }
        if (!nearest)
          continue;
        ++local_tested;
        const size_t cell_a = size_t(y / ConflictLattice) * cells_a + x / ConflictLattice,
                     cell_b = size_t(v / ConflictLattice) * cells_b + u / ConflictLattice;
        const double metres = moved.norm();
        if (metres * 1000 * tolerance < nearest) {
          ++local_conflicting;
          own[cell_a] = other[cell_b] = 1;
        }
      }
    tested += local_tested;
    conflicting += local_conflicting;
  };
  auto merge = [&](uint32_t i, const std::vector<uint8_t>& local) {
    std::lock_guard<std::mutex> guard(locks[i]);
    for (size_t k = 0; k < local.size(); ++k)
      if (local[k] && conflicts[i][k] < 255)
        ++conflicts[i][k];
  };
  std::atomic<uint32_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_lock;
  auto work = [&] {
    try {
      for (uint32_t i; (i = next++) < n;) {
        if (!model.images[i].registered)
          continue;
        const auto di = read_depth(i);
        std::vector<uint8_t> own(conflicts[i].size());
        for (uint32_t j = i + 1; j < n; ++j) {
          if (!paired[i][j])
            continue;
          const auto dj = read_depth(j);
          std::vector<uint8_t> other(conflicts[j].size());
          check(i, j, di, dj, own, other);
          check(j, i, dj, di, other, own);
          merge(j, other);
        }
        merge(i, own);
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(failure_lock);
      if (!failure)
        failure = std::current_exception();
      next = uint32_t(n);
    }
  };
  const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
  {
    std::vector<std::thread> threads;
    for (unsigned k = 0; k < workers; ++k)
      threads.emplace_back(work);
    for (auto& thread : threads)
      thread.join();
  }
  if (failure)
    std::rethrow_exception(failure);
  // A cell loses its depth when it is not LAS-verified and at least ConflictMinimumViews views
  // contradict it. Agreeing views do not outvote them: in a LAS hole two groups of views can each
  // agree among themselves, and nothing tells which group is right.
  next = 0;
  std::atomic<uint64_t> dropped_total{0};
  auto rewrite = [&] {
    try {
      for (uint32_t i; (i = next++) < n;) {
        if (!model.images[i].registered)
          continue;
        auto depth = read_depth(i);
        const auto [cw, ch] = cells_of(i);
        uint64_t dropped = 0;
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
        auto& row = rows.at(i);
        if (dropped) {
          const auto path = output / "depths" / fs::u8path(scene.images[i].name + ".png");
          WriteImage(path, depth);
          row["depth_size"] = fs::file_size(path);
        }
        row["depth_pixels"] = row.at("depth_pixels").get<uint64_t>() - dropped;
        row["monocular"]["view_conflict_pixels"] = dropped;
        dropped_total += dropped;
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
    for (unsigned k = 0; k < workers; ++k)
      threads.emplace_back(rewrite);
    for (auto& thread : threads)
      thread.join();
  }
  if (failure)
    std::rethrow_exception(failure);
  size_t pairs = 0;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = i + 1; j < n; ++j)
      pairs += paired[i][j];
  return {{"neighbours", ConflictNeighbours},
          {"shared_tracks", ConflictSharedTracks},
          {"baseline_m", ConflictBaselineMetres},
          {"lattice_pixels", ConflictLattice},
          {"in_front_log", ConflictLog},
          {"drop", "cell not LAS-verified with at least 2 contradicting views"},
          {"neighbourhood", "3 x 3 nearest depth of the other view, all valid"},
          {"pairs", pairs},
          {"tested_samples", tested.load()},
          {"conflicting_samples", conflicting.load()},
          {"dropped_depth_pixels", dropped_total.load()},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
}
}  // namespace geosun

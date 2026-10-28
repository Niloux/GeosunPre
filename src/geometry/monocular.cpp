#include "geometry/monocular.h"
#include "geometry/supervision.h"
#include "io/scene.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>

namespace geosun {
namespace {
// Half field of view of a face and the margin every planned ray keeps to the face edge.
constexpr double FaceHalfDegrees = 45, FaceMarginDegrees = 5;
// Grid weights fall to zero over this outer band of the face, in normalized image coordinates.
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
constexpr double MinimumDepth = .05, MaximumDepth = 65.535;

double Radians(double degrees) { return degrees * Pi / 180; }
std::vector<Eigen::Matrix3d> FaceRotations(int ring, double tilt) {
  std::vector<Eigen::Matrix3d> rotations{Eigen::Matrix3d::Identity()};
  for (int k = 0; k < ring; ++k) {
    const double azimuth = 2 * Pi * k / ring;
    // Rotates the optical axis towards azimuth; a face's rows stay close to the lens rows.
    const Eigen::Matrix3d face_to_camera =
        Eigen::AngleAxisd(tilt, Eigen::Vector3d(-std::sin(azimuth), std::cos(azimuth), 0))
            .toRotationMatrix();
    rotations.push_back(face_to_camera.transpose());
  }
  return rotations;
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
          {"face_field_of_view_degrees", 2 * FaceHalfDegrees},
          {"edge_margin_degrees", FaceMarginDegrees},
          {"ring_tilt_degrees", tilt_degrees},
          {"face_axes", axes},
          {"prefilter_scale", prefilter},
          {"grid", {grid.width, grid.height}},
          {"covered_pixels", covered}};
}
Json FaceEstimate::ToJson() const {
  return {{"valid_pixels", valid}, {"shift", shift}, {"metric_scale", metric_scale}};
}
MonocularLens PlanMonocularLens(const Camera& source, const Camera& grid, int size) {
  if (size < 64)
    throw std::runtime_error("Monocular face size must be at least 64 pixels");
  MonocularLens lens;
  lens.source = source;
  lens.grid = grid;
  lens.size = size;
  lens.focal = size / (2 * std::tan(Radians(FaceHalfDegrees)));
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
  const double edge = std::tan(Radians(FaceHalfDegrees - FaceMarginDegrees));
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
    // The ring reaches the lens edge while overlapping the front face.
    const double tilt = std::clamp(widest - Radians(30), Radians(30), Radians(60));
    lens.tilt_degrees = tilt * 180 / Pi;
    for (int ring = 3;; ++ring) {
      if (ring > 12)
        throw std::runtime_error("Monocular faces cannot cover the lens field of view");
      rotations = FaceRotations(ring, tilt);
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
          std::clamp((1 - std::max(std::abs(xn), std::abs(yn))) / FeatherBand, 0., 1.), 2);
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
  std::vector<FaceEstimate> estimates;
  for (size_t f = 0; f < images.size(); ++f) {
    pixels_.Upload(images[f]);
    const auto outputs = models_.Predict("geometry", {pixels_.Normalize(size, size)});
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
    estimates.push_back(std::move(estimate));
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
  return {{"accepted", accepted}, {"anchors", anchors}, {"inliers", inliers},
          {"field_cells", Cells}, {"scales", scales}, {"faces", faces}};
}
FaceFit FitFaceScales(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                      const cv::Mat& las) {
  if (faces.size() != lens.faces.size() || las.type() != CV_16UC1 ||
      las.cols != lens.grid.width || las.rows != lens.grid.height)
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
  const int w = lens.grid.width;
  for (int y = 0; y < lens.grid.height; y += FitLattice)
    for (int x = 0; x < w; x += FitLattice) {
      const size_t i = size_t(y) * w + x;
      double distance[MonocularLens::MaxSamples];
      bool valid[MonocularLens::MaxSamples];
      Eigen::Vector3d normal;
      for (int k = 0; k < lens.counts[i]; ++k)
        valid[k] = SampleFace(faces[lens.samples[i][k].face], lens.samples[i][k], distance[k], normal);
      const double target = las.at<uint16_t>(y, x) / 1000.;
      for (int a = 0; a < lens.counts[i]; ++a) {
        if (!valid[a])
          continue;
        if (target > 0)
          anchors.push_back({lens.samples[i][a], float(std::log(target / distance[a]))});
        for (int b = a + 1; b < lens.counts[i]; ++b)
          if (valid[b])
            ties.push_back({lens.samples[i][a], lens.samples[i][b],
                            float(std::log(distance[b] / distance[a]))});
      }
    }
  const size_t count = faces.size();
  constexpr int n = FaceFit::Cells + 1;
  FaceFit fit;
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
Json EncodeMonocular(const MonocularLens& lens, const MonocularMaps& maps, const cv::Mat& las,
                     const cv::Mat& allowed, bool depth, double min_incidence, cv::Mat& depth_mm,
                     cv::Mat& normal_rgb) {
  const int w = lens.grid.width, h = lens.grid.height;
  if (las.size() != cv::Size(w, h) || allowed.size() != las.size() || allowed.type() != CV_8UC1 ||
      maps.distance.size() != las.size())
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
  std::vector<double> agreement;
  size_t disagreeing = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float d = maps.distance.at<float>(y, x);
      if (d <= 0 || !allowed.at<uint8_t>(y, x) || blended.at<uint8_t>(y, x))
        continue;
      const double target = las.at<uint16_t>(y, x) / 1000.;
      bool agrees = true;
      if (target > 0) {
        const double ratio = std::abs(std::log(d / target));
        agreement.push_back(ratio);
        agrees = ratio <= std::log1p(std::max(AgreementFraction, AgreementMetres / target));
        disagreeing += !agrees;
      }
      if (depth && agrees && d >= MinimumDepth && d <= MaximumDepth)
        depth_mm.at<uint16_t>(y, x) = uint16_t(std::clamp<long>(std::lround(d * 1000.), 1, 65535));
      const auto& n = maps.normal.at<cv::Vec3f>(y, x);
      const auto& ray = lens.rays.at<cv::Vec3f>(y, x);
      if (-(n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2]) >= min_incidence)
        normal_rgb.at<cv::Vec3b>(y, x) =
            cv::Vec3b(uint8_t(127.5f + 127.5f * n[2] + .5f), uint8_t(127.5f + 127.5f * n[1] + .5f),
                      uint8_t(127.5f + 127.5f * n[0] + .5f));
    }
  Json stats = {{"las_compared_pixels", agreement.size()}, {"las_disagreeing_pixels", disagreeing}};
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
  const auto lens = PlanMonocularLens(camera, MapCamera(camera, max_size), estimator.FaceSize());
  const double plan_seconds = Seconds(started);
  auto bgr = ReadImage(exposure->source, cv::IMREAD_COLOR);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  started = Clock::now();
  const auto estimates = estimator.Estimate(lens, MonocularEstimator::Faces(lens, rgb));
  const double estimate_seconds = Seconds(started);
  started = Clock::now();
  const auto alignment = AlignFaces(lens, estimates);
  const auto maps = GatherFaces(lens, estimates, [&](const FaceSample& sample) {
    return std::log(alignment.scales[sample.face]);
  });
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
      const auto& n = maps.normal.at<cv::Vec3f>(y, x);
      depth.at<uint16_t>(y, x) = uint16_t(std::clamp<long>(std::lround(d * 1000), 1, 65535));
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
}  // namespace geosun

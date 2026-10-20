#include "sfm/shutter.h"
#include "frontend/frontend.h"
#include "geometry/supervision.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <deque>
#include <fstream>
#include <set>

namespace geosun {
namespace {
// Rectification maps are solved on nodes this many pixels apart and interpolated between them;
// the LAS range grid has one cell per node spacing.
constexpr int GridStep = 8;
constexpr double MinimumRange = .05, MaximumRange = 65.535;
constexpr double RotationBound = .05, ReadoutBound = .05;
constexpr size_t MinimumImageHoldout = 16;
constexpr double MinimumExplainedRotation = .1;
Eigen::Matrix3d Readout(const ShutterMotion& motion, uint32_t image, double s) {
  const Eigen::Vector3d w = s * motion.rotation[image];
  Eigen::Matrix3d a;
  a << 1, -w.z(), w.y(), w.z(), 1, -w.x(), -w.y(), w.x(), 1;
  return a;
}
// Camera point of an observation under the rolling-shutter model.
Eigen::Vector3d Exposed(const Reconstruction& model, const ShutterMotion& motion,
                        const Observation& o, const Eigen::Vector3d& point) {
  const auto& image = model.images[o.image];
  const double s = motion.Fraction(model.cameras[image.camera], image.pixels[o.feature]);
  return motion.Apply(o.image, image.camera, s, image.pose.Apply(point));
}
double ShutterError(const Reconstruction& model, const ShutterMotion& motion, const Observation& o,
                    const Eigen::Vector3d& point) {
  const auto v = Exposed(model, motion, o, point);
  if (v.z() <= 0)
    return INFINITY;
  const auto& image = model.images[o.image];
  return (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm();
}
double ShutterPointCost(const Reconstruction& model, const ShutterMotion& motion,
                        const std::vector<Observation>& track, const Eigen::Vector3d& point) {
  double cost = 0;
  for (const auto& o : track) {
    const double error = ShutterError(model, motion, o, point);
    if (!std::isfinite(error))
      return INFINITY;
    cost += error <= 2 ? .5 * error * error : 2 * (error - 1);
  }
  return cost;
}
// RefinePoint with the cameras and their readout motion fixed.
void RefineShutterPoint(const Reconstruction& model, const ShutterMotion& motion,
                        const std::vector<Observation>& track, Eigen::Vector3d& point) {
  double cost = ShutterPointCost(model, motion, track, point);
  for (int iteration = 0; iteration < 6; ++iteration) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (const auto& o : track) {
      const auto& image = model.images[o.image];
      const auto& camera = model.cameras[image.camera];
      const auto& pixel = image.pixels[o.feature];
      const double s = motion.Fraction(camera, pixel);
      const Eigen::Vector3d v = motion.Apply(o.image, image.camera, s, image.pose.Apply(point));
      const Eigen::Vector2d residual = camera.Project(v) - pixel;
      const Eigen::Matrix<double, 2, 3> j = camera.ProjectionJacobian(v) *
                                            Readout(motion, o.image, s) *
                                            image.pose.Rotation().toRotationMatrix();
      const double weight = std::min(1., 2 / std::max(residual.norm(), 1e-12));
      normal.noalias() += weight * j.transpose() * j;
      rhs.noalias() -= weight * j.transpose() * residual;
    }
    const Eigen::Vector3d step = normal.ldlt().solve(rhs);
    if (!step.allFinite())
      throw std::runtime_error("Nonfinite rolling-shutter triangulation refinement");
    bool accepted = false;
    for (int backtrack = 0; backtrack < 6; ++backtrack) {
      const Eigen::Vector3d candidate = point + step * std::pow(.5, backtrack);
      const double next = ShutterPointCost(model, motion, track, candidate);
      if (std::isfinite(next) && next < cost) {
        point = candidate;
        const double improvement = cost - next;
        cost = next;
        accepted = true;
        if (improvement < 1e-7 * std::max(1., cost))
          return;
        break;
      }
    }
    if (!accepted)
      break;
  }
}
// Pixel the reference pose records for a ray observed at pixel, at the given range.
Eigen::Vector2d Rectify(const Camera& camera, const ShutterMotion& motion, uint32_t image,
                        uint32_t lens, const Eigen::Vector2d& pixel, double range) {
  const double s = motion.Fraction(camera, pixel);
  return camera.Project(motion.Reference(image, lens, s, range * camera.Ray(pixel)));
}
double ExposureTime(const Exposure& e) { return e.week * 604800. + e.seconds; }
// Readout rotation predicted by the readout time and the rotation rate between neighbouring
// exposures of the same lens. The readout time comes from the translation term, so agreement
// is a physical check independent of the per-image rotations.
Json ReadoutConsistency(const Scene& scene, const Reconstruction& model,
                        const ShutterMotion& motion, bool& consistent) {
  Json lenses = Json::array();
  consistent = true;
  for (uint32_t lens = 0; lens < 2; ++lens) {
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == lens)
        ids.push_back(i);
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
      return ExposureTime(scene.images[a]) < ExposureTime(scene.images[b]);
    });
    double wo = 0, oo = 0, ww = 0;
    std::vector<double> rates, rotations;
    for (size_t k = 1; k + 1 < ids.size(); ++k) {
      const uint32_t a = ids[k - 1], i = ids[k], b = ids[k + 1];
      const double dt = ExposureTime(scene.images[b]) - ExposureTime(scene.images[a]);
      if (dt <= 0 || dt > 3)
        continue;
      const Eigen::AngleAxisd relative(model.images[b].pose.Rotation() *
                                       model.images[a].pose.Rotation().conjugate());
      const Eigen::Vector3d omega = relative.angle() * relative.axis() / dt;
      const auto& w = motion.rotation[i];
      wo += w.dot(omega);
      oo += omega.squaredNorm();
      ww += w.squaredNorm();
      rates.push_back(omega.norm() * 180 / Pi);
      rotations.push_back(w.norm() * 180 / Pi);
    }
    if (ids.empty())
      continue;
    const double readout = motion.readout[lens];
    const double explained =
        ww > 0 ? 1 - (ww - 2 * readout * wo + readout * readout * oo) / ww : 0;
    const bool passed = rates.size() >= 20 && explained >= MinimumExplainedRotation;
    consistent = consistent && passed;
    lenses.push_back({{"camera", lens},
                      {"images", ids.size()},
                      {"rate_samples", rates.size()},
                      {"readout_s", readout},
                      {"rotation_fit_readout_s", oo > 0 ? wo / oo : 0},
                      {"explained_rotation", explained},
                      {"rotation_rate_deg_s", Statistics(rates)},
                      {"readout_rotation_deg", Statistics(rotations)},
                      {"passed", passed}});
  }
  return lenses;
}
struct HoldoutErrors {
  std::vector<double> all;
  std::map<uint32_t, std::vector<double>> images;
  void Add(uint32_t image, double error) {
    all.push_back(error);
    images[image].push_back(error);
  }
};
// Push-pull fill of LAS range holes in inverse range, so missing surfaces take the nearby
// geometry and regions without any record fall back to a distant range.
cv::Mat FillRanges(const cv::Mat& range) {
  std::vector<cv::Mat> value{cv::Mat::zeros(range.size(), CV_32FC1)},
      weight{cv::Mat::zeros(range.size(), CV_32FC1)};
  for (int y = 0; y < range.rows; ++y)
    for (int x = 0; x < range.cols; ++x) {
      const float r = range.at<float>(y, x);
      if (r > 0 && std::isfinite(r)) {
        value[0].at<float>(y, x) = 1.f / r;
        weight[0].at<float>(y, x) = 1;
      }
    }
  while (value.back().cols > 1 || value.back().rows > 1) {
    const auto& v = value.back();
    const auto& w = weight.back();
    cv::Mat nv = cv::Mat::zeros((v.rows + 1) / 2, (v.cols + 1) / 2, CV_32FC1), nw = nv.clone();
    for (int y = 0; y < nv.rows; ++y)
      for (int x = 0; x < nv.cols; ++x) {
        float sum = 0, total = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const int yy = 2 * y + dy, xx = 2 * x + dx;
            if (yy < v.rows && xx < v.cols) {
              sum += w.at<float>(yy, xx) * v.at<float>(yy, xx);
              total += w.at<float>(yy, xx);
            }
          }
        nv.at<float>(y, x) = total > 0 ? sum / total : 0;
        nw.at<float>(y, x) = std::min(1.f, total);
      }
    value.push_back(nv);
    weight.push_back(nw);
  }
  for (size_t level = value.size() - 1; level-- > 0;) {
    cv::Mat up;
    cv::resize(value[level + 1], up, value[level].size(), 0, 0, cv::INTER_LINEAR);
    value[level] = weight[level].mul(value[level]) + (1 - weight[level]).mul(up);
  }
  cv::Mat millimetres(range.size(), CV_16UC1);
  for (int y = 0; y < range.rows; ++y)
    for (int x = 0; x < range.cols; ++x) {
      const float inverse = value[0].at<float>(y, x);
      const double r = inverse > 0 ? std::clamp(1. / inverse, MinimumRange, MaximumRange)
                                   : MaximumRange;
      millimetres.at<uint16_t>(y, x) = uint16_t(std::lround(r * 1000));
    }
  return millimetres;
}
double SampleRange(const cv::Mat& millimetres, double x, double y) {
  x = std::clamp(x, 0., millimetres.cols - 1.);
  y = std::clamp(y, 0., millimetres.rows - 1.);
  const int x0 = std::min(int(x), millimetres.cols - 2 < 0 ? 0 : millimetres.cols - 2),
            y0 = std::min(int(y), millimetres.rows - 2 < 0 ? 0 : millimetres.rows - 2);
  const int x1 = std::min(x0 + 1, millimetres.cols - 1), y1 = std::min(y0 + 1, millimetres.rows - 1);
  const double fx = x - x0, fy = y - y0;
  auto at = [&](int yy, int xx) {
    const uint16_t v = millimetres.at<uint16_t>(yy, xx);
    return v ? v / 1000. : MaximumRange;
  };
  return (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x1)) +
         fy * ((1 - fx) * at(y1, x0) + fx * at(y1, x1));
}
// Offset just past SOI and the leading APPn/COM segments; kept receives APP1-APP15 and COM.
size_t MetadataEnd(const std::vector<uint8_t>& jpeg, std::vector<uint8_t>* kept) {
  if (jpeg.size() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8)
    throw std::runtime_error("Not a JPEG stream");
  size_t at = 2;
  while (at + 4 <= jpeg.size() && jpeg[at] == 0xFF) {
    const uint8_t marker = jpeg[at + 1];
    if (marker == 0xFF) {
      ++at;
      continue;
    }
    if ((marker < 0xE0 || marker > 0xEF) && marker != 0xFE)
      break;
    const size_t length = size_t(jpeg[at + 2]) << 8 | jpeg[at + 3];
    if (length < 2 || at + 2 + length > jpeg.size())
      throw std::runtime_error("Truncated JPEG metadata segment");
    if (kept && marker != 0xE0)
      kept->insert(kept->end(), jpeg.begin() + at, jpeg.begin() + at + 2 + length);
    at += 2 + length;
  }
  return at;
}
}  // namespace
Json MotionToJson(const ShutterMotion& motion, const Reconstruction& model) {
  Json images = Json::array();
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      const auto& w = motion.rotation[i];
      const auto& v = motion.velocity[i];
      images.push_back({{"image_index", i},
                        {"rotation_rad", {w.x(), w.y(), w.z()}},
                        {"velocity_m_s", {v.x(), v.y(), v.z()}}});
    }
  return {{"readout_axis", motion.axis ? "rows" : "columns"},
          {"readout_fraction", "(c - size / 2) / size along the readout axis"},
          {"camera_point", "(I + s [rotation]x) (v - s readout velocity)"},
          {"frames", "rotation and velocity in the reference camera frame"},
          {"readout_s", motion.readout},
          {"images", images}};
}
ShutterMotion MotionFromJson(const Json& value, size_t count) {
  ShutterMotion motion;
  const auto axis = value.at("readout_axis").get<std::string>();
  if (axis != "rows" && axis != "columns")
    throw std::runtime_error("Unknown rolling-shutter readout axis: " + axis);
  motion.axis = axis == "rows";
  motion.rotation.assign(count, Eigen::Vector3d::Zero());
  motion.velocity.assign(count, Eigen::Vector3d::Zero());
  motion.readout = value.at("readout_s").get<std::array<double, 2>>();
  for (const auto& row : value.at("images")) {
    const auto i = row.at("image_index").get<size_t>();
    if (i >= count)
      throw std::runtime_error("Rolling-shutter motion refers to an unknown image");
    const auto w = row.at("rotation_rad").get<std::array<double, 3>>();
    const auto v = row.at("velocity_m_s").get<std::array<double, 3>>();
    motion.rotation[i] = {w[0], w[1], w[2]};
    motion.velocity[i] = {v[0], v[1], v[2]};
  }
  for (const double t : motion.readout)
    if (!std::isfinite(t) || std::abs(t) > ReadoutBound)
      throw std::runtime_error("Invalid rolling-shutter readout time");
  return motion;
}
namespace {
// Inputs shared by the readout-axis hypotheses.
struct ShutterInputs {
  Json source_quality;
  std::vector<Eigen::Vector3d> velocity;
  std::vector<Eigen::Vector3d> navigation;
  std::vector<PlaneConstraint> planes, training_planes;
  std::set<uint64_t> training;
  std::vector<uint64_t> validation;
};
ShutterCorrection CorrectAlongAxis(const Scene& scene, const Reconstruction& source,
                                   const ShutterInputs& inputs, int axis, int workers,
                                   const AlignmentLimits& limits, int ba_workers) {
  const auto started = Clock::now();
  const auto& source_quality = inputs.source_quality;
  const auto& planes = inputs.planes;
  const auto& validation = inputs.validation;
  ShutterCorrection result;
  auto model = source;
  auto& motion = result.motion;
  const size_t n = source.images.size();
  motion.axis = axis;
  motion.rotation.assign(n, Eigen::Vector3d::Zero());
  motion.velocity = inputs.velocity;
  Log(std::string("Rolling-shutter readout along ") + (axis ? "rows" : "columns"));
  auto& report = result.report;
  report = {{"readout_axis", axis ? "rows" : "columns"}, {"used", false}};
  report["optimization"] =
      Optimize(model, scene.cameras, "principal", ba_workers, 150, {}, inputs.training_planes,
               inputs.navigation, inputs.training, {}, nullptr, &motion);
  // Held-out reprojection with every camera fixed: the source projection against the
  // rolling-shutter projection, each with its points refit to the same observations.
  auto before = source;
  ParallelFor(validation.size(), workers, 64, [&](size_t k) {
    const uint64_t id = validation[k];
    RefinePoint(before, before.points[id].track, before.points[id].xyz);
    RefineShutterPoint(model, motion, model.points[id].track, model.points[id].xyz);
  });
  HoldoutErrors held_before, held_after;
  for (const uint64_t id : validation)
    for (const auto& o : source.points[id].track) {
      const auto& image = before.images[o.image];
      held_before.Add(o.image, (before.cameras[image.camera].Project(
                                    image.pose.Apply(before.points[id].xyz)) -
                                image.pixels[o.feature])
                                   .norm());
      held_after.Add(o.image, ShutterError(model, motion, o, model.points[id].xyz));
    }
  Json degraded = Json::array();
  size_t compared_images = 0;
  for (const auto& [image, errors] : held_before.images) {
    if (errors.size() < MinimumImageHoldout)
      continue;
    ++compared_images;
    const auto& after = held_after.images.at(image);
    // Per image only the median is held to the limit: with a few hundred held-out observations
    // an image's P90 follows its outliers. The P90 is checked over all held-out observations.
    // The limit also admits the sampling error of the image's median, the half-width of its 95%
    // distribution-free interval (order statistics l and n - l + 1).
    const double a = Quantile(errors, .5), b = Quantile(after, .5), c = Quantile(errors, .9),
                 d = Quantile(after, .9);
    auto sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    const size_t count = sorted.size(),
                 low = size_t(std::max(1., std::floor((count - 1.96 * std::sqrt(count)) / 2)));
    const double allowance = (sorted[count - low] - sorted[low - 1]) / 2,
                 limit = a * 1.1 + .1 + allowance;
    if (b > limit)
      degraded.push_back({{"image_index", image},
                          {"name", scene.images[image].name},
                          {"samples", errors.size()},
                          {"before_median", a},
                          {"after_median", b},
                          {"median_allowance", allowance},
                          {"median_limit", limit},
                          {"before_p90", c},
                          {"after_p90", d}});
  }
  // Every observation moves to the reference pose at the range of its point; observations
  // without a point use the median range of the image's points.
  auto& rectified = result.candidate;
  rectified = model;
  std::vector<std::vector<double>> shifts(n);
  ParallelFor(n, workers, 4, [&](size_t i) {
    auto& image = rectified.images[i];
    if (!image.registered)
      return;
    const auto& camera = model.cameras[image.camera];
    std::vector<double> ranges(image.pixels.size(), 0), known;
    for (uint32_t k = 0; k < image.pixels.size(); ++k)
      if (image.points[k] != NoPoint) {
        ranges[k] = Exposed(model, motion, {uint32_t(i), k}, model.points[image.points[k]].xyz).norm();
        known.push_back(ranges[k]);
      }
    const double fallback = known.empty() ? MaximumRange : Quantile(known, .5);
    for (uint32_t k = 0; k < image.pixels.size(); ++k) {
      const auto original = model.images[i].pixels[k];
      image.pixels[k] = Rectify(camera, motion, uint32_t(i), image.camera, original,
                                ranges[k] > 0 ? ranges[k] : fallback);
      if (image.points[k] != NoPoint)
        shifts[i].push_back((image.pixels[k] - original).norm());
    }
  });
  rectified.Filter();
  std::vector<double> shift;
  for (const auto& rows : shifts)
    shift.insert(shift.end(), rows.begin(), rows.end());
  const auto quality = rectified.Quality(scene, limits.median_px, limits.p90_px);
  const auto plane_before = PlaneErrors(source, planes), plane_after = PlaneErrors(rectified, planes);
  bool consistent = false;
  const auto consistency = ReadoutConsistency(scene, rectified, motion, consistent);
  std::vector<double> rotation, translation;
  bool bounded = true;
  for (uint32_t i = 0; i < n; ++i)
    if (source.images[i].registered) {
      rotation.push_back(motion.rotation[i].norm() * 180 / Pi);
      translation.push_back(std::abs(motion.readout[source.images[i].camera]) *
                            motion.velocity[i].norm());
      bounded = bounded && motion.rotation[i].cwiseAbs().maxCoeff() < RotationBound * .999;
    }
  for (uint32_t lens = 0; lens < 2; ++lens)
    bounded = bounded && std::abs(motion.readout[lens]) < ReadoutBound * .999;
  const auto validation_before = Statistics(held_before.all),
             validation_after = Statistics(held_after.all);
  const double source_p90 = source_quality.at("reprojection_px").at("p90");
  const Json checks = {
      {"visual_acceptance", quality.at("passed")},
      {"held_out_improvement",
       !held_before.all.empty() &&
           validation_after.at("median").get<double>() <
               validation_before.at("median").get<double>() &&
           validation_after.at("p90").get<double>() < validation_before.at("p90").get<double>()},
      {"per_image_validation", degraded.empty()},
      {"full_reprojection", quality.at("reprojection_px").at("p90").is_number() &&
                                quality.at("reprojection_px").at("p90").get<double>() <=
                                    source_p90 * 1.05 + .05},
      {"surface_fit", plane_after.at("p90").is_number() &&
                          plane_after.at("p90").get<double>() <=
                              plane_before.at("p90").get<double>() * 1.25 + .01},
      {"physical_readout", consistent},
      {"motion_within_bounds", bounded},
      {"registered_images_preserved", rectified.Registered() == source.Registered()}};
  result.accepted = FailedChecks(checks).empty();
  report["used"] = result.accepted;
  report["reason"] = result.accepted ? "accepted" : "rolling-shutter candidate failed acceptance";
  report["readout_s"] = motion.readout;
  report["readout_rotation_deg"] = Statistics(rotation);
  report["readout_translation_m"] = Statistics(translation);
  report["readout_consistency"] = consistency;
  report["observation_shift_px"] = Statistics(shift);
  report["validation_before_px"] = validation_before;
  report["validation_after_px"] = validation_after;
  report["validation_images_compared"] = compared_images;
  report["degraded_validation_images"] = degraded;
  report["plane_before_m"] = plane_before;
  report["plane_after_m"] = plane_after;
  report["limits"] = {{"minimum_image_holdout", MinimumImageHoldout},
                      {"validation_relative_increase", 1.1},
                      {"validation_absolute_increase_px", .1},
                      {"validation_median_allowance",
                       "half-width of the 95% order-statistic interval of the image's held-out "
                       "median before correction"},
                      {"minimum_explained_rotation", MinimumExplainedRotation},
                      {"rotation_bound_rad", RotationBound},
                      {"readout_bound_s", ReadoutBound}};
  report["checks"] = checks;
  report["failed_checks"] = FailedChecks(checks);
  report["source_quality"] = source_quality;
  report["candidate_quality"] = quality;
  report["seconds"] = Seconds(started);
  return result;
}
}  // namespace
ShutterCorrection CorrectRollingShutter(const Scene& scene, const Reconstruction& source,
                                        int workers, const AlignmentLimits& limits,
                                        int ba_workers) {
  if (!ba_workers)
    ba_workers = workers;
  const auto started = Clock::now();
  ShutterInputs inputs;
  inputs.source_quality = source.Quality(scene, limits.median_px, limits.p90_px);
  if (!inputs.source_quality.at("passed").get<bool>())
    throw std::runtime_error("Rolling-shutter source reconstruction failed visual acceptance");
  const auto velocities = NavigationVelocities(scene);
  inputs.velocity.assign(source.images.size(), Eigen::Vector3d::Zero());
  for (uint32_t i = 0; i < source.images.size(); ++i)
    if (source.images[i].registered)
      inputs.velocity[i] = source.images[i].pose.Rotation() * velocities[i];
  inputs.navigation = NavigationPositions(scene);
  Log("Rolling-shutter correction: " + std::to_string(source.Registered()) + " images");
  inputs.planes = LasPlaneConstraints(scene, source, workers);
  // Whole tracks are held out of the camera solve and refit afterwards with cameras fixed.
  for (uint64_t id = 0; id < source.points.size(); ++id) {
    const auto& point = source.points[id];
    if (point.track.size() < 2)
      continue;
    if (point.track.size() >= 3 && id % 5 == 0)
      inputs.validation.push_back(id);
    else
      inputs.training.insert(id);
  }
  for (const auto& plane : inputs.planes)
    if (inputs.training.count(plane.point)) {
      inputs.training_planes.push_back(plane);
      // Keep the accepted LAS offset; the constraints hold the metric frame, not a new fit.
      inputs.training_planes.back().center = source.points[plane.point].xyz;
    }
  // Sensors may be stored rotated, so rows and columns are both solved. An accepted hypothesis
  // wins over a rejected one; otherwise the lower held-out median decides.
  ShutterCorrection best;
  Json attempts = Json::array();
  for (const int axis : {1, 0}) {
    auto attempt = CorrectAlongAxis(scene, source, inputs, axis, workers, limits, ba_workers);
    const auto& r = attempt.report;
    Json explained = Json::array();
    for (const auto& lens : r.at("readout_consistency"))
      explained.push_back(lens.at("explained_rotation"));
    attempts.push_back({{"readout_axis", r.at("readout_axis")},
                        {"accepted", attempt.accepted},
                        {"failed_checks", r.at("failed_checks")},
                        {"readout_s", r.at("readout_s")},
                        {"explained_rotation", explained},
                        {"validation_after_px", r.at("validation_after_px")},
                        {"seconds", r.at("seconds")}});
    const auto held = [](const ShutterCorrection& c) {
      const auto& median = c.report.at("validation_after_px").at("median");
      return median.is_number() ? median.get<double>() : INFINITY;
    };
    if (best.report.is_null() || (attempt.accepted && !best.accepted) ||
        (attempt.accepted == best.accepted && held(attempt) < held(best)))
      best = std::move(attempt);
  }
  auto& report = best.report;
  report["method"] = "per-image readout rotation and per-lens readout time with the navigation "
                     "velocity, jointly with cameras, points and intrinsics, along rows and columns";
  report["version"] = 1;
  report["holdout_rule"] = "whole tracks of at least three views with IDs divisible by five";
  report["training_tracks"] = inputs.training.size();
  report["validation_tracks"] = inputs.validation.size();
  report["surface_constraints"] = inputs.planes.size();
  report["axis_attempts"] = attempts;
  report["seconds"] = Seconds(started);
  return best;
}
Json PrepareRectification(const Scene& scene, const Reconstruction& model,
                          const ShutterMotion& motion, const fs::path& work,
                          const fs::path& directory, int workers) {
  const auto started = Clock::now();
  int size = 0;
  for (const auto& camera : model.cameras)
    size = std::max({size, camera.width, camera.height});
  const int grid = (size + GridStep - 1) / GridStep;
  fs::create_directories(directory / "ranges");
  fs::create_directories(directory / "masks");
  const std::array<cv::Mat, 2> domains{MaskDomain(scene.cameras[0]), MaskDomain(scene.cameras[1])};
  std::vector<double> coverage;
  std::deque<std::future<double>> jobs;
  auto finish = [&] {
    coverage.push_back(jobs.front().get());
    jobs.pop_front();
    if (coverage.size() % 64 == 0)
      Log("Rolling-shutter rectification grids " + std::to_string(coverage.size()) + "/" +
          std::to_string(model.Registered()));
  };
  LasRangeMaps(scene, model, grid, directory, [&](uint32_t id, cv::Mat range) {
    if (jobs.size() >= size_t(std::max(1, workers)))
      finish();
    jobs.push_back(std::async(std::launch::async, [&, id, range = std::move(range)] {
      const uint32_t lens = model.images[id].camera;
      const auto& camera = model.cameras[lens];
      const double radius = camera.Radius();
      size_t inside = 0, measured = 0;
      for (int y = 0; y < range.rows; ++y)
        for (int x = 0; x < range.cols; ++x) {
          const double dx = ((x + .5) * camera.width / range.cols - camera.params[2]) /
                            camera.params[0],
                       dy = ((y + .5) * camera.height / range.rows - camera.params[3]) /
                            camera.params[1];
          if (dx * dx + dy * dy <= radius * radius) {
            ++inside;
            measured += range.at<float>(y, x) > 0;
          }
        }
      const auto millimetres = FillRanges(range);
      WriteImage(directory / "ranges" / (std::to_string(id) + ".png"), millimetres);
      auto mask = RectifyRaster(ReadImage(MaskPath(work, id), cv::IMREAD_GRAYSCALE), camera,
                                motion, id, lens, millimetres, cv::INTER_NEAREST);
      mask.setTo(0, domains[lens] == 0);
      WriteImage(directory / "masks" / (std::to_string(id) + ".png"), mask);
      return inside ? double(measured) / inside : 0.;
    }));
  });
  while (!jobs.empty())
    finish();
  return {{"grid_step_px", GridStep},
          {"range_grid_size", grid},
          {"range_source", "nearest valid LAS record per grid cell, holes filled in inverse range"},
          {"las_range_coverage", Statistics(coverage)},
          {"images", coverage.size()},
          {"seconds", Seconds(started)}};
}
cv::Mat RectifyRaster(const cv::Mat& source, const Camera& camera, const ShutterMotion& motion,
                      uint32_t image, uint32_t lens, const cv::Mat& range_mm, int interpolation) {
  if (source.cols != camera.width || source.rows != camera.height || range_mm.empty() ||
      range_mm.type() != CV_16UC1)
    throw std::runtime_error("Rolling-shutter rectification input differs from its camera");
  const int w = camera.width, h = camera.height;
  const int gw = (w + GridStep - 1) / GridStep + 1, gh = (h + GridStep - 1) / GridStep + 1;
  const double sx = double(range_mm.cols) / w, sy = double(range_mm.rows) / h;
  // Outside the valid fisheye domain the masks are 0; keep those pixels in place.
  const double radius = camera.Radius() * 1.02;
  cv::Mat nodes(gh, gw, CV_32FC2);
  for (int j = 0; j < gh; ++j)
    for (int i = 0; i < gw; ++i) {
      const Eigen::Vector2d q(i * GridStep + .5, j * GridStep + .5);
      const double dx = (q.x() - camera.params[2]) / camera.params[0],
                   dy = (q.y() - camera.params[3]) / camera.params[1];
      Eigen::Vector2d p = q;
      if (dx * dx + dy * dy <= radius * radius) {
        const Eigen::Vector3d reference =
            SampleRange(range_mm, q.x() * sx - .5, q.y() * sy - .5) * camera.Ray(q);
        for (int iteration = 0; iteration < 4; ++iteration)
          p = camera.Project(
              motion.Apply(image, lens, motion.Fraction(camera, p), reference));
      }
      nodes.at<cv::Vec2f>(j, i) = cv::Vec2f(float(p.x() - .5), float(p.y() - .5));
    }
  cv::Mat output(source.size(), source.type());
  constexpr int strip = 256;
  for (int y0 = 0; y0 < h; y0 += strip) {
    const int rows = std::min(strip, h - y0);
    cv::Mat map(rows, w, CV_32FC2);
    for (int y = 0; y < rows; ++y) {
      const double gy = double(y0 + y) / GridStep;
      const int j = std::min(int(gy), gh - 2);
      const float fy = float(gy - j);
      auto* row = map.ptr<cv::Vec2f>(y);
      const auto* top = nodes.ptr<cv::Vec2f>(j);
      const auto* bottom = nodes.ptr<cv::Vec2f>(j + 1);
      for (int x = 0; x < w; ++x) {
        const double gx = double(x) / GridStep;
        const int i = std::min(int(gx), gw - 2);
        const float fx = float(gx - i);
        row[x] = (1 - fy) * ((1 - fx) * top[i] + fx * top[i + 1]) +
                 fy * ((1 - fx) * bottom[i] + fx * bottom[i + 1]);
      }
    }
    cv::Mat destination = output.rowRange(y0, y0 + rows);
    cv::remap(source, destination, map, cv::noArray(), interpolation, cv::BORDER_CONSTANT,
              cv::Scalar::all(0));
  }
  return output;
}
double ShutterPointError(const Reconstruction& model, const ShutterMotion& motion,
                         const ModelPoint& point) {
  double error = 0;
  for (const auto& o : point.track)
    error += ShutterError(model, motion, o, point.xyz);
  return point.track.empty() ? 0 : error / point.track.size();
}
Reconstruction OriginalPhotoObservations(const Reconstruction& corrected,
                                         const ShutterMotion& motion, int workers) {
  auto original = corrected;
  ParallelFor(corrected.images.size(), std::max(1, workers), 4, [&](size_t i) {
    const auto& source = corrected.images[i];
    auto& image = original.images[i];
    if (!image.registered)
      return;
    const auto& camera = corrected.cameras[image.camera];
    std::vector<double> known;
    for (uint32_t k = 0; k < image.pixels.size(); ++k)
      if (image.points[k] != NoPoint)
        known.push_back(image.pose.Apply(corrected.points[image.points[k]].xyz).norm());
    const double fallback = known.empty() ? MaximumRange : Quantile(known, .5);
    // Invert the rectification: the original pixel is the one that rectifies to the delivered
    // reference pixel at the range of its point.
    for (uint32_t k = 0; k < image.pixels.size(); ++k) {
      const auto& target = source.pixels[k];
      Eigen::Vector2d pixel = target;
      for (int iteration = 0; iteration < 8; ++iteration) {
        double range = fallback;
        if (image.points[k] != NoPoint)
          range = motion
                      .Apply(uint32_t(i), image.camera, motion.Fraction(camera, pixel),
                             image.pose.Apply(corrected.points[image.points[k]].xyz))
                      .norm();
        const Eigen::Vector2d step =
            target - Rectify(camera, motion, uint32_t(i), image.camera, pixel, range);
        pixel += step;
        if (step.norm() < 1e-9)
          break;
      }
      image.pixels[k] = pixel;
    }
  });
  for (auto& point : original.points)
    point.error = ShutterPointError(original, motion, point);
  return original;
}
Json DeliveryShutterJson(const Scene& scene, const Reconstruction& model,
                         const ShutterMotion& motion) {
  Json images = Json::array();
  uint32_t image_id = 0;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      const auto& w = motion.rotation[i];
      const auto& v = motion.velocity[i];
      const Eigen::Vector3d moved = motion.readout[model.images[i].camera] * v;
      images.push_back({{"image_id", ++image_id},
                        {"camera_id", model.images[i].camera + 1},
                        {"name", scene.images[i].name},
                        {"rotation_rad", {w.x(), w.y(), w.z()}},
                        {"velocity_m_s", {v.x(), v.y(), v.z()}},
                        {"translation_m", {moved.x(), moved.y(), moved.z()}}});
    }
  return {{"version", 1},
          {"model", "linear rolling shutter about the reference pose"},
          {"readout_axis", motion.axis ? "rows" : "columns"},
          {"readout_fraction",
           "s = (c - N / 2) / N, c the pixel coordinate along the readout axis with the top-left "
           "pixel centre at (0.5, 0.5), N the image width or height along that axis"},
          {"camera_point",
           "p(s) = (I + s [rotation_rad]x) (R X + t - s translation_m), translation_m = readout_s "
           "velocity_m_s"},
          {"reference_pose", "COLMAP images.bin world-to-camera R, t, the pose at s = 0"},
          {"frames", "rotation_rad, velocity_m_s and translation_m in the OpenCV reference camera; "
                     "rotation_rad and translation_m span one full readout"},
          {"projection", "the pixel x of X satisfies x = fisheye(p(s(x))); iterate from s = 0"},
          {"cameras",
           {{{"camera_id", 1}, {"readout_s", motion.readout[0]}},
            {{"camera_id", 2}, {"readout_s", motion.readout[1]}}}},
          {"images", images}};
}
ShutterMotion DeliveryShutterFromJson(const Json& value, const Scene& scene,
                                      const Reconstruction& model) {
  if (value.at("version") != 1)
    throw std::runtime_error("Unsupported rolling-shutter delivery version");
  ShutterMotion motion;
  const auto axis = value.at("readout_axis").get<std::string>();
  if (axis != "rows" && axis != "columns")
    throw std::runtime_error("Unknown rolling-shutter readout axis: " + axis);
  motion.axis = axis == "rows";
  for (uint32_t c = 0; c < 2; ++c) {
    const auto& row = value.at("cameras").at(c);
    if (row.at("camera_id") != c + 1)
      throw std::runtime_error("Rolling-shutter camera identity differs");
    motion.readout[c] = row.at("readout_s");
    if (!std::isfinite(motion.readout[c]) || std::abs(motion.readout[c]) > ReadoutBound)
      throw std::runtime_error("Invalid rolling-shutter readout time");
  }
  const auto& images = value.at("images");
  if (images.size() != model.images.size())
    throw std::runtime_error("Rolling-shutter image count differs");
  for (uint32_t i = 0; i < model.images.size(); ++i) {
    const auto& row = images[i];
    if (row.at("image_id") != i + 1 || row.at("name") != scene.images[i].name)
      throw std::runtime_error("Rolling-shutter image identity differs");
    const auto w = row.at("rotation_rad").get<std::array<double, 3>>();
    const auto v = row.at("velocity_m_s").get<std::array<double, 3>>();
    const auto t = row.at("translation_m").get<std::array<double, 3>>();
    motion.rotation.emplace_back(w[0], w[1], w[2]);
    motion.velocity.emplace_back(v[0], v[1], v[2]);
    const Eigen::Vector3d moved(t[0], t[1], t[2]);
    if ((moved - motion.readout[model.images[i].camera] * motion.velocity.back()).norm() > 1e-9)
      throw std::runtime_error("Rolling-shutter translation differs from readout and velocity");
  }
  return motion;
}
void WriteRectifiedJpeg(const std::vector<uint8_t>& original, const cv::Mat& image,
                        const fs::path& destination) {
  std::vector<uint8_t> encoded, metadata;
  if (!cv::imencode(".jpg", image, encoded, {cv::IMWRITE_JPEG_QUALITY, 95}))
    throw std::runtime_error("Cannot encode rectified image");
  MetadataEnd(original, &metadata);
  const size_t body = MetadataEnd(encoded, nullptr);
  std::ofstream stream(destination, std::ios::binary);
  stream.write("\xFF\xD8", 2);
  stream.write(reinterpret_cast<const char*>(metadata.data()), std::streamsize(metadata.size()));
  stream.write(reinterpret_cast<const char*>(encoded.data() + body),
               std::streamsize(encoded.size() - body));
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write rectified image: " + destination.u8string());
}
}  // namespace geosun

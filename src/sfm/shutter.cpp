#include "sfm/shutter.h"
#include "frontend/frontend.h"
#include "geometry/supervision.h"
#include <algorithm>
#include <optional>
#include <set>

namespace geosun {
namespace {
constexpr double MaximumRange = 65.535;
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
    cost += TrackCost(error);
  }
  return cost;
}
// RefinePoint with the cameras and their readout motion fixed.
void RefineShutterPoint(const Reconstruction& model, const ShutterMotion& motion,
                        const std::vector<Observation>& track, Eigen::Vector3d& point) {
  RefineTrackPoint(
      track, point,
      [&](const Eigen::Vector3d& p) { return ShutterPointCost(model, motion, track, p); },
      [&](const Observation& o, const Eigen::Vector3d& p, Eigen::Vector2d& residual,
          Eigen::Matrix<double, 2, 3>& j) {
        const auto& image = model.images[o.image];
        const auto& camera = model.cameras[image.camera];
        const auto& pixel = image.pixels[o.feature];
        const double s = motion.Fraction(camera, pixel);
        const Eigen::Vector3d v = motion.Apply(o.image, image.camera, s, image.pose.Apply(p));
        residual = camera.Project(v) - pixel;
        j = camera.ProjectionJacobian(v) * Readout(motion, o.image, s) *
            image.pose.Rotation().toRotationMatrix();
      },
      "Nonfinite rolling-shutter triangulation refinement");
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
  ExposureKinematics kinematics;
  std::vector<Eigen::Vector3d> velocity;
  std::vector<Eigen::Vector3d> navigation;
  std::vector<PlaneConstraint> planes, training_planes;
  std::set<uint64_t> training;
  std::vector<uint64_t> validation;
  std::vector<std::array<uint32_t, 2>> frames;
};
struct EvaluatedAxis {
  ShutterCorrection correction;
  Reconstruction unrectified;
};
Reconstruction RectifyModel(const Reconstruction& model, const ShutterMotion& motion, int workers,
                            std::vector<double>& shift) {
  Reconstruction rectified = model;
  const size_t n = model.images.size();
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
  shift.clear();
  for (const auto& rows : shifts)
    shift.insert(shift.end(), rows.begin(), rows.end());
  return rectified;
}
bool MotionStats(const Reconstruction& model, const ShutterMotion& motion,
                 std::vector<double>& rotation, std::vector<double>& translation) {
  rotation.clear();
  translation.clear();
  bool bounded = true;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      rotation.push_back(motion.rotation[i].norm() * 180 / Pi);
      translation.push_back(std::abs(motion.readout[model.images[i].camera]) *
                            motion.velocity[i].norm());
      bounded = bounded && motion.rotation[i].cwiseAbs().maxCoeff() < RotationBound * .999;
    }
  for (uint32_t lens = 0; lens < 2; ++lens)
    bounded = bounded && std::abs(motion.readout[lens]) < ReadoutBound * .999;
  return bounded;
}
struct AxisScreen {
  int axis = 1;
  bool consistent = false;
  std::array<double, 2> readout{};
  std::vector<double> explained;
  double cost_drop_ratio = 0;
  size_t tracks = 0;
  double seconds = 0;
  Json ToJson() const {
    return {{"readout_axis", axis ? "rows" : "columns"},
            {"consistent", consistent},
            {"readout_s", readout},
            {"explained_rotation", explained},
            {"cost_drop_ratio", cost_drop_ratio},
            {"tracks", tracks},
            {"seconds", seconds}};
  }
};
std::set<uint64_t> SelectScreenTracks(const Reconstruction& source,
                                      const std::set<uint64_t>& training) {
  std::vector<uint64_t> pool;
  pool.reserve(training.size());
  for (uint64_t id : training)
    if (source.points[id].track.size() >= 3)
      pool.push_back(id);
  if (pool.empty())
    pool.assign(training.begin(), training.end());
  const size_t stride = std::max<size_t>(1, pool.size() / 8000);
  std::set<uint64_t> selected;
  std::vector<size_t> observations(source.images.size(), 0);
  auto add = [&](uint64_t id) {
    if (!selected.insert(id).second)
      return;
    for (const auto& o : source.points[id].track)
      ++observations[o.image];
  };
  for (size_t i = 0; i < pool.size(); i += stride)
    add(pool[i]);
  for (uint64_t id : training)
    for (const auto& o : source.points[id].track)
      if (observations[o.image] < 24) {
        add(id);
        break;
      }
  return selected;
}
AxisScreen ScreenAlongAxis(const Scene& scene, const Reconstruction& source,
                           const ShutterInputs& inputs, const std::set<uint64_t>& screen_tracks,
                           const std::vector<PlaneConstraint>& screen_planes, int axis,
                           int ba_workers) {
  const auto started = Clock::now();
  AxisScreen screen;
  screen.axis = axis;
  screen.tracks = screen_tracks.size();
  auto model = source;
  ShutterMotion motion;
  motion.axis = axis;
  motion.rotation.assign(source.images.size(), Eigen::Vector3d::Zero());
  motion.velocity = inputs.velocity;
  const auto opt = Optimize(model, scene.cameras, "fixed", ba_workers, 3, {}, screen_planes,
                            inputs.navigation, screen_tracks, &motion, inputs.frames);
  const auto consistency = ReadoutConsistency(scene, model, motion, screen.consistent);
  screen.readout = motion.readout;
  for (const auto& lens : consistency)
    screen.explained.push_back(lens.at("explained_rotation").get<double>());
  const double initial_cost = opt.at("initial_cost").get<double>(),
               final_cost = opt.at("final_cost").get<double>();
  screen.cost_drop_ratio = initial_cost > 0 ? (initial_cost - final_cost) / initial_cost : 0;
  screen.seconds = Seconds(started);
  return screen;
}
EvaluatedAxis CorrectAlongAxis(const Scene& scene, const Reconstruction& source,
                               const ShutterInputs& inputs, int axis,
                               std::array<double, 2> initial_readout, int workers,
                               const AlignmentLimits& limits, int ba_workers) {
  const auto started = Clock::now();
  const auto& source_quality = inputs.source_quality;
  const auto& planes = inputs.planes;
  const auto& validation = inputs.validation;
  EvaluatedAxis evaluated;
  auto& result = evaluated.correction;
  auto model = source;
  auto& motion = result.motion;
  const size_t n = source.images.size();
  motion.axis = axis;
  motion.rotation.assign(n, Eigen::Vector3d::Zero());
  motion.velocity = inputs.velocity;
  for (size_t c = 0; c < 2; ++c)
    motion.readout[c] = std::clamp(initial_readout[c], -.04, .04);
  Log(std::string("Rolling-shutter readout along ") + (axis ? "rows" : "columns"));
  auto& report = result.report;
  report = {{"readout_axis", axis ? "rows" : "columns"}, {"used", false}};
  // Synchronous L/R reference (middle-line) poses are rigid; this is the only BA given the rig.
  report["optimization"] =
      Optimize(model, scene.cameras, "principal", ba_workers, 150, {}, inputs.training_planes,
               inputs.navigation, inputs.training, &motion, inputs.frames);
  // Held-out reprojection with every camera fixed: the source projection against the
  // rolling-shutter projection, each with its points refit to the same observations.
  auto before = source;
  ParallelFor(validation.size(), workers, 64, [&](size_t k) {
    const uint64_t id = validation[k];
    RefinePoint(before, before.points[id].track, before.points[id].xyz);
    RefineShutterPoint(model, motion, model.points[id].track, model.points[id].xyz);
  });
  evaluated.unrectified = model;
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
  std::vector<double> shift;
  auto& rectified = result.candidate;
  rectified = RectifyModel(model, motion, workers, shift);
  const auto quality = rectified.Quality(scene, limits.median_px, limits.p90_px);
  const auto plane_before = PlaneErrors(source, planes), plane_after = PlaneErrors(rectified, planes);
  bool consistent = false;
  const auto consistency = ReadoutConsistency(scene, rectified, motion, consistent);
  std::vector<double> rotation, translation;
  const bool bounded = MotionStats(source, motion, rotation, translation);
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
  return evaluated;
}
void PolishAcceptedShutter(const Scene& scene, const Reconstruction& source,
                           const ShutterInputs& inputs, int workers, const AlignmentLimits& limits,
                           int ba_workers, EvaluatedAxis& accepted_axis) {
  const auto started = Clock::now();
  Reconstruction model = accepted_axis.unrectified;
  ShutterMotion motion = accepted_axis.correction.motion;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      const Eigen::Vector3d v_nav = model.images[i].pose.Rotation() * inputs.kinematics.velocity[i];
      const Eigen::Vector3d r_cam =
          model.images[i].pose.Rotation() * inputs.kinematics.lever_arm[i];
      const double tau = motion.readout[model.images[i].camera];
      motion.velocity[i] = std::abs(tau) > 1e-3
                               ? Eigen::Vector3d(v_nav - (motion.rotation[i] / tau).cross(r_cam))
                               : v_nav;
    }
  std::set<uint64_t> all_tracks;
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& point = model.points[id];
    if (point.track.size() < 2 || !point.xyz.allFinite())
      continue;
    const bool positive_depth =
        std::all_of(point.track.begin(), point.track.end(), [&](const auto& o) {
          return model.images[o.image].registered &&
                 model.images[o.image].pose.Apply(point.xyz).z() > 1e-3;
        });
    if (positive_depth) {
      all_tracks.insert(id);
      model.points[id].error = ShutterPointError(model, motion, model.points[id]);
    }
  }
  auto& report = accepted_axis.correction.report;
  const auto polish_planes = LasPlaneConstraints(scene, model, workers);
  Log("Rolling-shutter polish: " + std::to_string(all_tracks.size()) + " tracks, " +
      std::to_string(polish_planes.size()) + " LAS planes");
  const auto polish_opt =
      Optimize(model, scene.cameras, "principal", ba_workers, 100, {}, polish_planes,
               inputs.navigation, all_tracks, &motion, inputs.frames);
  std::vector<double> shift;
  auto rectified = RectifyModel(model, motion, workers, shift);
  const auto quality = rectified.Quality(scene, limits.median_px, limits.p90_px);
  const auto plane_after = PlaneErrors(rectified, inputs.planes);
  const auto polish_plane_fit = PlaneErrors(rectified, polish_planes);
  bool consistent = false;
  const auto consistency = ReadoutConsistency(scene, rectified, motion, consistent);
  std::vector<double> rotation, translation;
  const bool bounded = MotionStats(source, motion, rotation, translation);
  const auto& unpolished_quality = report.at("candidate_quality");
  const auto& source_images = inputs.source_quality.at("images");
  const auto& unpolished_images = unpolished_quality.at("images");
  const auto& polished_images = quality.at("images");
  bool per_image_stable = true;
  for (size_t i = 0; i < source.images.size(); ++i) {
    if (!source.images[i].registered)
      continue;
    const auto& a_p90 = source_images[i].at("reprojection_px").at("p90");
    const auto& b_p90 = unpolished_images[i].at("reprojection_px").at("p90");
    const auto& c_p90 = polished_images[i].at("reprojection_px").at("p90");
    if (!a_p90.is_number() || !b_p90.is_number() || !c_p90.is_number() ||
        c_p90.get<double>() > std::max(a_p90.get<double>(), b_p90.get<double>()) * 1.1 + .1) {
      per_image_stable = false;
      break;
    }
  }
  const double source_p90 = inputs.source_quality.at("reprojection_px").at("p90");
  const auto& plane_before = report.at("plane_before_m");
  const Json checks = {
      {"visual_acceptance", quality.at("passed")},
      {"per_image_stability", per_image_stable},
      {"full_reprojection", quality.at("reprojection_px").at("p90").is_number() &&
                                quality.at("reprojection_px").at("p90").get<double>() <=
                                    source_p90 * 1.05 + .05},
      {"surface_fit", inputs.planes.empty() ||
                          (plane_after.at("p90").is_number() &&
                           plane_after.at("p90").get<double>() <=
                               plane_before.at("p90").get<double>() * 1.25 + .01)},
      {"physical_readout", consistent},
      {"motion_within_bounds", bounded},
      {"registered_images_preserved", rectified.Registered() == source.Registered()}};
  const auto failed = FailedChecks(checks);
  if (!failed.empty()) {
    report["polish"] = {{"used", false},
                        {"tracks", all_tracks.size()},
                        {"surface_constraints", polish_planes.size()},
                        {"checks", checks},
                        {"failed_checks", failed},
                        {"seconds", Seconds(started)}};
    return;
  }
  accepted_axis.correction.candidate = std::move(rectified);
  accepted_axis.correction.motion = std::move(motion);
  report["readout_s"] = accepted_axis.correction.motion.readout;
  report["readout_rotation_deg"] = Statistics(rotation);
  report["readout_translation_m"] = Statistics(translation);
  report["readout_consistency"] = consistency;
  report["observation_shift_px"] = Statistics(shift);
  report["plane_after_m"] = plane_after;
  report["candidate_quality"] = quality;
  report["polish"] = {{"used", true},
                      {"tracks", all_tracks.size()},
                      {"surface_constraints", polish_planes.size()},
                      {"plane_after_reassociated_m", polish_plane_fit},
                      {"optimization", polish_opt},
                      {"checks", checks},
                      {"seconds", Seconds(started)}};
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
  inputs.kinematics = ReadExposureKinematics(scene);
  inputs.velocity.assign(source.images.size(), Eigen::Vector3d::Zero());
  for (uint32_t i = 0; i < source.images.size(); ++i)
    if (source.images[i].registered)
      inputs.velocity[i] = source.images[i].pose.Rotation() * inputs.kinematics.velocity[i];
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
  inputs.frames = SynchronousFrames(scene);
  for (const auto& plane : inputs.planes)
    if (inputs.training.count(plane.point)) {
      inputs.training_planes.push_back(plane);
      // Keep the accepted LAS offset; the constraints hold the metric frame, not a new fit.
      inputs.training_planes.back().center = source.points[plane.point].xyz;
    }
  // Screen both readout axes sequentially on a compact multi-view track subset so the full BA runs
  // uncontended on ba_workers for the winning axis, falling back to the second axis if the first
  // fails acceptance or screening does not unambiguously separate them.
  const auto screen_tracks = SelectScreenTracks(source, inputs.training);
  std::vector<PlaneConstraint> screen_planes;
  for (const auto& plane : inputs.training_planes)
    if (screen_tracks.count(plane.point))
      screen_planes.push_back(plane);
  const std::array<AxisScreen, 2> screens{
      ScreenAlongAxis(scene, source, inputs, screen_tracks, screen_planes, 1, ba_workers),
      ScreenAlongAxis(scene, source, inputs, screen_tracks, screen_planes, 0, ba_workers)};
  for (const auto& s : screens)
    Log(std::string("Shutter axis screen ") + (s.axis ? "rows" : "columns") + ": consistent=" +
        (s.consistent ? "true" : "false") + ", cost_drop=" + std::to_string(s.cost_drop_ratio) +
        ", seconds=" + std::to_string(s.seconds));
  const bool prefer_columns = (screens[1].consistent != screens[0].consistent)
                                  ? screens[1].consistent
                                  : (screens[1].cost_drop_ratio > screens[0].cost_drop_ratio);
  const size_t primary_idx = prefer_columns ? 1 : 0, secondary_idx = 1 - primary_idx;
  std::array<std::optional<EvaluatedAxis>, 2> full;
  full[primary_idx] = CorrectAlongAxis(scene, source, inputs, screens[primary_idx].axis,
                                       screens[primary_idx].readout, workers, limits, ba_workers);
  const bool unambiguous = full[primary_idx]->correction.accepted &&
                           screens[primary_idx].consistent && !screens[secondary_idx].consistent;
  if (!unambiguous)
    full[secondary_idx] =
        CorrectAlongAxis(scene, source, inputs, screens[secondary_idx].axis,
                         screens[secondary_idx].readout, workers, limits, ba_workers);
  EvaluatedAxis best;
  Json attempts = Json::array(), screen_report = Json::array();
  const auto held = [](const ShutterCorrection& c) {
    const auto& median = c.report.at("validation_after_px").at("median");
    return median.is_number() ? median.get<double>() : INFINITY;
  };
  for (size_t idx = 0; idx < 2; ++idx) {
    const auto& s = screens[idx];
    screen_report.push_back(s.ToJson());
    if (full[idx]) {
      auto& attempt = *full[idx];
      const auto& r = attempt.correction.report;
      Json explained = Json::array();
      for (const auto& lens : r.at("readout_consistency"))
        explained.push_back(lens.at("explained_rotation"));
      attempts.push_back({{"readout_axis", r.at("readout_axis")},
                          {"evaluated", "full"},
                          {"accepted", attempt.correction.accepted},
                          {"failed_checks", r.at("failed_checks")},
                          {"readout_s", r.at("readout_s")},
                          {"explained_rotation", explained},
                          {"validation_after_px", r.at("validation_after_px")},
                          {"seconds", r.at("seconds")}});
      if (best.correction.report.is_null() ||
          (attempt.correction.accepted && !best.correction.accepted) ||
          (attempt.correction.accepted == best.correction.accepted &&
           held(attempt.correction) < held(best.correction)))
        best = std::move(attempt);
    } else {
      attempts.push_back({{"readout_axis", s.axis ? "rows" : "columns"},
                          {"evaluated", "screen"},
                          {"accepted", false},
                          {"failed_checks", Json::array({"physical_readout"})},
                          {"readout_s", s.readout},
                          {"explained_rotation", s.explained},
                          {"cost_drop_ratio", s.cost_drop_ratio},
                          {"seconds", s.seconds}});
    }
  }
  if (best.correction.accepted)
    PolishAcceptedShutter(scene, source, inputs, workers, limits, ba_workers, best);
  auto& report = best.correction.report;
  report["method"] = "per-image readout rotation and per-lens readout time with the navigation "
                     "velocity, jointly with cameras, points and intrinsics, screened along rows "
                     "and columns";
  report["holdout_rule"] = "whole tracks of at least three views with IDs divisible by five";
  report["training_tracks"] = inputs.training.size();
  report["validation_tracks"] = inputs.validation.size();
  report["surface_constraints"] = inputs.planes.size();
  report["axis_screen"] = screen_report;
  report["axis_attempts"] = attempts;
  report["seconds"] = Seconds(started);
  return std::move(best.correction);
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
  return {{"model", "linear rolling shutter about the reference pose"},
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
    if (row.at("image_id") != i + 1 || row.at("name") != scene.images[i].name ||
        (row.contains("camera_id") && row.at("camera_id") != model.images[i].camera + 1))
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
}  // namespace geosun

#include "sfm/shutter.h"
#include "frontend/frontend.h"
#include "geometry/supervision.h"
#include "io/cache.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>

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
bool HasPositiveReferenceDepth(const Reconstruction& model, const ModelPoint& point,
                               double min_depth = 1e-3) {
  return point.xyz.allFinite() &&
         std::all_of(point.track.begin(), point.track.end(), [&](const Observation& o) {
           return model.images[o.image].registered &&
                  model.images[o.image].pose.Apply(point.xyz).z() > min_depth;
         });
}
bool HasPositiveShutterDepth(const Reconstruction& model, const ShutterMotion& motion,
                             const ModelPoint& point, double min_depth = 1e-3) {
  return HasPositiveReferenceDepth(model, point, min_depth) &&
         std::all_of(point.track.begin(), point.track.end(), [&](const Observation& o) {
           return Exposed(model, motion, o, point.xyz).z() > min_depth;
         });
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
  std::vector<RigFrame> frames;
  bool imu_attitude_verified = false;
  std::array<double, 2> imu_boresight_median_deg{-1., -1.};
  std::vector<Eigen::Vector3d> imu_omega;
  std::string rig_mode = "none";
};
std::array<double, 2> VerifyImuAttitude(const Reconstruction& model,
                                        const ExposureKinematics& kinematics, bool& verified) {
  std::array<double, 2> medians{-1., -1.};
  verified = false;
  if (!kinematics.has_attitude || kinematics.body_rotation.size() != model.images.size())
    return medians;
  size_t verified_lenses = 0;
  for (uint32_t cid = 0; cid < 2; ++cid) {
    std::vector<Eigen::Quaterniond> samples;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        samples.push_back(
            (model.images[i].pose.Rotation() * kinematics.body_rotation[i]).normalized());
    if (samples.size() < 8)
      continue;
    std::vector<uint8_t> inlier(samples.size(), 1);
    Eigen::Quaterniond q_mean = Eigen::Quaterniond::Identity();
    for (int pass = 0; pass < 3; ++pass) {
      Eigen::Matrix4d sum = Eigen::Matrix4d::Zero();
      size_t count = 0;
      for (size_t k = 0; k < samples.size(); ++k)
        if (inlier[k]) {
          sum += samples[k].coeffs() * samples[k].coeffs().transpose();
          ++count;
        }
      if (count < 4)
        return medians;
      const Eigen::Vector4d eig =
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(sum).eigenvectors().col(3);
      q_mean = Eigen::Quaterniond(eig[3], eig[0], eig[1], eig[2]).normalized();
      for (size_t k = 0; k < samples.size(); ++k)
        inlier[k] = samples[k].angularDistance(q_mean) <= 5.0 * Pi / 180.0;
    }
    std::vector<double> errors_deg;
    errors_deg.reserve(samples.size());
    for (const auto& q : samples)
      errors_deg.push_back(q.angularDistance(q_mean) * 180.0 / Pi);
    medians[cid] = Quantile(errors_deg, .5);
    if (medians[cid] > 2.0)
      return medians;
    ++verified_lenses;
  }
  verified = verified_lenses > 0;
  return medians;
}
std::vector<RigFrame> BuildRigFrames(const Scene& scene, const Reconstruction& model,
                                     const ShutterInputs& inputs, std::string& rig_mode) {
  rig_mode = "none";
  std::map<int, std::array<uint32_t, 2>> lenses;
  for (uint32_t i = 0; i < scene.images.size(); ++i)
    lenses.try_emplace(scene.images[i].frame, std::array<uint32_t, 2>{UINT32_MAX, UINT32_MAX})
        .first->second[scene.images[i].camera] = i;
  std::vector<std::array<uint32_t, 2>> paired;
  double max_abs_dt = 0;
  for (const auto& [frame, ids] : lenses) {
    (void)frame;
    if (ids[0] == UINT32_MAX || ids[1] == UINT32_MAX)
      continue;
    const auto& a = scene.images[ids[0]];
    const auto& b = scene.images[ids[1]];
    if (a.week != b.week)
      return {};
    max_abs_dt = std::max(max_abs_dt, std::abs(a.seconds - b.seconds));
    paired.push_back(ids);
  }
  if (paired.empty())
    return {};
  std::vector<RigFrame> frames;
  frames.reserve(paired.size());
  if (max_abs_dt <= .002) {
    rig_mode = "synchronous";
    for (const auto& ids : paired)
      frames.push_back({ids[0], ids[1],
                        ExposureTime(scene.images[ids[0]]) - ExposureTime(scene.images[ids[1]]),
                        true, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()});
    return frames;
  }
  if (!inputs.imu_attitude_verified || max_abs_dt > .2)
    return {};
  std::array<std::vector<double>, 3> offsets;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      const Eigen::Vector3d diff = model.images[i].pose.Center() - inputs.navigation[i];
      for (int k = 0; k < 3; ++k)
        offsets[k].push_back(diff[k]);
    }
  if (offsets[0].empty())
    return {};
  const Eigen::Vector3d c_offset(Quantile(offsets[0], .5), Quantile(offsets[1], .5),
                                 Quantile(offsets[2], .5));
  rig_mode = "asynchronous";
  for (const auto& ids : paired) {
    const uint32_t l = ids[0], r = ids[1];
    const Eigen::Quaterniond dR =
        (inputs.kinematics.body_rotation[l] * inputs.kinematics.body_rotation[r].conjugate())
            .normalized();
    const Eigen::Vector3d p_l = inputs.kinematics.body_position[l] + c_offset;
    const Eigen::Vector3d p_r = inputs.kinematics.body_position[r] + c_offset;
    const Eigen::Vector3d dt = p_l - dR * p_r;
    frames.push_back({l, r, ExposureTime(scene.images[l]) - ExposureTime(scene.images[r]), false,
                      dR, dt});
  }
  return frames;
}
Pose PredictCompanionPose(const Pose& known, const Pose& rig, const RigFrame& f,
                          bool missing_is_right) {
  Pose predicted;
  if (missing_is_right) {
    const Eigen::Quaterniond q_lt = (known.Rotation() * f.delta_rotation).normalized();
    const Eigen::Vector3d t_lt = known.Rotation() * f.delta_translation + known.Translation();
    predicted.Set((rig.Rotation() * q_lt).normalized(),
                  rig.Rotation() * t_lt + rig.Translation());
  } else {
    const Eigen::Quaterniond q_lt = (rig.Rotation().conjugate() * known.Rotation()).normalized();
    const Eigen::Vector3d t_lt =
        rig.Rotation().conjugate() * (known.Translation() - rig.Translation());
    const Eigen::Quaterniond q_l = (q_lt * f.delta_rotation.conjugate()).normalized();
    const Eigen::Vector3d t_l = t_lt - q_l * f.delta_translation;
    predicted.Set(q_l, t_l);
  }
  return predicted;
}
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
                            inputs.navigation, screen_tracks, &motion, inputs.frames, {}, false,
                            inputs.imu_omega);
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
  std::set<uint64_t> active_training;
  for (const uint64_t id : inputs.training)
    if (HasPositiveShutterDepth(model, motion, model.points[id]))
      active_training.insert(id);
  Log(std::string("Rolling-shutter readout along ") + (axis ? "rows" : "columns"));
  auto& report = result.report;
  report = {{"readout_axis", axis ? "rows" : "columns"}, {"used", false}};
  // Synchronous L/R reference (middle-line) poses are rigid; this is the only BA given the rig.
  report["optimization"] =
      Optimize(model, scene.cameras, "principal", ba_workers, 150, {}, inputs.training_planes,
               inputs.navigation, active_training, &motion, inputs.frames, {}, false,
               inputs.imu_omega);
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
    if (point.track.size() < 2 || !HasPositiveShutterDepth(model, motion, point))
      continue;
    all_tracks.insert(id);
    model.points[id].error = ShutterPointError(model, motion, model.points[id]);
  }
  auto& report = accepted_axis.correction.report;
  const auto polish_planes = LasPlaneConstraints(scene, model, workers);
  Log("Rolling-shutter polish: " + std::to_string(all_tracks.size()) + " tracks, " +
      std::to_string(polish_planes.size()) + " LAS planes");
  const auto polish_opt =
      Optimize(model, scene.cameras, "principal", ba_workers, 100, {}, polish_planes,
               inputs.navigation, all_tracks, &motion, inputs.frames, {}, false, inputs.imu_omega);
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
  accepted_axis.unrectified = std::move(model);
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

void UpdateImageVelocity(const Reconstruction& model, const ShutterInputs& inputs,
                         ShutterMotion& motion, uint32_t id) {
  const Eigen::Vector3d v_nav = model.images[id].pose.Rotation() * inputs.kinematics.velocity[id];
  const Eigen::Vector3d r_cam = model.images[id].pose.Rotation() * inputs.kinematics.lever_arm[id];
  const double tau = motion.readout[model.images[id].camera];
  motion.velocity[id] =
      std::abs(tau) > 1e-3 ? Eigen::Vector3d(v_nav - (motion.rotation[id] / tau).cross(r_cam))
                           : v_nav;
}
struct LocalFeatureData {
  std::vector<float> normalized_descriptors;  // N * 256 L2-normalized floats
  std::vector<std::array<uint8_t, 3>> colors;
  int grid_w = 0, grid_h = 0;
  std::vector<std::vector<uint32_t>> cells;
  static constexpr int CellSize = 32;
  void Build(const Camera& cam, FeatureSet&& fs) {
    const size_t n = fs.pixels.size();
    if (fs.dimension != 256 || fs.descriptors.size() != n * 256)
      throw std::runtime_error("Unexpected feature descriptor dimensions during rig rescue");
    normalized_descriptors.resize(n * 256);
    for (size_t i = 0; i < n; ++i) {
      const float* src = fs.descriptors.data() + i * 256;
      float* dst = normalized_descriptors.data() + i * 256;
      double norm2 = 0;
      for (int d = 0; d < 256; ++d)
        norm2 += double(src[d]) * double(src[d]);
      const float inv = norm2 > 1e-12 ? float(1.0 / std::sqrt(norm2)) : 0.f;
      for (int d = 0; d < 256; ++d)
        dst[d] = src[d] * inv;
    }
    colors = std::move(fs.colors);
    grid_w = std::max(1, (cam.width + CellSize - 1) / CellSize);
    grid_h = std::max(1, (cam.height + CellSize - 1) / CellSize);
    cells.assign(size_t(grid_w) * size_t(grid_h), {});
    for (size_t i = 0; i < n; ++i) {
      const int cx = std::clamp(int(fs.pixels[i][0]) / CellSize, 0, grid_w - 1);
      const int cy = std::clamp(int(fs.pixels[i][1]) / CellSize, 0, grid_h - 1);
      cells[size_t(cy) * size_t(grid_w) + size_t(cx)].push_back(uint32_t(i));
    }
  }
  const float* Descriptor(uint32_t f) const { return normalized_descriptors.data() + size_t(f) * 256; }
};
float DescriptorCosine(const float* a, const float* b) {
  float sum = 0;
  for (int d = 0; d < 256; ++d)
    sum += a[d] * b[d];
  return sum;
}
bool ProjectShutter(const Reconstruction& model, const ShutterMotion& motion, uint32_t id,
                    const Eigen::Vector3d& world_xyz, Eigen::Vector2d& out) {
  const auto& image = model.images[id];
  const auto& cam = model.cameras[image.camera];
  const Eigen::Vector3d p_ref = image.pose.Apply(world_xyz);
  if (!p_ref.allFinite())
    return false;
  double s = 0;
  Eigen::Vector2d px = Eigen::Vector2d::Zero();
  for (int it = 0; it < 4; ++it) {
    const Eigen::Vector3d p_exp = motion.Apply(id, image.camera, s, p_ref);
    if (!p_exp.allFinite() || p_exp.z() <= 1e-3)
      return false;
    px = cam.Project(p_exp);
    if (!px.allFinite())
      return false;
    s = motion.Fraction(cam, px);
  }
  const Eigen::Vector3d final_exp = motion.Apply(id, image.camera, s, p_ref);
  if (!final_exp.allFinite() || final_exp.z() <= 1e-3)
    return false;
  px = cam.Project(final_exp);
  if (!px.allFinite() || px.x() < 0 || px.y() < 0 || px.x() >= cam.width || px.y() >= cam.height)
    return false;
  const double dx = (px.x() - cam.params[2]) / cam.params[0],
               dy = (px.y() - cam.params[3]) / cam.params[1];
  const double radius = cam.Radius();
  if (dx * dx + dy * dy > radius * radius)
    return false;
  out = px;
  return true;
}
Eigen::Matrix<double, 3, 4> InstantProjection(const Reconstruction& model,
                                              const ShutterMotion& motion,
                                              const Observation& o) {
  const auto& image = model.images[o.image];
  const auto& cam = model.cameras[image.camera];
  const double s = motion.Fraction(cam, image.pixels[o.feature]);
  const Eigen::Matrix3d a = Readout(motion, o.image, s);
  Eigen::Matrix<double, 3, 4> p;
  p.leftCols<3>() = a * image.pose.Rotation().toRotationMatrix();
  p.col(3) =
      a * (image.pose.Translation() - s * motion.readout[image.camera] * motion.velocity[o.image]);
  return p;
}
bool TriangulateShutterTrack(const Reconstruction& model, const ShutterMotion& motion,
                             const std::vector<Observation>& track, Eigen::Vector3d& point) {
  if (track.size() < 2)
    return false;
  Eigen::MatrixXd design(2 * track.size(), 4);
  for (size_t i = 0; i < track.size(); ++i) {
    const auto& image = model.images[track[i].image];
    const auto& cam = model.cameras[image.camera];
    const Eigen::Vector3d ray = cam.Ray(image.pixels[track[i].feature]);
    const auto p = InstantProjection(model, motion, track[i]);
    design.row(2 * i) = ray.x() * p.row(2) - ray.z() * p.row(0);
    design.row(2 * i + 1) = ray.y() * p.row(2) - ray.z() * p.row(1);
  }
  const Eigen::Vector4d v = design.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
  if (std::abs(v[3]) < 1e-12)
    return false;
  point = v.head<3>() / v[3];
  if (!point.allFinite())
    return false;
  for (const auto& o : track)
    if (Exposed(model, motion, o, point).z() <= 1e-3)
      return false;
  RefineShutterPoint(model, motion, track, point);
  if (!point.allFinite())
    return false;
  for (const auto& o : track) {
    const double z = Exposed(model, motion, o, point).z();
    if (!std::isfinite(z) || z <= 1e-3)
      return false;
  }
  return true;
}
double TrackParallaxCosine(const Reconstruction& model, const std::vector<Observation>& track,
                           const Eigen::Vector3d& point) {
  double best_cos = 1.0;
  std::vector<Eigen::Vector3d> dirs;
  dirs.reserve(track.size());
  for (const auto& o : track)
    dirs.push_back((point - model.images[o.image].pose.Center()).normalized());
  for (size_t i = 0; i < dirs.size(); ++i)
    for (size_t j = i + 1; j < dirs.size(); ++j)
      best_cos = std::min(best_cos, dirs[i].dot(dirs[j]));
  return best_cos;
}
double FixedShutterPositionSigma(const Reconstruction& model, const ShutterMotion& motion,
                                 const ModelPoint& point) {
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  for (const auto& o : point.track) {
    const auto& image = model.images[o.image];
    const auto& cam = model.cameras[image.camera];
    const double s = motion.Fraction(cam, image.pixels[o.feature]);
    const Eigen::Vector3d v = motion.Apply(o.image, image.camera, s, image.pose.Apply(point.xyz));
    if (v.z() <= 1e-6)
      return INFINITY;
    const Eigen::Matrix<double, 2, 3> j = cam.ProjectionJacobian(v) * Readout(motion, o.image, s) *
                                          image.pose.Rotation().toRotationMatrix();
    information.noalias() += j.transpose() * j;
  }
  const auto eigen = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(information).eigenvalues().eval();
  return std::max(.5, point.error) / std::sqrt(std::max(eigen[0], 1e-18));
}
double PointToSegmentDistSq(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                            const Eigen::Vector2d& b) {
  const Eigen::Vector2d d = b - a;
  const double len2 = d.squaredNorm();
  const double t = len2 > 1e-12 ? std::clamp((p - a).dot(d) / len2, 0., 1.) : 0.;
  return (p - (a + t * d)).squaredNorm();
}
// Finds the best descriptor match in image v along the rolling-shutter epipolar ray of feature
// (u, fu). When margin_px > 0 and ratio_ok is populated, applies Lowe ratio or strong cos guard.
bool BestAlongEpipolar(const Reconstruction& model, const ShutterMotion& motion, uint32_t u,
                       uint32_t fu, const LocalFeatureData& fd_u, uint32_t v,
                       const LocalFeatureData& fd_v, std::vector<uint32_t>& seen_stamp,
                       uint32_t& stamp, uint32_t& best_fv, float& best_cos) {
  constexpr int NumDepthSteps = 48;
  constexpr double InvNear = 1.0 / 0.25, InvFar = 1.0 / 50.0, CorridorPx = 10.0;
  const auto& img_u = model.images[u];
  const auto& img_v = model.images[v];
  const auto& cam_u = model.cameras[img_u.camera];
  const Eigen::Vector2d& px_u = img_u.pixels[fu];
  const double s_u = motion.Fraction(cam_u, px_u);
  const Eigen::Vector3d ray_u = cam_u.Ray(px_u);
  const Eigen::Matrix3d a_inv_u = Readout(motion, u, s_u).inverse();
  const Eigen::Matrix3d r_u_t = img_u.pose.Rotation().toRotationMatrix().transpose();
  const Eigen::Vector3d origin_u =
      r_u_t * (s_u * motion.readout[img_u.camera] * motion.velocity[u] - img_u.pose.Translation());
  const Eigen::Vector3d dir_u = r_u_t * (a_inv_u * ray_u);

  std::array<Eigen::Vector2d, NumDepthSteps> poly;
  std::array<bool, NumDepthSteps> poly_ok{};
  int valid_count = 0;
  for (int k = 0; k < NumDepthSteps; ++k) {
    const double d = 1.0 / (InvNear - double(k) * (InvNear - InvFar) / double(NumDepthSteps - 1));
    const Eigen::Vector3d world = origin_u + d * dir_u;
    if (ProjectShutter(model, motion, v, world, poly[size_t(k)])) {
      poly_ok[size_t(k)] = true;
      ++valid_count;
    }
  }
  if (!valid_count)
    return false;

  ++stamp;
  const float* desc_u = fd_u.Descriptor(fu);
  float top1 = -1.f, top2 = -1.f;
  uint32_t top1_id = UINT32_MAX;
  auto visit_cell = [&](int cx, int cy) {
    if (cx < 0 || cx >= fd_v.grid_w || cy < 0 || cy >= fd_v.grid_h)
      return;
    for (const uint32_t fv : fd_v.cells[size_t(cy) * size_t(fd_v.grid_w) + size_t(cx)]) {
      if (seen_stamp[fv] == stamp)
        continue;
      seen_stamp[fv] = stamp;
      const Eigen::Vector2d& pv = img_v.pixels[fv];
      double min_d2 = INFINITY;
      for (int k = 0; k < NumDepthSteps; ++k) {
        if (!poly_ok[size_t(k)])
          continue;
        if (k + 1 < NumDepthSteps && poly_ok[size_t(k + 1)] &&
            (poly[size_t(k + 1)] - poly[size_t(k)]).squaredNorm() <= 400.0 * 400.0) {
          min_d2 = std::min(min_d2, PointToSegmentDistSq(pv, poly[size_t(k)], poly[size_t(k + 1)]));
        } else {
          min_d2 = std::min(min_d2, (pv - poly[size_t(k)]).squaredNorm());
        }
      }
      if (min_d2 > CorridorPx * CorridorPx)
        continue;
      const float c = DescriptorCosine(desc_u, fd_v.Descriptor(fv));
      if (c > top1) {
        top2 = top1;
        top1 = c;
        top1_id = fv;
      } else if (c > top2) {
        top2 = c;
      }
    }
  };
  for (int k = 0; k < NumDepthSteps; ++k) {
    if (!poly_ok[size_t(k)])
      continue;
    const auto visit_around = [&](const Eigen::Vector2d& p) {
      const int cx = int(p.x()) / LocalFeatureData::CellSize,
                cy = int(p.y()) / LocalFeatureData::CellSize;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          visit_cell(cx + dx, cy + dy);
    };
    visit_around(poly[size_t(k)]);
    if (k + 1 < NumDepthSteps && poly_ok[size_t(k + 1)]) {
      const double len = (poly[size_t(k + 1)] - poly[size_t(k)]).norm();
      if (len > 16.0 && len <= 400.0) {
        const int steps = int(std::ceil(len / 16.0));
        for (int s = 1; s < steps; ++s) {
          const double alpha = double(s) / double(steps);
          visit_around((1.0 - alpha) * poly[size_t(k)] + alpha * poly[size_t(k + 1)]);
        }
      }
    }
  }
  if (top1_id == UINT32_MAX || top1 < 0.72f)
    return false;
  if (top2 > -0.5f && top1 < 0.85f) {
    const double d1 = std::sqrt(std::max(0., 2.0 - 2.0 * double(top1))),
                 d2 = std::sqrt(std::max(1e-6, 2.0 - 2.0 * double(top2)));
    if (d1 / d2 > 0.90)
      return false;
  }
  best_fv = top1_id;
  best_cos = top1;
  return true;
}

void RescueRigFrames(const Scene& scene, const fs::path& work, const ShutterInputs& inputs,
                     int workers, const AlignmentLimits& limits, int ba_workers,
                     EvaluatedAxis& accepted_axis) {
  const auto started = Clock::now();
  auto& report = accepted_axis.correction.report;
  Reconstruction model = accepted_axis.unrectified;
  ShutterMotion motion = accepted_axis.correction.motion;
  Pose rig;
  if (inputs.frames.empty() || model.Registered() == model.images.size() ||
      !RigExtrinsic(model, inputs.frames, rig)) {
    report["rig_rescue"] = {{"attempted_images", 0}, {"rescued_images", 0}};
    return;
  }
  struct Candidate {
    size_t frame_index = 0;
    uint32_t missing = 0, known = 0;
    bool missing_is_right = true;
  };
  std::vector<Candidate> candidates;
  const double tl_global = motion.readout[0], tr_global = motion.readout[1];
  for (size_t fi = 0; fi < inputs.frames.size(); ++fi) {
    const auto& f = inputs.frames[fi];
    const bool r0 = model.images[f.left].registered, r1 = model.images[f.right].registered;
    if (r0 == r1)
      continue;
    const uint32_t missing = r0 ? f.right : f.left, known = r0 ? f.left : f.right;
    if ((model.images[known].pose.Center() - inputs.navigation[known]).norm() >= NavigationGate)
      continue;
    const Pose predicted = PredictCompanionPose(model.images[known].pose, rig, f, !r1);
    Eigen::Vector3d pred_rot = Eigen::Vector3d::Zero();
    if (inputs.imu_attitude_verified && inputs.imu_omega[missing].allFinite()) {
      const double tau_missing = motion.readout[model.images[missing].camera];
      pred_rot = tau_missing * (predicted.Rotation() * inputs.imu_omega[missing]);
    } else if (f.synchronous) {
      if (!r1 && std::abs(tl_global) > 1e-6)
        pred_rot = (tr_global / tl_global) * (rig.Rotation() * motion.rotation[known]);
      else if (r1 && std::abs(tr_global) > 1e-6)
        pred_rot = (tl_global / tr_global) * (rig.Rotation().conjugate() * motion.rotation[known]);
    }
    if ((predicted.Center() - inputs.navigation[missing]).norm() >= NavigationGate)
      continue;
    for (int k = 0; k < 3; ++k)
      pred_rot[k] = std::clamp(pred_rot[k], -RotationBound * .98, RotationBound * .98);
    model.images[missing].pose = predicted;
    model.images[missing].registered = true;
    model.images[missing].points.assign(model.images[missing].pixels.size(), NoPoint);
    motion.rotation[missing] = pred_rot;
    UpdateImageVelocity(model, inputs, motion, missing);
    candidates.push_back({fi, missing, known, !r1});
  }
  if (candidates.empty()) {
    report["rig_rescue"] = {{"attempted_images", 0}, {"rescued_images", 0}};
    return;
  }

  const Reconstruction saved_model = accepted_axis.unrectified;
  const ShutterMotion saved_motion = accepted_axis.correction.motion;
  std::set<uint32_t> cand_set, local_set;
  for (const auto& c : candidates) {
    cand_set.insert(c.missing);
    local_set.insert(c.missing);
    local_set.insert(c.known);
  }
  for (uint32_t id = 0; id < uint32_t(model.images.size()); ++id) {
    if (!model.images[id].registered || local_set.count(id))
      continue;
    const double t_id = ExposureTime(scene.images[id]);
    const Eigen::Vector3d c_id = model.images[id].pose.Center();
    for (const auto& cand : candidates) {
      if (std::abs(ExposureTime(scene.images[cand.missing]) - t_id) <= 8.0 ||
          (model.images[cand.missing].pose.Center() - c_id).squaredNorm() <= 36.0) {
        local_set.insert(id);
        break;
      }
    }
  }

  std::unordered_map<uint32_t, LocalFeatureData> features;
  for (const uint32_t id : local_set) {
    auto fs = ReadFeatures(FeaturePath(work, id));
    if (fs.pixels.size() != model.images[id].pixels.size())
      throw std::runtime_error("Feature count mismatch during rig rescue");
    LocalFeatureData lfd;
    lfd.Build(model.cameras[model.images[id].camera], std::move(fs));
    features.emplace(id, std::move(lfd));
  }

  struct RawEdge {
    uint32_t a = 0, fa = 0, b = 0, fb = 0;
    float weight = 0;
  };
  std::vector<RawEdge> raw_edges;
  std::map<std::pair<uint32_t, uint32_t>, std::vector<Match>> archive_pairs;
  {
    const auto matches_path = MatchesPath(work);
    const auto index = IndexMatchArchive(matches_path, false);
    std::ifstream stream(matches_path, std::ios::binary);
    if (!stream)
      throw std::runtime_error("Cannot read matches archive during rig rescue");
    for (const auto& [key, offset] : index) {
      if (!local_set.count(key.first) || !local_set.count(key.second))
        continue;
      if (!cand_set.count(key.first) && !cand_set.count(key.second))
        continue;
      auto ms = ReadMatchRecord(stream, offset, key.first, key.second);
      const auto& fd_a = features.at(key.first);
      const auto& fd_b = features.at(key.second);
      for (const auto& m : ms.matches) {
        const float c = DescriptorCosine(fd_a.Descriptor(m.a), fd_b.Descriptor(m.b));
        raw_edges.push_back({key.first, m.a, key.second, m.b, std::max(m.score, c)});
      }
      archive_pairs.emplace(key, std::move(ms.matches));
    }
  }

  const double min_parallax_cos = std::cos(1.5 * Pi / 180.0);
  // 1) Attach candidates to existing 3D points via guided 2D-3D projection and archive matches.
  struct AttachProposal {
    uint32_t img = 0, feat = 0;
    uint64_t pid = 0;
    float cosine = 0;
    double err = 0;
  };
  std::vector<AttachProposal> attach_proposals;
  std::vector<uint8_t> has_local_obs(model.points.size(), 0);
  for (uint64_t pid = 0; pid < model.points.size(); ++pid)
    for (const auto& o : model.points[pid].track)
      if (local_set.count(o.image) && !cand_set.count(o.image)) {
        has_local_obs[pid] = 1;
        break;
      }

  for (const auto& e : raw_edges) {
    auto check_link = [&](uint32_t u, uint32_t fu, uint32_t v, uint32_t fv) {
      if (!cand_set.count(u) || cand_set.count(v))
        return;
      const uint64_t pid = model.images[v].points[fv];
      if (pid == NoPoint)
        return;
      const auto& pt = model.points[pid];
      if (pt.track.empty() || Exposed(model, motion, {u, fu}, pt.xyz).z() <= 0.15)
        return;
      const double err = ShutterError(model, motion, {u, fu}, pt.xyz);
      if (err > 6.0)
        return;
      const Eigen::Vector3d dir_u = (pt.xyz - model.images[u].pose.Center()).normalized();
      bool parallax_ok = false;
      for (const auto& o : pt.track)
        if (dir_u.dot((pt.xyz - model.images[o.image].pose.Center()).normalized()) <=
            min_parallax_cos) {
          parallax_ok = true;
          break;
        }
      if (!parallax_ok)
        return;
      const float c = DescriptorCosine(features.at(u).Descriptor(fu), features.at(v).Descriptor(fv));
      if (c >= 0.55f)
        attach_proposals.push_back({u, fu, pid, std::max(c, e.weight), err});
    };
    check_link(e.a, e.fa, e.b, e.fb);
    check_link(e.b, e.fb, e.a, e.fa);
  }

  for (const uint32_t u : cand_set) {
    const auto& fd_u = features.at(u);
    const auto& img_u = model.images[u];
    for (uint64_t pid = 0; pid < model.points.size(); ++pid) {
      if (!has_local_obs[pid])
        continue;
      const auto& pt = model.points[pid];
      Eigen::Vector2d px_pred;
      if (!ProjectShutter(model, motion, u, pt.xyz, px_pred))
        continue;
      const double dist = (pt.xyz - img_u.pose.Center()).norm();
      if (dist < 0.15 || dist > 60.0)
        continue;
      const Eigen::Vector3d dir_u = (pt.xyz - img_u.pose.Center()).normalized();
      bool parallax_ok = false;
      for (const auto& o : pt.track)
        if (dir_u.dot((pt.xyz - model.images[o.image].pose.Center()).normalized()) <=
            min_parallax_cos) {
          parallax_ok = true;
          break;
        }
      if (!parallax_ok)
        continue;
      const int cx = int(px_pred.x()) / LocalFeatureData::CellSize,
                cy = int(px_pred.y()) / LocalFeatureData::CellSize;
      float best_cos = -1.f, second_cos = -1.f;
      uint32_t best_fu = UINT32_MAX;
      double best_err = 9.0, second_err = 9.0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int gx = cx + dx, gy = cy + dy;
          if (gx < 0 || gx >= fd_u.grid_w || gy < 0 || gy >= fd_u.grid_h)
            continue;
          for (const uint32_t fu : fd_u.cells[size_t(gy) * size_t(fd_u.grid_w) + size_t(gx)]) {
            const double err = (img_u.pixels[fu] - px_pred).norm();
            if (err > 8.0)
              continue;
            const float* du = fd_u.Descriptor(fu);
            float max_c = -1.f;
            for (const auto& o : pt.track) {
              const auto it = features.find(o.image);
              if (it != features.end())
                max_c = std::max(max_c, DescriptorCosine(du, it->second.Descriptor(o.feature)));
            }
            if (max_c > best_cos) {
              second_cos = best_cos;
              second_err = best_err;
              best_cos = max_c;
              best_fu = fu;
              best_err = err;
            } else if (max_c > second_cos) {
              second_cos = max_c;
              second_err = err;
            }
          }
        }
      if (best_fu == UINT32_MAX || best_cos < 0.72f)
        continue;
      if (second_cos > best_cos - 0.03f && second_err <= best_err + 2.5 &&
          !(best_cos >= 0.88f && best_err <= 2.5))
        continue;
      attach_proposals.push_back({u, best_fu, pid, best_cos, best_err});
    }
  }

  std::sort(attach_proposals.begin(), attach_proposals.end(),
            [](const AttachProposal& a, const AttachProposal& b) {
              if (a.cosine != b.cosine)
                return a.cosine > b.cosine;
              return a.err < b.err;
            });
  std::set<std::pair<uint32_t, uint64_t>> img_has_point;
  std::set<uint64_t> touched_existing;
  size_t attached_obs = 0;
  for (const auto& ap : attach_proposals) {
    if (model.images[ap.img].points[ap.feat] != NoPoint)
      continue;
    if (img_has_point.count({ap.img, ap.pid}))
      continue;
    auto& pt = model.points[ap.pid];
    bool already_in_track = false;
    for (const auto& o : pt.track)
      if (o.image == ap.img) {
        already_in_track = true;
        break;
      }
    if (already_in_track)
      continue;
    pt.track.push_back({ap.img, ap.feat});
    model.images[ap.img].points[ap.feat] = ap.pid;
    img_has_point.emplace(ap.img, ap.pid);
    touched_existing.insert(ap.pid);
    ++attached_obs;
  }

  // 2) Same-lens guided rolling-shutter epipolar-corridor matching on close neighbor pairs.
  std::set<std::pair<uint32_t, uint32_t>> corridor_pairs;
  for (const uint32_t u : cand_set) {
    for (const uint32_t v : local_set) {
      if (u == v || model.images[u].camera != model.images[v].camera)
        continue;
      const double dt = std::abs(ExposureTime(scene.images[u]) - ExposureTime(scene.images[v]));
      const double dist = (model.images[u].pose.Center() - model.images[v].pose.Center()).norm();
      if (dt > 5.0 && dist > 4.5)
        continue;
      const auto p = std::make_pair(std::min(u, v), std::max(u, v));
      if (!corridor_pairs.insert(p).second)
        continue;
      // Skip corridor search if the existing archive matches already have abundant valid links.
      size_t valid_raw = 0;
      std::set<int> cells;
      const auto it = archive_pairs.find(p);
      if (it != archive_pairs.end()) {
        for (const auto& m : it->second) {
          const Observation oa{p.first, m.a}, ob{p.second, m.b};
          Eigen::Vector3d xyz;
          if (TriangulateShutterTrack(model, motion, {oa, ob}, xyz) &&
              ShutterError(model, motion, oa, xyz) <= 4.0 &&
              ShutterError(model, motion, ob, xyz) <= 4.0 &&
              TrackParallaxCosine(model, {oa, ob}, xyz) <= min_parallax_cos) {
            ++valid_raw;
            cells.insert(model.cameras[model.images[u].camera].CoverageCell(
                model.images[u].pixels[p.first == u ? m.a : m.b]));
          }
        }
      }
      if (valid_raw >= 120 && cells.size() >= 16)
        continue;
      const auto& fd_a = features.at(p.first);
      const auto& fd_b = features.at(p.second);
      const size_t na = model.images[p.first].pixels.size(), nb = model.images[p.second].pixels.size();
      std::vector<uint32_t> stamp_b(nb, 0), stamp_a(na, 0);
      uint32_t cur_sb = 0, cur_sa = 0;
      std::vector<int64_t> fwd_best(na, -1);
      std::vector<float> fwd_cos(na, 0.f);
      std::unordered_map<uint32_t, std::pair<uint32_t, float>> rev_cache;
      for (uint32_t fa = 0; fa < uint32_t(na); ++fa) {
        if (model.images[p.first].points[fa] != NoPoint)
          continue;
        uint32_t fb = UINT32_MAX;
        float fc = 0.f;
        if (BestAlongEpipolar(model, motion, p.first, fa, fd_a, p.second, fd_b, stamp_b, cur_sb,
                              fb, fc)) {
          fwd_best[fa] = fb;
          fwd_cos[fa] = fc;
        }
      }
      for (uint32_t fa = 0; fa < uint32_t(na); ++fa) {
        if (fwd_best[fa] < 0)
          continue;
        const uint32_t fb = uint32_t(fwd_best[fa]);
        if (model.images[p.second].points[fb] != NoPoint)
          continue;
        auto rit = rev_cache.find(fb);
        if (rit == rev_cache.end()) {
          uint32_t back_fa = UINT32_MAX;
          float back_cos = 0.f;
          if (!BestAlongEpipolar(model, motion, p.second, fb, fd_b, p.first, fd_a, stamp_a, cur_sa,
                                 back_fa, back_cos))
            back_fa = UINT32_MAX;
          rit = rev_cache.emplace(fb, std::make_pair(back_fa, back_cos)).first;
        }
        if (rit->second.first == fa)
          raw_edges.push_back({p.first, fa, p.second, fb, std::max(fwd_cos[fa], rit->second.second)});
      }
    }
  }

  // 3) Triangulate new multi-view tracks among unattached features.
  std::sort(raw_edges.begin(), raw_edges.end(), [](const RawEdge& a, const RawEdge& b) {
    return a.weight > b.weight;
  });
  std::vector<RawEdge> verified_edges;
  for (const auto& e : raw_edges) {
    if (!cand_set.count(e.a) && !cand_set.count(e.b))
      continue;
    if (model.images[e.a].points[e.fa] != NoPoint || model.images[e.b].points[e.fb] != NoPoint)
      continue;
    const Observation oa{e.a, e.fa}, ob{e.b, e.fb};
    Eigen::Vector3d xyz;
    if (!TriangulateShutterTrack(model, motion, {oa, ob}, xyz))
      continue;
    const double za = Exposed(model, motion, oa, xyz).z(),
                 zb = Exposed(model, motion, ob, xyz).z();
    if (za < 0.15 || za > 50.0 || zb < 0.15 || zb > 50.0)
      continue;
    if (ShutterError(model, motion, oa, xyz) > 4.0 || ShutterError(model, motion, ob, xyz) > 4.0)
      continue;
    if (TrackParallaxCosine(model, {oa, ob}, xyz) > min_parallax_cos)
      continue;
    verified_edges.push_back(e);
  }

  std::vector<uint32_t> local_list(local_set.begin(), local_set.end());
  std::unordered_map<uint32_t, size_t> local_offset;
  size_t total_feats = 0;
  for (const uint32_t id : local_list) {
    local_offset[id] = total_feats;
    total_feats += model.images[id].pixels.size();
  }
  std::vector<uint32_t> node_image(total_feats);
  for (const uint32_t id : local_list)
    std::fill_n(node_image.begin() + std::ptrdiff_t(local_offset[id]),
                model.images[id].pixels.size(), id);
  std::vector<size_t> parent(total_feats), comp_size(total_feats, 1);
  std::iota(parent.begin(), parent.end(), 0);
  std::unordered_map<size_t, std::vector<uint32_t>> comp_images;
  auto uf_root = [&](size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto get_imgs = [&](size_t r) -> std::vector<uint32_t> {
    const auto it = comp_images.find(r);
    return it != comp_images.end() ? it->second : std::vector<uint32_t>{node_image[r]};
  };
  for (const auto& e : verified_edges) {
    const size_t na = local_offset.at(e.a) + e.fa, nb = local_offset.at(e.b) + e.fb;
    size_t ra = uf_root(na), rb = uf_root(nb);
    if (ra == rb)
      continue;
    auto ia = get_imgs(ra), ib = get_imgs(rb);
    bool conflict = false;
    for (const uint32_t id : ia)
      if (std::binary_search(ib.begin(), ib.end(), id)) {
        conflict = true;
        break;
      }
    if (conflict)
      continue;
    if (comp_size[ra] < comp_size[rb]) {
      std::swap(ra, rb);
      std::swap(ia, ib);
    }
    std::vector<uint32_t> merged;
    std::merge(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(merged));
    parent[rb] = ra;
    comp_size[ra] += comp_size[rb];
    comp_images[ra] = std::move(merged);
    comp_images.erase(rb);
  }

  std::unordered_map<size_t, std::vector<Observation>> cand_tracks;
  for (const uint32_t id : local_list) {
    const size_t off = local_offset[id];
    for (uint32_t k = 0; k < uint32_t(model.images[id].pixels.size()); ++k) {
      if (model.images[id].points[k] != NoPoint)
        continue;
      const size_t r = uf_root(off + k);
      if (comp_size[r] >= 2)
        cand_tracks[r].push_back({id, k});
    }
  }
  const uint64_t new_point_start = model.points.size();
  std::set<uint64_t> new_points;
  for (auto& [root_id, track] : cand_tracks) {
    (void)root_id;
    const bool touches_cand = std::any_of(track.begin(), track.end(), [&](const Observation& o) {
      return cand_set.count(o.image) != 0;
    });
    if (!touches_cand)
      continue;
    Eigen::Vector3d xyz;
    if (!TriangulateShutterTrack(model, motion, track, xyz))
      continue;
    if (TrackParallaxCosine(model, track, xyz) > min_parallax_cos)
      continue;
    ModelPoint mp;
    mp.xyz = xyz;
    mp.track = std::move(track);
    mp.error = ShutterPointError(model, motion, mp);
    double max_err = 0;
    bool depth_ok = true;
    std::array<uint64_t, 3> rgb_sum{};
    for (const auto& o : mp.track) {
      const double z = Exposed(model, motion, o, mp.xyz).z();
      if (z < 0.15 || z > 50.0) {
        depth_ok = false;
        break;
      }
      max_err = std::max(max_err, ShutterError(model, motion, o, mp.xyz));
      const auto& col = features.at(o.image).colors[o.feature];
      for (int c = 0; c < 3; ++c)
        rgb_sum[c] += col[c];
    }
    if (!depth_ok || max_err > 3.5 || mp.error > 2.5 ||
        FixedShutterPositionSigma(model, motion, mp) > 0.05)
      continue;
    for (int c = 0; c < 3; ++c)
      mp.color[c] = uint8_t(rgb_sum[c] / mp.track.size());
    const uint64_t pid = model.points.size();
    model.points.push_back(std::move(mp));
    new_points.insert(pid);
  }
  model.RebuildReferences();

  std::set<uint64_t> selected_points = touched_existing;
  selected_points.insert(new_points.begin(), new_points.end());
  auto sanitize_tracks = [&](double max_cand_err, bool strict_new) {
    for (auto it = selected_points.begin(); it != selected_points.end();) {
      const uint64_t pid = *it;
      auto& pt = model.points[pid];
      if (pt.track.empty()) {
        it = selected_points.erase(it);
        continue;
      }
      pt.track.erase(
          std::remove_if(pt.track.begin(), pt.track.end(),
                         [&](const Observation& o) {
                           if (!model.images[o.image].registered)
                             return true;
                           if (model.images[o.image].pose.Apply(pt.xyz).z() <= 1e-3 ||
                               Exposed(model, motion, o, pt.xyz).z() <= 1e-3)
                             return true;
                           if (cand_set.count(o.image) &&
                               ShutterError(model, motion, o, pt.xyz) > max_cand_err)
                             return true;
                           return false;
                         }),
          pt.track.end());
      if (pid >= new_point_start) {
        pt.error = ShutterPointError(model, motion, pt);
        bool keep = pt.track.size() >= 2 && TrackParallaxCosine(model, pt.track, pt.xyz) <= min_parallax_cos;
        if (keep && strict_new)
          keep = pt.error <= 2.5 && FixedShutterPositionSigma(model, motion, pt) <= 0.05;
        if (!keep)
          pt.track.clear();
      } else if (pt.track.size() < 2) {
        pt.track.clear();
      }
      if (pt.track.empty())
        it = selected_points.erase(it);
      else
        ++it;
    }
    model.RebuildReferences();
  };

  sanitize_tracks(6.0, false);
  if (!selected_points.empty()) {
    std::set<uint32_t> active_cands;
    for (const uint32_t u : cand_set)
      if (std::any_of(model.images[u].points.begin(), model.images[u].points.end(),
                      [](uint64_t pid) { return pid != NoPoint; }))
        active_cands.insert(u);
    if (!active_cands.empty()) {
      Optimize(model, scene.cameras, "fixed", ba_workers, 40, active_cands, {}, inputs.navigation,
               selected_points, &motion, inputs.frames, selected_points, true, inputs.imu_omega);
      for (const uint32_t u : active_cands)
        UpdateImageVelocity(model, inputs, motion, u);
      sanitize_tracks(4.0, false);
      if (!selected_points.empty()) {
        Optimize(model, scene.cameras, "fixed", ba_workers, 30, active_cands, {},
                 inputs.navigation, selected_points, &motion, inputs.frames, touched_existing,
                 true, inputs.imu_omega);
        for (const uint32_t u : active_cands)
          UpdateImageVelocity(model, inputs, motion, u);
        sanitize_tracks(3.0, true);
      }
    }
  }

  // 4) Strict per-frame acceptance and rollback of failed candidates.
  std::set<uint32_t> rejected_cands, rescued_cands;
  std::vector<std::string> rescued_names;
  for (const auto& cand : candidates) {
    const uint32_t id = cand.missing;
    const auto& image = model.images[id];
    const auto& cam = model.cameras[image.camera];
    const auto& f = inputs.frames[cand.frame_index];
    std::vector<double> errs;
    std::set<int> cells;
    for (size_t k = 0; k < image.points.size(); ++k)
      if (image.points[k] != NoPoint) {
        const double e = ShutterError(model, motion, {id, uint32_t(k)},
                                      model.points[image.points[k]].xyz);
        if (std::isfinite(e)) {
          errs.push_back(e);
          cells.insert(cam.CoverageCell(image.pixels[k]));
        }
      }
    const Pose expected =
        PredictCompanionPose(model.images[cand.known].pose, rig, f, cand.missing_is_right);
    const double pos_drift = (image.pose.Center() - expected.Center()).norm();
    const double rot_drift = image.pose.Rotation().angularDistance(expected.Rotation());
    const double nav_err = (image.pose.Center() - inputs.navigation[id]).norm();
    const double max_pos_drift = f.synchronous ? 0.03 : 0.05;
    const bool ok = errs.size() >= MinimumImageObservations && cells.size() >= MinimumImageCells &&
                    Quantile(errs, .5) <= limits.median_px && Quantile(errs, .9) <= limits.p90_px &&
                    pos_drift <= max_pos_drift && rot_drift <= 1.5 * Pi / 180.0 &&
                    nav_err < NavigationGate;
    if (ok) {
      rescued_cands.insert(id);
      rescued_names.push_back(scene.images[id].name);
    } else {
      rejected_cands.insert(id);
      model.images[id].registered = false;
      model.images[id].points.assign(model.images[id].pixels.size(), NoPoint);
      motion.rotation[id].setZero();
      motion.velocity[id].setZero();
    }
  }
  if (!rejected_cands.empty())
    sanitize_tracks(3.0, true);
  if (rescued_cands.empty()) {
    accepted_axis.unrectified = saved_model;
    accepted_axis.correction.motion = saved_motion;
    report["rig_rescue"] = {{"attempted_images", candidates.size()},
                            {"rescued_images", 0},
                            {"names", Json::array()},
                            {"added_points", 0},
                            {"added_observations", 0},
                            {"seconds", Seconds(started)}};
    return;
  }

  for (const uint64_t pid : selected_points)
    model.points[pid].error = ShutterPointError(model, motion, model.points[pid]);

  std::vector<double> shift;
  auto rectified = RectifyModel(model, motion, workers, shift);
  const auto quality = rectified.Quality(scene, limits.median_px, limits.p90_px);
  bool consistent = false;
  const auto consistency = ReadoutConsistency(scene, rectified, motion, consistent);
  std::vector<double> rotation, translation;
  const bool bounded = MotionStats(model, motion, rotation, translation);
  if (!quality.at("passed").get<bool>() || !consistent || !bounded) {
    accepted_axis.unrectified = saved_model;
    accepted_axis.correction.motion = saved_motion;
    Log("Rig rescue rejected after global check (" + std::to_string(rescued_cands.size()) +
        " candidate images rolled back)");
    report["rig_rescue"] = {{"attempted_images", candidates.size()},
                            {"rescued_images", 0},
                            {"names", Json::array()},
                            {"added_points", 0},
                            {"added_observations", 0},
                            {"seconds", Seconds(started)}};
    return;
  }

  size_t added_points = 0, added_obs = 0;
  for (uint64_t pid = new_point_start; pid < model.points.size(); ++pid)
    if (!model.points[pid].track.empty()) {
      ++added_points;
      added_obs += model.points[pid].track.size();
    }
  for (const uint64_t pid : touched_existing)
    for (const auto& o : model.points[pid].track)
      if (rescued_cands.count(o.image))
        ++added_obs;

  accepted_axis.unrectified = std::move(model);
  accepted_axis.correction.candidate = std::move(rectified);
  accepted_axis.correction.motion = std::move(motion);
  report["readout_rotation_deg"] = Statistics(rotation);
  report["readout_translation_m"] = Statistics(translation);
  report["readout_consistency"] = consistency;
  report["observation_shift_px"] = Statistics(shift);
  report["candidate_quality"] = quality;
  report["rig_rescue"] = {{"attempted_images", candidates.size()},
                          {"rescued_images", rescued_names.size()},
                          {"names", rescued_names},
                          {"added_points", added_points},
                          {"added_observations", added_obs},
                          {"attached_existing_candidates", attached_obs},
                          {"seconds", Seconds(started)}};
  Log("Rig rescue: recovered " + std::to_string(rescued_names.size()) + "/" +
      std::to_string(candidates.size()) + " rig frames (+" + std::to_string(added_points) +
      " points, +" + std::to_string(added_obs) + " obs)");
}
}  // namespace
ShutterCorrection CorrectRollingShutter(const Scene& scene, const Reconstruction& source,
                                        const fs::path& work, int workers,
                                        const AlignmentLimits& limits, int ba_workers) {
  if (!ba_workers)
    ba_workers = workers;
  const auto started = Clock::now();
  ShutterInputs inputs;
  inputs.source_quality = source.Quality(scene, limits.median_px, limits.p90_px);
  if (!inputs.source_quality.at("passed").get<bool>())
    throw std::runtime_error("Rolling-shutter source reconstruction failed visual acceptance");
  inputs.kinematics = ReadExposureKinematics(scene);
  inputs.imu_boresight_median_deg =
      VerifyImuAttitude(source, inputs.kinematics, inputs.imu_attitude_verified);
  if (inputs.imu_attitude_verified)
    inputs.imu_omega = inputs.kinematics.omega_world;
  inputs.velocity.assign(source.images.size(), Eigen::Vector3d::Zero());
  for (uint32_t i = 0; i < source.images.size(); ++i)
    if (source.images[i].registered) {
      Eigen::Vector3d v_world = inputs.kinematics.velocity[i];
      if (inputs.imu_attitude_verified)
        v_world -= inputs.kinematics.omega_world[i].cross(inputs.kinematics.lever_arm[i]);
      inputs.velocity[i] = source.images[i].pose.Rotation() * v_world;
    }
  inputs.navigation = NavigationPositions(scene);
  Log("Rolling-shutter correction: " + std::to_string(source.Registered()) + " images");
  inputs.planes = LasPlaneConstraints(scene, source, workers);
  // Whole tracks are held out of the camera solve and refit afterwards with cameras fixed.
  for (uint64_t id = 0; id < source.points.size(); ++id) {
    const auto& point = source.points[id];
    if (point.track.size() < 2 || !HasPositiveReferenceDepth(source, point))
      continue;
    if (point.track.size() >= 3 && id % 5 == 0)
      inputs.validation.push_back(id);
    else
      inputs.training.insert(id);
  }
  inputs.frames = BuildRigFrames(scene, source, inputs, inputs.rig_mode);
  Log("Rolling-shutter rig mode: " + inputs.rig_mode + " (" +
      std::to_string(inputs.frames.size()) + " pairs), imu_attitude=" +
      std::string(inputs.imu_attitude_verified ? "verified" : "unavailable"));
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
  if (best.correction.accepted) {
    PolishAcceptedShutter(scene, source, inputs, workers, limits, ba_workers, best);
    RescueRigFrames(scene, work, inputs, workers, limits, ba_workers, best);
  }
  auto& report = best.correction.report;
  Json boresight_json = Json::array();
  for (double deg : inputs.imu_boresight_median_deg)
    boresight_json.push_back(deg >= 0 ? Json(deg) : Json(nullptr));
  report["method"] = "per-image readout rotation and per-lens readout time with the navigation "
                     "velocity, jointly with cameras, points and intrinsics, screened along rows "
                     "and columns";
  report["rig_mode"] = inputs.rig_mode;
  report["imu_attitude"] = {{"available", inputs.kinematics.has_attitude},
                            {"verified", inputs.imu_attitude_verified},
                            {"boresight_median_deg", boresight_json}};
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

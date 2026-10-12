#include <ceres/ceres.h>
#include <colmap/estimators/bundle_adjustment_ceres.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <set>

#include "depth_cost.h"
#include "joint_costs.h"
#include "registration.h"

namespace registration {
namespace {
constexpr double Pi = 3.14159265358979323846;
struct Match {
  colmap::point3D_t id;
  Eigen::Vector3d centre, normal;
  Tile tile;
  double confidence, sigma, plane, weight = 0;
};
struct Constraints {
  std::vector<Match> matches;
  Json summary;
  bool available = false;
  std::string reason;
};
std::pair<Points, std::vector<uint8_t>> VisualPlanes(const Points& points) {
  Cloud cloud{&points};
  Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  tree.buildIndex();
  Points normals[2];
  std::vector<uint8_t> valid[2];
  for (int scale = 0; scale < 2; ++scale) {
    const size_t k = scale ? 32 : 16;
    normals[scale].resize(points.size());
    valid[scale].resize(points.size());
    if (points.size() < k)
      return {Points(points.size(), Eigen::Vector3d::Zero()),
              std::vector<uint8_t>(points.size(), 0)};
    for (size_t i = 0; i < points.size(); ++i) {
      std::array<size_t, 32> ids;
      std::array<double, 32> squared;
      tree.knnSearch(points[i].data(), k, ids.data(), squared.data());
      Eigen::Vector3d centre = Eigen::Vector3d::Zero();
      for (size_t j = 0; j < k; ++j) centre += points[ids[j]];
      centre /= double(k);
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
      for (size_t j = 0; j < k; ++j) {
        const Eigen::Vector3d d = points[ids[j]] - centre;
        covariance += d * d.transpose();
      }
      covariance /= double(k);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solve(covariance);
      if (solve.info() != Eigen::Success)
        throw std::runtime_error("Visual plane eigensolve failed");
      const auto values = solve.eigenvalues().cwiseMax(0.);
      normals[scale][i] = solve.eigenvectors().col(0);
      valid[scale][i] = values[0] / std::max(values.sum(), 1e-12) < .025 &&
                        values[1] > .04 * values[2] &&
                        std::sqrt(values[0]) <= .05 &&
                        squared[k - 1] <= 1.5 * 1.5;
    }
  }
  for (size_t i = 0; i < points.size(); ++i)
    valid[1][i] =
        valid[0][i] && valid[1][i] &&
        std::abs(normals[0][i].dot(normals[1][i])) >= std::cos(15 * Pi / 180);
  return {std::move(normals[1]), std::move(valid[1])};
}
Constraints Correspondences(const colmap::Reconstruction& rec,
                            const colmap::Sim3d& transform,
                            const std::vector<colmap::point3D_t>& ids,
                            const Regions& regions, const Surface& surface,
                            double maximum) {
  Constraints result;
  Points points;
  std::map<colmap::point3D_t, size_t> index;
  for (size_t i = 0; i < ids.size(); ++i) {
    points.push_back(transform * rec.Point3D(ids[i]).xyz);
    index[ids[i]] = i;
  }
  auto [normals, available] = VisualPlanes(points);
  std::vector<Eigen::Matrix3d> information(ids.size(), Eigen::Matrix3d::Zero());
  std::vector<size_t> counts(ids.size(), 0);
  for (const auto& [iid, image] : rec.Images())
    if (image.HasPose()) {
      const auto pose = image.CamFromWorld();
      const Eigen::Matrix3d rotation = pose.rotation().toRotationMatrix();
      const auto& camera = rec.Camera(image.CameraId());
      for (const auto& obs : image.Points2D())
        if (obs.HasPoint3D() && index.count(obs.point3D_id)) {
          const size_t i = index.at(obs.point3D_id);
          const Eigen::Vector3d local = pose * rec.Point3D(obs.point3D_id).xyz;
          if (!local.allFinite() || local.z() <= 0) continue;
          const double step = std::max(local.norm(), .1) * 1e-6;
          Eigen::Matrix<double, 2, 3> jacobian;
          bool valid = true;
          for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d plus = local, minus = local;
            plus[axis] += step;
            minus[axis] -= step;
            const auto a = camera.ImgFromCam(plus),
                       b = camera.ImgFromCam(minus);
            if (!a || !b || !a->allFinite() || !b->allFinite()) {
              valid = false;
              break;
            }
            jacobian.col(axis) = (*a - *b) / (2 * step);
          }
          if (valid) {
            jacobian = (jacobian * rotation /
                        std::max(1., rec.Point3D(obs.point3D_id).error))
                           .eval();
            information[i] += jacobian.transpose() * jacobian;
            ++counts[i];
          }
        }
    }
  const Eigen::Matrix3d linear =
      transform.scale() * transform.rotation().toRotationMatrix();
  size_t ambiguous_count = 0, point_only_count = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solve(information[i]);
    if (solve.info() != Eigen::Success)
      throw std::runtime_error("Point precision eigensolve failed");
    const Eigen::Vector3d values = solve.eigenvalues();
    if (counts[i] < 2 || values[0] <= std::max(values[2] * 1e-8, 1e-12))
      continue;
    const Eigen::Matrix3d covariance =
        linear * solve.eigenvectors() * values.cwiseInverse().asDiagonal() *
        solve.eigenvectors().transpose() * linear.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> precision(covariance);
    if (precision.info() != Eigen::Success)
      throw std::runtime_error("Point covariance eigensolve failed");
    const double std = std::sqrt(std::max(0., precision.eigenvalues()[2]));
    const bool point_only = !available[i] && std <= .1;
    if (std > .3 || (!available[i] && !point_only)) continue;
    std::array<size_t, 8> neighbors;
    std::array<double, 8> squared;
    surface.tree->knnSearch(points[i].data(), 8, neighbors.data(),
                            squared.data());
    std::array<bool, 8> eligible{};
    size_t choice = 8;
    for (size_t j = 0; j < 8; ++j) {
      const auto n = neighbors[j];
      eligible[j] =
          surface.coherent[n] && squared[j] <= maximum * maximum &&
          std::abs((points[i] - surface.centres[n]).dot(surface.normals[n])) <=
              maximum &&
          (!available[i] || std::abs(normals[i].dot(surface.normals[n])) >=
                                std::cos(25 * Pi / 180));
      if (choice == 8 && eligible[j]) choice = j;
    }
    if (choice == 8) continue;
    const auto nearest = neighbors[choice];
    bool ambiguous = false;
    for (size_t j = 0; j < 8; ++j)
      if (eligible[j] &&
          std::abs(surface.normals[neighbors[j]].dot(
              surface.normals[nearest])) >= std::cos(20 * Pi / 180) &&
          std::sqrt(squared[j]) <= std::sqrt(squared[choice]) + .1 &&
          std::abs((surface.centres[neighbors[j]] - surface.centres[nearest])
                       .dot(surface.normals[nearest])) > .05)
        ambiguous = true;
    if (ambiguous) {
      ++ambiguous_count;
      continue;
    }
    const auto normal = surface.normals[nearest];
    const double normal_std =
                     std::sqrt(std::max(0., normal.dot(covariance * normal))),
                 noise = surface.noise[nearest];
    const double confidence =
        std::min(1., std::pow(.02 / std::max(normal_std, 1e-12), 2)) *
        std::min(1., std::pow(.01 / std::max(noise, 1e-12), 2));
    const double sigma =
        std::sqrt(normal_std * normal_std + std::pow(std::max(noise, .01), 2));
    if (point_only) ++point_only_count;
    result.matches.push_back(
        {ids[i], surface.centres[nearest], normal, regions.tiles[i], confidence,
         sigma, (points[i] - surface.centres[nearest]).dot(normal)});
  }
  if (result.matches.size() < 100) {
    result.reason = "Too few reliable LAS correspondences for joint BA";
    return result;
  }
  std::map<Tile, std::map<Tile, size_t>> directions;
  std::vector<Tile> labels;
  for (const auto& match : result.matches) {
    Eigen::Index axis;
    match.normal.cwiseAbs().maxCoeff(&axis);
    const double sign = match.normal[axis] < 0 ? -1. : 1.;
    Tile label;
    for (int j = 0; j < 3; ++j)
      label[j] = int64_t(std::nearbyint(match.normal[j] * sign * 4));
    labels.push_back(label);
    ++directions[label][match.tile];
  }
  std::map<Tile, double> masses;
  double mass_sum = 0.;
  for (const auto& [direction, tiles] : directions) {
    size_t support = 0;
    for (const auto& [tile, n] : tiles) support += n;
    masses[direction] = std::min(1., support / 30.);
    mass_sum += masses[direction];
  }
  Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
  for (const auto& m : result.matches)
    pivot += transform * rec.Point3D(m.id).xyz;
  pivot /= double(result.matches.size());
  Eigen::Matrix<double, 7, 7> info = Eigen::Matrix<double, 7, 7>::Zero();
  std::vector<double> relative, errors;
  for (size_t i = 0; i < result.matches.size(); ++i) {
    auto& m = result.matches[i];
    m.weight =
        masses.at(labels[i]) / mass_sum * m.confidence /
        (directions.at(labels[i]).size() * directions.at(labels[i]).at(m.tile));
    const Eigen::Vector3d point = (transform * rec.Point3D(m.id).xyz) - pivot;
    Eigen::Matrix<double, 7, 1> row;
    row.head<3>() = point.cross(m.normal);
    row.segment<3>(3) = m.normal;
    row[6] = point.dot(m.normal);
    info += m.weight * row * row.transpose();
    relative.push_back(std::abs(m.plane) / m.sigma);
    errors.push_back(std::abs(m.plane));
  }
  const Eigen::Matrix<double, 7, 1> norm =
      info.diagonal().cwiseMax(1e-24).cwiseSqrt();
  const Eigen::Matrix<double, 7, 7> normalized =
      norm.cwiseInverse().asDiagonal() * info *
      norm.cwiseInverse().asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>> solve(normalized);
  if (solve.info() != Eigen::Success)
    throw std::runtime_error("Joint information eigensolve failed");
  int rank = 0;
  for (int j = 0; j < 7; ++j)
    if (solve.eigenvalues()[j] > solve.eigenvalues()[6] * 1e-6) ++rank;
  result.summary = {
      {"fit_points", ids.size()},
      {"matches", result.matches.size()},
      {"precise_point_only_matches", point_only_count},
      {"ambiguous_parallel_faces", ambiguous_count},
      {"similarity_information_rank", rank},
      {"relative_plane_error_sigma", Statistics(relative)},
      {"initial_plane_error", Statistics(errors)},
      {"noise_scope",
       "Conditional visual precision and conservative LAS noise proxies"}};
  if (rank < 7) {
    result.reason = "Joint support cannot constrain all similarity directions";
    return result;
  }
  result.available = true;
  return result;
}
struct Observation {
  colmap::image_t image;
  colmap::point3D_t point;
  Eigen::Vector2d pixel;
};
struct Observations {
  std::vector<Observation> rows;
  std::map<std::string, std::vector<size_t>> groups;
  Observations(const colmap::Reconstruction& rec,
               const colmap::Sim3d& partition, const Regions& regions) {
    auto ids = rec.RegImageIds();
    std::sort(ids.begin(), ids.end());
    for (auto id : ids) {
      const auto& image = rec.Image(id);
      const auto& c = rec.Camera(image.CameraId());
      const double theta = 80 * Pi / 180;
      double radius = theta;
      for (int j = 0; j < 4; ++j)
        radius += theta * c.params[4 + j] * std::pow(theta, 2 * j + 2);
      for (const auto& point : image.Points2D())
        if (point.HasPoint3D()) {
          const size_t i = rows.size();
          rows.push_back({id, point.point3D_id, point.xy});
          const double rho =
              std::hypot((point.xy.x() - c.params[2]) / c.params[0],
                         (point.xy.y() - c.params[3]) / c.params[1]) /
              radius;
          const int zone = rho < .5 ? 0 : rho < .8 ? 1 : 2;
          const Eigen::Vector3d xyz =
              partition * rec.Point3D(point.point3D_id).xyz;
          Tile tile;
          for (int j = 0; j < 3; ++j)
            tile[j] = int64_t(
                std::floor((xyz[j] - regions.origin[j]) / regions.size));
          for (const auto& name :
               {std::string("all"),
                "camera/" + std::to_string(image.CameraId()),
                "image/" + std::to_string(id),
                "camera_zone/" + std::to_string(image.CameraId()) + "," +
                    std::to_string(zone),
                "tile/" + std::to_string(tile[0]) + "," +
                    std::to_string(tile[1]) + "," + std::to_string(tile[2])})
            groups[name].push_back(i);
        }
    }
  }
  std::pair<std::vector<double>, std::vector<double>> Errors(
      const colmap::Reconstruction& rec) const {
    std::vector<double> errors, depths;
    errors.reserve(rows.size());
    depths.reserve(rows.size());
    for (const auto& row : rows) {
      const auto& image = rec.Image(row.image);
      const Eigen::Vector3d local =
          image.CamFromWorld() * rec.Point3D(row.point).xyz;
      const auto pixel = rec.Camera(image.CameraId()).ImgFromCam(local);
      depths.push_back(local.z());
      errors.push_back(local.allFinite() && local.z() > 1e-9 && pixel &&
                               pixel->allFinite()
                           ? (*pixel - row.pixel).norm()
                           : INFINITY);
    }
    return {errors, depths};
  }
  Json Gate(const std::vector<double>& baseline,
            const std::vector<double>& candidate) const {
    Json failures = Json::array();
    for (const auto& [name, ids] : groups) {
      if (name != "all" && ids.size() < 30) continue;
      std::vector<double> a, b;
      for (auto i : ids) {
        a.push_back(baseline[i]);
        b.push_back(candidate[i]);
      }
      std::array<double, 3> tolerance;
      if (name == "all")
        tolerance = {.02, .05, .1};
      else if (name.rfind("image/", 0) == 0)
        tolerance = {.05, .1, .2};
      else if (name.rfind("tile/", 0) == 0)
        tolerance = {.1, .2, .3};
      else
        tolerance = {.03, .08, .15};
      int metric = 0;
      for (double q : {.5, .9, .99}) {
        const double old = Quantile(a, q), next = Quantile(b, q);
        if (!std::isfinite(next) || next - old > tolerance[metric])
          failures.push_back(
              {{"group", name},
               {"quantile", q},
               {"delta_px",
                std::isfinite(next) ? Json(next - old) : Json(nullptr)},
               {"maximum_delta_px", tolerance[metric]}});
        ++metric;
      }
    }
    return {{"passed", failures.empty()}, {"failures", failures}};
  }
};
double Objective(const colmap::Reconstruction& rec,
                 const colmap::Sim3d& transform,
                 const std::vector<double>& errors,
                 const std::map<colmap::image_t, Eigen::Vector3d>& positions,
                 const Constraints& constraints) {
  double cost = 0.;
  for (double e : errors) cost += Huber(e, 2.);
  const double pose_mass = errors.size() * .001 / (3 * positions.size());
  for (const auto& [id, p] : positions)
    cost +=
        pose_mass *
        Huber(((transform * rec.Image(id).ProjectionCenter()) - p).norm() / .5,
              2.);
  for (const auto& m : constraints.matches)
    cost +=
        errors.size() * .005 * m.weight *
        Huber(((transform * rec.Point3D(m.id).xyz) - m.centre).dot(m.normal) /
                  .05,
              2.);
  return cost;
}
Json GeometryGate(const colmap::Reconstruction& base,
                  const colmap::Reconstruction& rec,
                  const colmap::Sim3d& initial,
                  const colmap::Sim3d& transform) {
  double cameras = 0., angles = 0., points = 0.;
  Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
  for (const auto& [id, im] : base.Images()) {
    const auto& other = rec.Image(id);
    cameras = std::max(
        cameras, initial.scale() *
                     (im.ProjectionCenter() - other.ProjectionCenter()).norm());
    const Eigen::Matrix3d delta =
        other.CamFromWorld().rotation().toRotationMatrix() *
        im.CamFromWorld().rotation().toRotationMatrix().transpose();
    angles = std::max(angles, Eigen::AngleAxisd(delta).angle() * 180 / Pi);
    pivot += initial * im.ProjectionCenter();
  }
  pivot /= double(base.NumRegImages());
  for (const auto& [id, p] : base.Points3D())
    points = std::max(points,
                      initial.scale() * (rec.Point3D(id).xyz - p.xyz).norm());
  const auto delta = transform * colmap::Inverse(initial);
  Json checks = {
      {"local_camera_displacement_m", {cameras, .15}},
      {"local_camera_rotation_deg", {angles, .5}},
      {"local_point_displacement_m", {points, .5}},
      {"global_rotation_deg",
       {Eigen::AngleAxisd(delta.rotation()).angle() * 180 / Pi, 1.}},
      {"global_log_scale_change", {std::abs(std::log(delta.scale())), .005}},
      {"global_pivot_displacement_m", {((delta * pivot) - pivot).norm(), .15}}};
  Json failures = Json::array();
  for (const auto& [name, pair] : checks.items())
    if (!std::isfinite(pair[0].get<double>()) ||
        pair[0].get<double>() > pair[1].get<double>() + 1e-10)
      failures.push_back(name);
  return {
      {"passed", failures.empty()}, {"failures", failures}, {"checks", checks}};
}
colmap::Sim3d Solve(colmap::Reconstruction& rec,
                    const colmap::Reconstruction& base,
                    const colmap::Sim3d& transform,
                    const colmap::Sim3d& initial,
                    const std::map<colmap::image_t, Eigen::Vector3d>& positions,
                    const Constraints& constraints, int workers, Json& report) {
  colmap::BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.refine_sensor_from_rig = false;
  options.print_summary = false;
  options.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
  options.ceres->loss_function_scale = 2.;
  auto& solver = options.ceres->solver_options;
  solver.num_threads = workers;
  solver.max_num_iterations = 20;
  solver.initial_trust_region_radius = 10.;
  solver.max_trust_region_radius = 10000.;
  solver.function_tolerance = 1e-7;
  colmap::BundleAdjustmentConfig config;
  for (auto id : rec.RegImageIds()) config.AddImage(id);
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  auto adjuster =
      colmap::CreateDefaultCeresBundleAdjuster(options, config, rec);
  auto& problem = *adjuster->Problem();
  for (const auto& [id, image] : rec.Images())
    if (image.HasPose()) {
      auto& pose = rec.Frame(image.FrameId()).RigFromWorld();
      const bool included = problem.HasParameterBlock(pose.params.data());
      for (const auto& observation : image.Points2D())
        if (observation.HasPoint3D()) {
          auto& point = rec.Point3D(observation.point3D_id);
          problem.AddResidualBlock(
              new geosunpre::PositiveDepthCost((pose * point.xyz).z()), nullptr,
              point.xyz.data(), pose.params.data());
        }
      if (!included) problem.SetParameterBlockConstant(pose.params.data());
    }
  Eigen::Quaterniond quaternion = transform.rotation();
  std::array<double, 4> translation{
      transform.translation().x(), transform.translation().y(),
      transform.translation().z(), std::log(transform.scale())};
  const auto observations = rec.ComputeNumObservations();
  const double pose_mass = observations * .001 / (3 * positions.size());
  auto* pose_loss = new ceres::HuberLoss(std::sqrt(pose_mass) * 2.);
  for (const auto& [id, p] : positions) {
    auto& pose = rec.Frame(rec.Image(id).FrameId()).RigFromWorld();
    problem.AddResidualBlock(
        new geosunpre::Sim3PositionPriorCost({p.x(), p.y(), p.z()},
                                             std::sqrt(pose_mass) / .5),
        pose_loss, pose.params.data(), quaternion.coeffs().data(),
        translation.data());
  }
  for (const auto& m : constraints.matches) {
    const double mass = observations * .005 * m.weight;
    if (mass <= 0) continue;
    problem.AddResidualBlock(
        new geosunpre::Sim3PointPlaneCost(
            {m.centre.x(), m.centre.y(), m.centre.z()},
            {m.normal.x(), m.normal.y(), m.normal.z()}, std::sqrt(mass) / .05),
        new ceres::HuberLoss(std::sqrt(mass) * 2.),
        rec.Point3D(m.id).xyz.data(), quaternion.coeffs().data(),
        translation.data());
  }
  problem.SetManifold(quaternion.coeffs().data(),
                      new ceres::EigenQuaternionManifold());
  for (int j = 0; j < 3; ++j) {
    problem.SetParameterLowerBound(translation.data(), j,
                                   initial.translation()[j] - .3);
    problem.SetParameterUpperBound(translation.data(), j,
                                   initial.translation()[j] + .3);
  }
  problem.SetParameterLowerBound(translation.data(), 3,
                                 std::log(initial.scale()) - .005);
  problem.SetParameterUpperBound(translation.data(), 3,
                                 std::log(initial.scale()) + .005);
  const double bound = .2 / initial.scale();
  for (const auto& [id, p] : rec.Points3D()) {
    auto& point = rec.Point3D(id);
    if (!problem.HasParameterBlock(point.xyz.data()) ||
        problem.IsParameterBlockConstant(point.xyz.data()))
      continue;
    for (int j = 0; j < 3; ++j) {
      problem.SetParameterLowerBound(point.xyz.data(), j,
                                     base.Point3D(id).xyz[j] - bound);
      problem.SetParameterUpperBound(point.xyz.data(), j,
                                     base.Point3D(id).xyz[j] + bound);
    }
  }
  const auto started = Clock::now();
  ceres::Solver::Summary summary;
  ceres::Solve(options.ceres->CreateSolverOptions(config, problem), &problem,
               &summary);
  rec.UpdatePoint3DErrors();
  report = {{"solver", summary.BriefReport()},
            {"solution_usable", summary.IsSolutionUsable()},
            {"elapsed_seconds", Seconds(started)},
            {"position_priors", positions.size()},
            {"las_constraints", constraints.matches.size()},
            {"global_sim3_variable", true},
            {"full_solver_report", summary.FullReport()}};
  return colmap::Sim3d(std::exp(translation[3]), quaternion.normalized(),
                       {translation[0], translation[1], translation[2]});
}
std::pair<colmap::Reconstruction, colmap::Sim3d> Interpolate(
    const colmap::Reconstruction& before, const colmap::Reconstruction& after,
    const colmap::Sim3d& first, const colmap::Sim3d& second, double fraction) {
  auto rec = before;
  for (const auto& [id, f] : before.Frames()) {
    auto& frame = rec.Frame(id);
    const auto& a = f.RigFromWorld();
    const auto& b = after.Frame(id).RigFromWorld();
    const Eigen::Quaterniond rotation =
        Eigen::Quaterniond(a.rotation())
            .slerp(fraction, Eigen::Quaterniond(b.rotation()));
    const Eigen::Vector3d ca = -(a.rotation().inverse() * a.translation()),
                          cb = -(b.rotation().inverse() * b.translation()),
                          centre = (1 - fraction) * ca + fraction * cb;
    frame.SetRigFromWorld(colmap::Rigid3d(rotation, -(rotation * centre)));
  }
  for (const auto& [id, p] : before.Points3D())
    rec.Point3D(id).xyz =
        (1 - fraction) * p.xyz + fraction * after.Point3D(id).xyz;
  rec.UpdatePoint3DErrors();
  const auto rotation =
      Eigen::Quaterniond(first.rotation())
          .slerp(fraction, Eigen::Quaterniond(second.rotation()));
  return {std::move(rec),
          colmap::Sim3d(std::exp((1 - fraction) * std::log(first.scale()) +
                                 fraction * std::log(second.scale())),
                        rotation,
                        (1 - fraction) * first.translation() +
                            fraction * second.translation())};
}
}  // namespace
JointResult RefineJoint(
    const colmap::Reconstruction& base, const colmap::Sim3d& initial,
    const colmap::Sim3d& partition_transform,
    const std::map<colmap::image_t, Eigen::Vector3d>& positions,
    const Points& target, const std::vector<colmap::point3D_t>& ids,
    const Regions& regions, double voxel, double maximum, int workers) {
  JointResult result;
  result.transform = initial;
  auto surface = CoherentSurfaces(target, voxel, workers);
  auto constraints =
      Correspondences(base, initial, ids, regions, *surface, maximum);
  if (!constraints.available) {
    result.report = {{"status", "constraints_unavailable"},
                     {"reason", constraints.reason},
                     {"diagnostic", constraints.summary}};
    return result;
  }
  const auto& relative = constraints.summary.at("relative_plane_error_sigma");
  if (relative.at("median").get<double>() <= 1. &&
      relative.at("p90").get<double>() <= 3.) {
    result.report = {
        {"status", "constraints_unavailable"},
        {"reason",
         "Reliable surfaces already fit within relative noise proxies"},
        {"diagnostic", constraints.summary}};
    return result;
  }
  Observations observations(base, partition_transform, regions);
  const auto [baseline, base_depth] = observations.Errors(base);
  auto current = base;
  auto transform = initial;
  Json steps = Json::array();
  size_t updates = 0;
  for (int iteration = 0; iteration < 2; ++iteration) {
    if (iteration)
      constraints =
          Correspondences(current, transform, ids, regions, *surface, maximum);
    if (!constraints.available) {
      steps.push_back({{"round", iteration + 1},
                       {"status", "constraints_unavailable"},
                       {"reason", constraints.reason}});
      break;
    }
    const auto [errors, depths] = observations.Errors(current);
    const double old_cost =
        Objective(current, transform, errors, positions, constraints);
    auto proposal = current;
    Json solver;
    std::cerr << "Native joint BA round " << iteration + 1 << ": "
              << constraints.matches.size() << " LAS constraints\n";
    const auto proposed = Solve(proposal, base, transform, initial, positions,
                                constraints, workers, solver);
    Json trials = Json::array();
    bool accepted = false;
    for (double fraction : {1., .5, .25, .125, .0625, .03125}) {
      auto [candidate, matrix] =
          fraction == 1.
              ? std::pair<colmap::Reconstruction, colmap::Sim3d>{proposal,
                                                                 proposed}
              : Interpolate(current, proposal, transform, proposed, fraction);
      const auto [new_errors, new_depth] = observations.Errors(candidate);
      const auto visual = observations.Gate(baseline, new_errors),
                 physical = GeometryGate(base, candidate, initial, matrix);
      bool depth = true;
      for (size_t i = 0; i < depths.size(); ++i)
        if (!std::isfinite(new_depth[i]) || new_depth[i] < .05 * depths[i])
          depth = false;
      const double cost =
          Objective(candidate, matrix, new_errors, positions, constraints);
      const bool descent = std::isfinite(cost) && cost <= old_cost + 1e-6;
      const bool passed = solver.at("solution_usable").get<bool>() &&
                          visual.at("passed").get<bool>() &&
                          physical.at("passed").get<bool>() && depth && descent;
      trials.push_back({{"fraction", fraction},
                        {"passed", passed},
                        {"image_gate", visual},
                        {"geometry_gate", physical},
                        {"depth_guard_passed", depth},
                        {"mixed_objective",
                         std::isfinite(cost) ? Json(cost) : Json(nullptr)},
                        {"descent", descent}});
      if (passed) {
        current = std::move(candidate);
        transform = matrix;
        accepted = true;
        ++updates;
        break;
      }
    }
    steps.push_back({{"round", iteration + 1},
                     {"accepted_update", accepted},
                     {"solver", solver},
                     {"correspondences", constraints.summary},
                     {"mixed_objective_before", old_cost},
                     {"backtracking", trials}});
    if (!accepted) break;
  }
  const auto [errors, depths] = observations.Errors(current);
  const auto visual = observations.Gate(baseline, errors),
             physical = GeometryGate(base, current, initial, transform);
  const bool accepted = updates && visual.at("passed").get<bool>() &&
                        physical.at("passed").get<bool>() &&
                        std::all_of(depths.begin(), depths.end(), [](double x) {
                          return std::isfinite(x) && x > 1e-9;
                        });
  result.report = {
      {"status", "computed"},        {"accepted_for_delivery", accepted},
      {"accepted_updates", updates}, {"steps", steps},
      {"input_image_gate", visual},  {"geometry_gate", physical},
      {"fixed_intrinsics", true}};
  if (accepted) {
    result.model = std::make_unique<colmap::Reconstruction>(std::move(current));
    result.transform = transform;
  }
  return result;
}
}  // namespace registration

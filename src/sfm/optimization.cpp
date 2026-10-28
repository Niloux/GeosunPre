#include "sfm/optimization.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/product_manifold.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace geosun {
namespace {
template <class T>
void Center(const T* pose, T* out) {
  const T q[] = {pose[3], -pose[0], -pose[1], -pose[2]}, t[] = {-pose[4], -pose[5], -pose[6]};
  ceres::QuaternionRotatePoint(q, t, out);
}
Eigen::Matrix<double, 3, 4> RotationJacobian(const Eigen::Quaterniond& unit,
                                             const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& rotated, double norm) {
  const Eigen::Vector3d u = unit.vec();
  Eigen::Matrix3d cross;
  cross << 0, -point.z(), point.y(), point.z(), 0, -point.x(), -point.y(), point.x(), 0;
  Eigen::Matrix<double, 3, 4> j;
  j.leftCols<3>() = 2 * (u.dot(point) * Eigen::Matrix3d::Identity() + u * point.transpose() -
                         point * u.transpose() - unit.w() * cross);
  j.col(3) = 2 * (unit.w() * point + u.cross(point));
  // QuaternionRotatePoint normalizes the ambient quaternion. Account for that
  // normalization before Ceres applies the unit-quaternion manifold.
  j -= 2 * rotated * unit.coeffs().transpose();
  return j / norm;
}
void CameraJacobian(const Camera& camera, const Eigen::Vector3d& v, double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 2, 8, Eigen::RowMajor>> camera_j(jacobian);
  camera_j.setZero();
  const auto& cam = camera.params;
  const double r = std::hypot(v.x(), v.y()), theta = std::atan2(r, v.z()), t2 = theta * theta;
  const double d =
      r < 1e-15 ? 1 / v.z()
                : theta * (1 + t2 * (cam[4] + t2 * (cam[5] + t2 * (cam[6] + t2 * cam[7])))) / r;
  camera_j(0, 0) = v.x() * d;
  camera_j(1, 1) = v.y() * d;
  camera_j(0, 2) = camera_j(1, 3) = 1;
  double power = theta * t2;
  for (int k = 4; k < 8; ++k) {
    const double radial = r < 1e-15 ? 0 : power / r;
    camera_j(0, k) = cam[0] * v.x() * radial;
    camera_j(1, k) = cam[1] * v.y() * radial;
    power *= t2;
  }
}
struct Reprojection : ceres::SizedCostFunction<2, 7, 3, 8> {
  Eigen::Vector2d pixel;
  explicit Reprojection(const Eigen::Vector2d& measurement) : pixel(measurement) {}
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const auto* pose = parameters[0];
    const Eigen::Map<const Eigen::Vector3d> point(parameters[1]);
    Eigen::Quaterniond rotation(pose[3], pose[0], pose[1], pose[2]);
    const double norm = rotation.norm();
    rotation.coeffs() /= norm;
    const Eigen::Vector3d rotated = rotation * point;
    const Eigen::Vector3d v = rotated + Eigen::Map<const Eigen::Vector3d>(pose + 4);
    Camera camera;
    std::copy_n(parameters[2], 8, camera.params.begin());
    Eigen::Map<Eigen::Vector2d> output(residual);
    output = camera.Project(v) - pixel;
    if (!jacobians)
      return true;
    const auto j = camera.ProjectionJacobian(v);
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> pose_j(jacobians[0]);
      pose_j.leftCols<4>() = j * RotationJacobian(rotation, point, rotated, norm);
      pose_j.rightCols<3>() = j;
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[1]);
      point_j = j * rotation.toRotationMatrix();
    }
    if (jacobians[2])
      CameraJacobian(camera, v, jacobians[2]);
    return true;
  }
};
// Reprojection of an observation exposed at readout fraction s; see ShutterMotion. The extra
// blocks are the image's readout rotation and its lens readout time.
struct ShutterReprojection : ceres::SizedCostFunction<2, 7, 3, 8, 3, 1> {
  Eigen::Vector2d pixel;
  Eigen::Vector3d velocity;
  double s;
  ShutterReprojection(const Eigen::Vector2d& measurement, const Eigen::Vector3d& camera_velocity,
                      double fraction)
      : pixel(measurement), velocity(camera_velocity), s(fraction) {}
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const auto* pose = parameters[0];
    const Eigen::Map<const Eigen::Vector3d> point(parameters[1]), w(parameters[3]);
    const double readout = parameters[4][0];
    Eigen::Quaterniond rotation(pose[3], pose[0], pose[1], pose[2]);
    const double norm = rotation.norm();
    rotation.coeffs() /= norm;
    const Eigen::Vector3d rotated = rotation * point;
    const Eigen::Vector3d b =
        rotated + Eigen::Map<const Eigen::Vector3d>(pose + 4) - s * readout * velocity;
    const Eigen::Vector3d v = b + s * w.cross(b);
    Camera camera;
    std::copy_n(parameters[2], 8, camera.params.begin());
    Eigen::Map<Eigen::Vector2d> output(residual);
    output = camera.Project(v) - pixel;
    if (!jacobians)
      return true;
    Eigen::Matrix3d a;
    a << 1, -s * w.z(), s * w.y(), s * w.z(), 1, -s * w.x(), -s * w.y(), s * w.x(), 1;
    const auto j = camera.ProjectionJacobian(v);
    const Eigen::Matrix<double, 2, 3> ja = j * a;
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> pose_j(jacobians[0]);
      pose_j.leftCols<4>() = ja * RotationJacobian(rotation, point, rotated, norm);
      pose_j.rightCols<3>() = ja;
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[1]);
      point_j = ja * rotation.toRotationMatrix();
    }
    if (jacobians[2])
      CameraJacobian(camera, v, jacobians[2]);
    if (jacobians[3]) {
      Eigen::Matrix3d cross;
      cross << 0, -b.z(), b.y(), b.z(), 0, -b.x(), -b.y(), b.x(), 0;
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> rotation_j(jacobians[3]);
      rotation_j = -s * j * cross;
    }
    if (jacobians[4]) {
      Eigen::Map<Eigen::Vector2d> readout_j(jacobians[4]);
      readout_j = -s * ja * velocity;
    }
    return true;
  }
};
struct Depth : ceres::SizedCostFunction<1, 7, 3> {
  double minimum, gain;
  Depth(double minimum_depth, double weight) : minimum(minimum_depth), gain(weight) {}
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const auto* pose = parameters[0];
    const Eigen::Map<const Eigen::Vector3d> point(parameters[1]);
    Eigen::Quaterniond rotation(pose[3], pose[0], pose[1], pose[2]);
    const double norm = rotation.norm();
    rotation.coeffs() /= norm;
    const Eigen::Vector3d rotated = rotation * point;
    const bool active = rotated.z() + pose[6] < minimum;
    residual[0] = active ? gain * (minimum - rotated.z() - pose[6]) : 0;
    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> pose_j(jacobians[0]);
        pose_j.setZero();
        if (active) {
          pose_j.leftCols<4>() = -gain * RotationJacobian(rotation, point, rotated, norm).row(2);
          pose_j[6] = -gain;
        }
      }
      if (jacobians[1]) {
        Eigen::Map<Eigen::RowVector3d> point_j(jacobians[1]);
        point_j.setZero();
        if (active)
          point_j = -gain * rotation.toRotationMatrix().row(2);
      }
    }
    return true;
  }
};
struct Baseline {
  double length;
  template <class T>
  bool operator()(const T* a, const T* b, T* r) const {
    T ca[3], cb[3];
    Center(a, ca);
    Center(b, cb);
    T d = T(0);
    for (int i = 0; i < 3; ++i)
      d += (ca[i] - cb[i]) * (ca[i] - cb[i]);
    r[0] = T(1000) * (ceres::sqrt(d + T(1e-24)) / T(length) - T(1));
    return true;
  }
};
struct Plane {
  Eigen::Vector3d center, normal;
  double sigma;
  template <class T>
  bool operator()(const T* point, T* residual) const {
    residual[0] = T(0);
    for (int i = 0; i < 3; ++i)
      residual[0] += T(normal[i] / sigma) * (point[i] - T(center[i]));
    return true;
  }
};
struct Position {
  Eigen::Vector3d target;
  template <class T>
  bool operator()(const T* pose, T* r) const {
    T c[3];
    Center(pose, c);
    for (int i = 0; i < 3; ++i)
      r[i] = (c[i] - T(target[i])) / T(NavigationSigma);
    return true;
  }
};
// A synchronous frame: the R pose is the rig extrinsic applied to the L pose. Residuals are the
// rotation (rad) and translation (m, R camera frame) differences over their sigmas.
struct RigLink {
  template <class T>
  bool operator()(const T* left, const T* right, const T* rig, T* r) const {
    const T ql[] = {left[3], left[0], left[1], left[2]},
            qr[] = {right[3], right[0], right[1], right[2]},
            qe[] = {rig[3], rig[0], rig[1], rig[2]};
    T predicted[4], difference[4], moved[3];
    ceres::QuaternionProduct(qe, ql, predicted);
    const T inverse[] = {predicted[0], -predicted[1], -predicted[2], -predicted[3]};
    ceres::QuaternionProduct(qr, inverse, difference);
    const T sign = difference[0] < T(0) ? T(-2) : T(2);
    ceres::UnitQuaternionRotatePoint(qe, left + 4, moved);
    for (int i = 0; i < 3; ++i) {
      r[i] = sign * difference[i + 1] / T(RigRotationSigma);
      r[i + 3] = (right[4 + i] - moved[i] - rig[4 + i]) / T(RigPositionSigma);
    }
    return true;
  }
};
class Progress : public ceres::IterationCallback {
  Clock::time_point last = Clock::now();

 public:
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
    if (Seconds(last) >= 25) {
      Log("BA iteration " + std::to_string(s.iteration) + ", cost " + std::to_string(s.cost));
      last = Clock::now();
    }
    return ceres::SOLVER_CONTINUE;
  }
};
}  // namespace
double PointCost(const Reconstruction& model, const std::vector<Observation>& track,
                 const Eigen::Vector3d& point) {
  double cost = 0;
  for (const auto& o : track) {
    const auto& image = model.images[o.image];
    const Eigen::Vector3d v = image.pose.Apply(point);
    if (v.z() <= 0)
      return INFINITY;
    const double error = (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm();
    cost += TrackCost(error);
  }
  return cost;
}
void RefinePoint(const Reconstruction& model, const std::vector<Observation>& track,
                 Eigen::Vector3d& point) {
  RefineTrackPoint(
      track, point, [&](const Eigen::Vector3d& p) { return PointCost(model, track, p); },
      [&](const Observation& o, const Eigen::Vector3d& p, Eigen::Vector2d& residual,
          Eigen::Matrix<double, 2, 3>& j) {
        const auto& image = model.images[o.image];
        const auto& camera = model.cameras[image.camera];
        const Eigen::Vector3d v = image.pose.Apply(p);
        residual = camera.Project(v) - image.pixels[o.feature];
        j = camera.ProjectionJacobian(v) * image.pose.Rotation().toRotationMatrix();
      },
      "Nonfinite triangulation refinement");
}
void RefinePose(const Camera& camera, Pose& pose, const std::vector<Eigen::Vector3d>& points,
                const std::vector<Eigen::Vector2d>& pixels) {
  auto calibration = camera.params;
  auto fixed_points = points;
  ceres::Problem problem;
  auto* loss = new ceres::HuberLoss(2.);
  for (size_t i = 0; i < points.size(); ++i) {
    problem.AddResidualBlock(new Reprojection(pixels[i]), loss, pose.params.data(),
                             fixed_points[i].data(), calibration.data());
    problem.SetParameterBlockConstant(fixed_points[i].data());
  }
  problem.SetParameterBlockConstant(calibration.data());
  problem.SetManifold(
      pose.params.data(),
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>());
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 40;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable())
    throw std::runtime_error("Pose refinement failed: " + summary.BriefReport());
}
bool RigExtrinsic(const Reconstruction& m, const std::vector<std::array<uint32_t, 2>>& frames,
                  Pose& rig) {
  std::vector<Eigen::Quaterniond> rotations;
  std::vector<Eigen::Vector3d> translations;
  for (const auto& f : frames) {
    const auto& left = m.images[f[0]];
    const auto& right = m.images[f[1]];
    if (!left.registered || !right.registered)
      continue;
    const Eigen::Quaterniond q = right.pose.Rotation() * left.pose.Rotation().conjugate();
    rotations.push_back(q);
    translations.push_back(right.pose.Translation() - q * left.pose.Translation());
  }
  if (rotations.size() < 4)
    return false;
  // Start from the per-axis median translation and the full mean, then average the frames
  // within 1 degree and 5 cm of the estimate: a wrong registration does not move it.
  Eigen::Vector3d t;
  for (int k = 0; k < 3; ++k) {
    std::vector<double> values;
    for (const auto& x : translations)
      values.push_back(x[k]);
    t[k] = Quantile(values, .5);
  }
  std::vector<uint8_t> inlier(rotations.size(), 1);
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  for (int pass = 0; pass < 3; ++pass) {
    Eigen::Matrix4d sum = Eigen::Matrix4d::Zero();
    Eigen::Vector3d total = Eigen::Vector3d::Zero();
    size_t count = 0;
    for (size_t i = 0; i < rotations.size(); ++i)
      if (inlier[i]) {
        sum += rotations[i].coeffs() * rotations[i].coeffs().transpose();
        total += translations[i];
        ++count;
      }
    if (count < 4)
      return false;
    const Eigen::Vector4d mean =
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(sum).eigenvectors().col(3);
    q = Eigen::Quaterniond(mean[3], mean[0], mean[1], mean[2]).normalized();
    if (pass)
      t = total / double(count);
    for (size_t i = 0; i < rotations.size(); ++i)
      inlier[i] =
          rotations[i].angularDistance(q) <= Pi / 180 && (translations[i] - t).norm() <= .05;
  }
  rig.Set(q, t);
  return true;
}
Json Optimize(Reconstruction& m, const std::array<Camera, 2>& initial, const std::string& stage,
              int workers, int iterations, const std::set<uint32_t>& variable_images,
              const std::vector<PlaneConstraint>& planes, const std::vector<Eigen::Vector3d>& nav,
              const std::set<uint64_t>& selected_points,
              const std::vector<std::vector<uint8_t>>& precise_observations,
              const Reconstruction* reference, ShutterMotion* shutter,
              const std::vector<std::array<uint32_t, 2>>& frames) {
  const auto started = Clock::now();
  ceres::Problem problem;
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  auto* loss = new ceres::HuberLoss(2.);
  std::set<uint32_t> used_images, used_cameras;
  size_t observations = 0;
  size_t precise_count = 0, reference_count = 0;
  const bool metric = !nav.empty();
  size_t variable_count = 0;
  for (auto& p : m.points)
    if (p.track.size() >= 2) {
      if (!selected_points.empty() && !selected_points.count(uint64_t(&p - m.points.data())))
        continue;
      if (!variable_images.empty() &&
          std::none_of(p.track.begin(), p.track.end(),
                       [&](const auto& o) { return variable_images.count(o.image) != 0; }))
        continue;
      for (const auto& o : p.track) {
        ++observations;
        auto& image = m.images[o.image];
        const double depth = image.pose.Apply(p.xyz).z();
        if (depth <= 0)
          throw std::runtime_error("BA requires positive initial depth");
        const bool precise = !precise_observations.empty() && precise_observations[o.image][o.feature];
        precise_count += precise;
        Eigen::Vector2d pixel = image.pixels[o.feature];
        if (reference && !precise) {
          // Refine relative to the accepted geometry instead of fitting the
          // detector noise again on a different subset of its observations.
          const auto& original = reference->images[o.image];
          pixel = reference->cameras[original.camera].Project(
              original.pose.Apply(reference->points[&p - m.points.data()].xyz));
          ++reference_count;
        }
        if (shutter)
          problem.AddResidualBlock(
              new ShutterReprojection(pixel, shutter->velocity[o.image],
                                      shutter->Fraction(m.cameras[image.camera], pixel)),
              loss, image.pose.params.data(), p.xyz.data(), m.cameras[image.camera].params.data(),
              shutter->rotation[o.image].data(), &shutter->readout[image.camera]);
        else
          problem.AddResidualBlock(new Reprojection(pixel), loss, image.pose.params.data(),
                                   p.xyz.data(), m.cameras[image.camera].params.data());
        problem.AddResidualBlock(new Depth(.05 * depth, 1000 / depth), nullptr,
                                 image.pose.params.data(), p.xyz.data());
        used_images.insert(o.image);
        used_cameras.insert(image.camera);
      }
      ordering->AddElementToGroup(p.xyz.data(), 0);
    }
  const double observation_build_seconds = Seconds(started);
  if (used_images.size() < 2)
    throw std::runtime_error("Insufficient BA observations");
  for (uint32_t id : used_images) {
    auto* pose = m.images[id].pose.params.data();
    problem.SetManifold(
        pose,
        new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>());
    ordering->AddElementToGroup(pose, 1);
    if ((!metric && id == m.gauge_image) ||
        (!variable_images.empty() && !variable_images.count(id)))
      problem.SetParameterBlockConstant(pose);
    else
      ++variable_count;
  }
  if (!metric && used_images.count(m.gauge_image) && used_images.count(m.scale_image))
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Baseline, 1, 7, 7>(new Baseline{m.gauge_baseline}), nullptr,
        m.images[m.gauge_image].pose.params.data(), m.images[m.scale_image].pose.params.data());
  for (uint32_t cid : used_cameras) {
    auto* params = m.cameras[cid].params.data();
    ordering->AddElementToGroup(params, 1);
    if (stage == "fixed")
      problem.SetParameterBlockConstant(params);
    else {
      std::vector<int> fixed;
      if (stage == "focal")
        fixed = {2, 3, 6, 7};
      else if (stage == "radial")
        fixed = {2, 3};
      else if (stage != "principal")
        throw std::runtime_error("Unknown calibration stage: " + stage);
      if (!fixed.empty())
        problem.SetManifold(params, new ceres::SubsetManifold(8, fixed));
      const auto delta = initial[cid].CalibrationBounds();
      for (int i = 0; i < 8; ++i) {
        problem.SetParameterLowerBound(params, i, initial[cid].params[i] - delta[i]);
        problem.SetParameterUpperBound(params, i, initial[cid].params[i] + delta[i]);
      }
    }
  }
  if (shutter) {
    // At most ~3 degrees of rotation and 50 ms per readout; a solution at a bound is reported.
    for (uint32_t id : used_images) {
      auto* rotation = shutter->rotation[id].data();
      ordering->AddElementToGroup(rotation, 1);
      for (int i = 0; i < 3; ++i) {
        problem.SetParameterLowerBound(rotation, i, -.05);
        problem.SetParameterUpperBound(rotation, i, .05);
      }
    }
    for (uint32_t cid : used_cameras) {
      ordering->AddElementToGroup(&shutter->readout[cid], 1);
      problem.SetParameterLowerBound(&shutter->readout[cid], 0, -.05);
      problem.SetParameterUpperBound(&shutter->readout[cid], 0, .05);
    }
  }
  for (const auto& p : planes)
    if (problem.HasParameterBlock(m.points.at(p.point).xyz.data()))
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Plane, 1, 3>(new Plane{p.center, p.normal, p.sigma}),
          new ceres::HuberLoss(2), m.points[p.point].xyz.data());
  if (!nav.empty())
    for (uint32_t id : used_images)
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Position, 3, 7>(new Position{nav.at(id)}),
          new ceres::HuberLoss(2), m.images[id].pose.params.data());
  Pose rig;
  size_t rig_frames = 0;
  if (!frames.empty() && RigExtrinsic(m, frames, rig))
    for (const auto& f : frames)
      if (used_images.count(f[0]) && used_images.count(f[1])) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RigLink, 6, 7, 7, 7>(new RigLink),
                                 new ceres::HuberLoss(2), m.images[f[0]].pose.params.data(),
                                 m.images[f[1]].pose.params.data(), rig.params.data());
        if (!rig_frames++) {
          problem.SetManifold(rig.params.data(),
                              new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                                         ceres::EuclideanManifold<3>>());
          ordering->AddElementToGroup(rig.params.data(), 1);
        }
      }
  ceres::Solver::Options options;
  options.num_threads = workers;
  options.max_num_iterations = iterations;
  options.logging_type = ceres::SILENT;
  const bool iterative = variable_images.empty() && used_images.size() >= 64;
  options.linear_solver_type = iterative ? ceres::ITERATIVE_SCHUR : ceres::SPARSE_SCHUR;
  if (iterative)
    options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.use_explicit_schur_complement = iterative && used_images.size() <= 1024;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.linear_solver_ordering = ordering;
  // Below 1e-6 the final refinements only creep: in the shutter BA the last half of the
  // iterations moved camera centres by under 5 um and changed held-out error by 3e-5 px.
  options.function_tolerance = iterations <= 100 && planes.empty() ? 1e-5 : 1e-6;
  Progress progress;
  options.callbacks.push_back(&progress);
  ceres::Solver::Summary summary;
  Log("BA " + stage + ": " + std::to_string(used_images.size()) + " images, " +
      std::to_string(observations) + " observations");
  const double build_seconds = Seconds(started);
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable())
    throw std::runtime_error("BA failed: " + summary.BriefReport());
  if (metric)
    m.gauge_baseline =
        (m.images[m.gauge_image].pose.Center() - m.images[m.scale_image].pose.Center()).norm();
  for (const auto& c : m.cameras)
    c.Radius();
  Log(summary.BriefReport());
  const double elapsed = Seconds(started);
  int linear_iterations = 0, maximum_linear_iterations = 0, linear_limit_hits = 0;
  Json iteration_profile = Json::array();
  for (const auto& iteration : summary.iterations) {
    linear_iterations += iteration.linear_solver_iterations;
    maximum_linear_iterations =
        std::max(maximum_linear_iterations, iteration.linear_solver_iterations);
    linear_limit_hits += iteration.linear_solver_iterations >= options.max_linear_solver_iterations;
    iteration_profile.push_back({{"iteration", iteration.iteration},
                                 {"seconds", iteration.iteration_time_in_seconds},
                                 {"step_solver_seconds", iteration.step_solver_time_in_seconds},
                                 {"linear_iterations", iteration.linear_solver_iterations},
                                 {"cost", iteration.cost},
                                 {"cost_change", iteration.cost_change},
                                 {"gradient_max_norm", iteration.gradient_max_norm},
                                 {"trust_region_radius", iteration.trust_region_radius},
                                 {"valid", iteration.step_is_valid},
                                 {"accepted", iteration.step_is_successful}});
  }
  if (elapsed >= 5)
    Log("BA timing: build " + std::to_string(build_seconds) + " s, linear " +
        std::to_string(summary.linear_solver_time_in_seconds) + " s, Jacobian " +
        std::to_string(summary.jacobian_evaluation_time_in_seconds) + " s, residual " +
        std::to_string(summary.residual_evaluation_time_in_seconds) + " s, linear iterations " +
        std::to_string(linear_iterations));
  return {{"stage", stage},
          {"seconds", elapsed},
          {"iterations", summary.iterations.size()},
          {"initial_cost", summary.initial_cost},
          {"final_cost", summary.final_cost},
          {"images", used_images.size()},
          {"variable_images", variable_count},
          {"metric_constraints", metric},
          {"rig_frames", rig_frames},
          {"rolling_shutter", shutter != nullptr},
          {"observations", observations},
          {"precise_observations", precise_count},
          {"reference_observations", reference_count},
          {"solver", ceres::LinearSolverTypeToString(summary.linear_solver_type_used)},
          {"explicit_schur", options.use_explicit_schur_complement},
          {"build_seconds", build_seconds},
          {"linear_solver_seconds", summary.linear_solver_time_in_seconds},
          {"jacobian_seconds", summary.jacobian_evaluation_time_in_seconds},
          {"timing",
           {{"observation_build_seconds", observation_build_seconds},
            {"parameter_build_seconds", build_seconds - observation_build_seconds},
            {"ceres_seconds", summary.total_time_in_seconds},
            {"preprocessor_seconds", summary.preprocessor_time_in_seconds},
            {"minimizer_seconds", summary.minimizer_time_in_seconds},
            {"postprocessor_seconds", summary.postprocessor_time_in_seconds},
            {"residual_seconds", summary.residual_evaluation_time_in_seconds}}},
          {"threads", summary.num_threads_used},
          {"preconditioner", ceres::PreconditionerTypeToString(summary.preconditioner_type_used)},
          {"termination", ceres::TerminationTypeToString(summary.termination_type)},
          {"termination_message", summary.message},
          {"successful_steps", summary.num_successful_steps},
          {"unsuccessful_steps", summary.num_unsuccessful_steps},
          {"residual_evaluations", summary.num_residual_evaluations},
          {"jacobian_evaluations", summary.num_jacobian_evaluations},
          {"linear_solves", summary.num_linear_solves},
          {"linear_iterations", linear_iterations},
          {"maximum_linear_iterations", maximum_linear_iterations},
          {"linear_iteration_limit_hits", linear_limit_hits},
          {"function_tolerance", options.function_tolerance},
          {"iteration_profile", iteration_profile},
          {"converged", summary.termination_type == ceres::CONVERGENCE}};
}
}  // namespace geosun

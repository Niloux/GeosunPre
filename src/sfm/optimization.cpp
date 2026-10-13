#include "sfm/optimization.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/product_manifold.h>
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
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 8, Eigen::RowMajor>> camera_j(jacobians[2]);
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
      r[i] = (c[i] - T(target[i])) / T(.5);
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
Json Optimize(Reconstruction& m, const std::array<Camera, 2>& initial, const std::string& stage,
              int workers, int iterations, const std::set<uint32_t>& variable_images,
              const std::vector<PlaneConstraint>& planes, const std::vector<Eigen::Vector3d>& nav,
              const std::set<uint64_t>& selected_points) {
  const auto started = Clock::now();
  ceres::Problem problem;
  // Own losses through residual blocks; all generated observations share the
  // same robust reprojection loss and the problem deletes it exactly once.
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  auto* loss = new ceres::HuberLoss(2.);
  std::set<uint32_t> used_images, used_cameras;
  size_t observations = 0;
  const bool metric = !nav.empty() && !planes.empty();
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
        problem.AddResidualBlock(new Reprojection(image.pixels[o.feature]), loss,
                                 image.pose.params.data(), p.xyz.data(),
                                 m.cameras[image.camera].params.data());
        problem.AddResidualBlock(new Depth(.05 * depth, 1000 / depth), nullptr,
                                 image.pose.params.data(), p.xyz.data());
        used_images.insert(o.image);
        used_cameras.insert(image.camera);
      }
      ordering->AddElementToGroup(p.xyz.data(), 0);
    }
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
      const std::array<double, 8> delta = {
          initial[cid].params[0] * .1, initial[cid].params[1] * .1, 50, 50, .05, .02, .01, .005};
      for (int i = 0; i < 8; ++i) {
        problem.SetParameterLowerBound(params, i, initial[cid].params[i] - delta[i]);
        problem.SetParameterUpperBound(params, i, initial[cid].params[i] + delta[i]);
      }
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
  options.function_tolerance = iterations <= 100 && planes.empty() && nav.empty() ? 1e-5 : 1e-7;
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
  return {{"stage", stage},
          {"seconds", Seconds(started)},
          {"iterations", summary.iterations.size()},
          {"initial_cost", summary.initial_cost},
          {"final_cost", summary.final_cost},
          {"images", used_images.size()},
          {"variable_images", variable_count},
          {"metric_constraints", metric},
          {"observations", observations},
          {"solver", ceres::LinearSolverTypeToString(summary.linear_solver_type_used)},
          {"explicit_schur", options.use_explicit_schur_complement},
          {"build_seconds", build_seconds},
          {"linear_solver_seconds", summary.linear_solver_time_in_seconds},
          {"jacobian_seconds", summary.jacobian_evaluation_time_in_seconds},
          {"converged", summary.termination_type == ceres::CONVERGENCE}};
}
}  // namespace geosun

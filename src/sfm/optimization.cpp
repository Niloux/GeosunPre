#include "sfm/optimization.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/product_manifold.h>

namespace geosun {
namespace {
template <class T>
void Transform(const T* pose, const T* point, T* output) {
  const T q[] = {pose[3], pose[0], pose[1], pose[2]};
  ceres::QuaternionRotatePoint(q, point, output);
  for (int i = 0; i < 3; ++i)
    output[i] += pose[4 + i];
}
template <class T>
void Center(const T* pose, T* out) {
  const T q[] = {pose[3], -pose[0], -pose[1], -pose[2]}, t[] = {-pose[4], -pose[5], -pose[6]};
  ceres::QuaternionRotatePoint(q, t, out);
}
struct Reprojection {
  Eigen::Vector2d pixel;
  template <class T>
  bool operator()(const T* pose, const T* point, const T* cam, T* residual) const {
    T v[3];
    Transform(pose, point, v);
    const T r = ceres::sqrt(v[0] * v[0] + v[1] * v[1] + T(1e-24));
    const T theta = ceres::atan2(r, v[2]), t2 = theta * theta;
    const T d = theta * (T(1) + t2 * (cam[4] + t2 * (cam[5] + t2 * (cam[6] + t2 * cam[7])))) / r;
    residual[0] = cam[0] * v[0] * d + cam[2] - T(pixel.x());
    residual[1] = cam[1] * v[1] * d + cam[3] - T(pixel.y());
    return true;
  }
};
struct Depth {
  double minimum, gain;
  template <class T>
  bool operator()(const T* pose, const T* point, T* residual) const {
    T v[3];
    Transform(pose, point, v);
    residual[0] = v[2] < T(minimum) ? T(gain) * (T(minimum) - v[2]) : T(0);
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
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Reprojection, 2, 7, 3, 8>(new Reprojection{pixels[i]}),
        loss, pose.params.data(), fixed_points[i].data(), calibration.data());
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
              const std::vector<PlaneConstraint>& planes, const std::vector<Eigen::Vector3d>& nav) {
  const auto started = Clock::now();
  ceres::Problem problem;
  // Own losses through residual blocks; all generated observations share the
  // same robust reprojection loss and the problem deletes it exactly once.
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  auto* loss = new ceres::HuberLoss(2.);
  std::set<uint32_t> used_images, used_cameras;
  for (auto& p : m.points)
    if (p.track.size() >= 2) {
      if (!variable_images.empty() &&
          std::none_of(p.track.begin(), p.track.end(),
                       [&](const auto& o) { return variable_images.count(o.image) != 0; }))
        continue;
      for (const auto& o : p.track) {
        auto& image = m.images[o.image];
        const double depth = image.pose.Apply(p.xyz).z();
        if (depth <= 0)
          throw std::runtime_error("BA requires positive initial depth");
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Reprojection, 2, 7, 3, 8>(
                                     new Reprojection{image.pixels[o.feature]}),
                                 loss, image.pose.params.data(), p.xyz.data(),
                                 m.cameras[image.camera].params.data());
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Depth, 1, 7, 3>(new Depth{.05 * depth, 1000 / depth}),
            nullptr, image.pose.params.data(), p.xyz.data());
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
    if (id == m.gauge_image || (!variable_images.empty() && !variable_images.count(id)))
      problem.SetParameterBlockConstant(pose);
  }
  if (used_images.count(m.gauge_image) && used_images.count(m.scale_image))
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
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.linear_solver_ordering = ordering;
  options.function_tolerance = 1e-7;
  Progress progress;
  options.callbacks.push_back(&progress);
  ceres::Solver::Summary summary;
  Log("BA " + stage + ": " + std::to_string(used_images.size()) + " images, " +
      std::to_string(m.Observations()) + " observations");
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable())
    throw std::runtime_error("BA failed: " + summary.BriefReport());
  for (const auto& c : m.cameras)
    c.Radius();
  Log(summary.BriefReport());
  return {{"stage", stage},
          {"seconds", Seconds(started)},
          {"iterations", summary.iterations.size()},
          {"initial_cost", summary.initial_cost},
          {"final_cost", summary.final_cost},
          {"converged", summary.termination_type == ceres::CONVERGENCE}};
}
}  // namespace geosun

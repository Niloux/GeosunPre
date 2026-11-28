#include "sfm/optimization.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/product_manifold.h>
#include <Eigen/Eigenvalues>
#include <cuda_runtime.h>
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
void ProjectAndJacobians(const double* cam, const Eigen::Vector3d& v,
                         Eigen::Map<Eigen::Vector2d>& proj, Eigen::Matrix<double, 2, 3>* dv,
                         Eigen::Matrix<double, 2, 8>* dcam) {
  const double r = std::hypot(v.x(), v.y());
  if (r < 1e-15) {
    proj[0] = cam[2];
    proj[1] = cam[3];
    if (dv) {
      dv->setZero();
      (*dv)(0, 0) = cam[0] / v.z();
      (*dv)(1, 1) = cam[1] / v.z();
    }
    if (dcam) {
      dcam->setZero();
      const double inv_z = 1.0 / v.z();
      (*dcam)(0, 0) = v.x() * inv_z;
      (*dcam)(1, 1) = v.y() * inv_z;
      (*dcam)(0, 2) = 1.0;
      (*dcam)(1, 3) = 1.0;
    }
    return;
  }
  const double theta = std::atan2(r, v.z()), t2 = theta * theta;
  const double radial =
      theta * (1.0 + t2 * (cam[4] + t2 * (cam[5] + t2 * (cam[6] + t2 * cam[7]))));
  const double scale = radial / r;
  proj[0] = cam[0] * v.x() * scale + cam[2];
  proj[1] = cam[1] * v.y() * scale + cam[3];
  if (dv) {
    if (r < 1e-10) {
      dv->setZero();
      (*dv)(0, 0) = cam[0] / v.z();
      (*dv)(1, 1) = cam[1] / v.z();
    } else {
      const double slope =
          1.0 +
          t2 * (3.0 * cam[4] + t2 * (5.0 * cam[5] + t2 * (7.0 * cam[6] + t2 * 9.0 * cam[7])));
      const Eigen::Vector3d dr(v.x() / r, v.y() / r, 0.0);
      const Eigen::Vector3d dt = (v.z() * dr - Eigen::Vector3d(0.0, 0.0, r)) / v.squaredNorm();
      const Eigen::Vector3d ds = (slope * dt - scale * dr) / r;
      dv->row(0) = cam[0] * (v.x() * ds + Eigen::Vector3d(scale, 0.0, 0.0)).transpose();
      dv->row(1) = cam[1] * (v.y() * ds + Eigen::Vector3d(0.0, scale, 0.0)).transpose();
    }
  }
  if (dcam) {
    dcam->setZero();
    (*dcam)(0, 0) = v.x() * scale;
    (*dcam)(1, 1) = v.y() * scale;
    (*dcam)(0, 2) = 1.0;
    (*dcam)(1, 3) = 1.0;
    double power = theta * t2;
    for (int k = 4; k < 8; ++k) {
      const double radial_k = power / r;
      (*dcam)(0, k) = cam[0] * v.x() * radial_k;
      (*dcam)(1, k) = cam[1] * v.y() * radial_k;
      power *= t2;
    }
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
    const Eigen::Matrix3d R = rotation.toRotationMatrix();
    const Eigen::Vector3d rotated = R * point;
    const Eigen::Vector3d v = rotated + Eigen::Map<const Eigen::Vector3d>(pose + 4);
    if (v.z() <= 1e-6)
      return false;
    Eigen::Map<Eigen::Vector2d> output(residual);
    if (!jacobians) {
      ProjectAndJacobians(parameters[2], v, output, nullptr, nullptr);
      output -= pixel;
      return true;
    }
    Eigen::Matrix<double, 2, 3> j;
    Eigen::Matrix<double, 2, 8> cam_local;
    ProjectAndJacobians(parameters[2], v, output, (jacobians[0] || jacobians[1]) ? &j : nullptr,
                        jacobians[2] ? &cam_local : nullptr);
    output -= pixel;
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> pose_j(jacobians[0]);
      pose_j.leftCols<4>() = j * RotationJacobian(rotation, point, rotated, norm);
      pose_j.rightCols<3>() = j;
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[1]);
      point_j = j * R;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 8, Eigen::RowMajor>> cam_j(jacobians[2]);
      cam_j = cam_local;
    }
    return true;
  }
};
// Reprojection of an observation exposed at readout fraction s; see ShutterMotion. The image
// parameter block packs [pose(7), readout_rotation(3)] and the camera block packs
// [intrinsics(8), readout_time(1)].
struct ShutterReprojection : ceres::SizedCostFunction<2, 10, 3, 9> {
  Eigen::Vector2d pixel;
  Eigen::Vector3d velocity;
  double s;
  ShutterReprojection(const Eigen::Vector2d& measurement, const Eigen::Vector3d& camera_velocity,
                      double fraction)
      : pixel(measurement), velocity(camera_velocity), s(fraction) {}
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const auto* pose = parameters[0];
    const Eigen::Map<const Eigen::Vector3d> point(parameters[1]), w(pose + 7);
    const double readout = parameters[2][8];
    Eigen::Quaterniond rotation(pose[3], pose[0], pose[1], pose[2]);
    const double norm = rotation.norm();
    rotation.coeffs() /= norm;
    const Eigen::Matrix3d R = rotation.toRotationMatrix();
    const Eigen::Vector3d rotated = R * point;
    if (rotated.z() + pose[6] <= 1e-6)
      return false;
    const Eigen::Vector3d b =
        rotated + Eigen::Map<const Eigen::Vector3d>(pose + 4) - s * readout * velocity;
    const Eigen::Vector3d phi = s * w;
    const Eigen::Matrix3d r_rs = RodriguesMatrix(phi);
    const Eigen::Vector3d v = r_rs * b;
    if (v.z() <= 1e-6)
      return false;
    Eigen::Map<Eigen::Vector2d> output(residual);
    if (!jacobians) {
      ProjectAndJacobians(parameters[2], v, output, nullptr, nullptr);
      output -= pixel;
      return true;
    }
    Eigen::Matrix<double, 2, 3> j;
    Eigen::Matrix<double, 2, 8> cam_local;
    ProjectAndJacobians(parameters[2], v, output, &j, jacobians[2] ? &cam_local : nullptr);
    output -= pixel;
    const Eigen::Matrix<double, 2, 3> ja = j * r_rs;
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 10, Eigen::RowMajor>> pose_j(jacobians[0]);
      pose_j.leftCols<4>() = ja * RotationJacobian(rotation, point, rotated, norm);
      pose_j.middleCols<3>(4) = ja;
      pose_j.rightCols<3>() = -s * j * SkewSymmetric(v) * LeftJacobianSO3(phi);
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[1]);
      point_j = ja * R;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> cam_j(jacobians[2]);
      cam_j.leftCols<8>() = cam_local;
      cam_j.col(8) = -s * ja * velocity;
    }
    return true;
  }
};
template <int PoseDim>
struct Depth : ceres::SizedCostFunction<2, PoseDim, 3> {
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
    residual[1] = 0;
    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, PoseDim, Eigen::RowMajor>> pose_j(jacobians[0]);
        pose_j.setZero();
        if (active) {
          pose_j.template block<1, 4>(0, 0) =
              -gain * RotationJacobian(rotation, point, rotated, norm).row(2);
          pose_j(0, 6) = -gain;
        }
      }
      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[1]);
        point_j.setZero();
        if (active)
          point_j.row(0) = -gain * rotation.toRotationMatrix().row(2);
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
struct PlaneCost : ceres::SizedCostFunction<2, 3> {
  Eigen::Vector3d unit, center;
  PlaneCost(const Eigen::Vector3d& c, const Eigen::Vector3d& normal, double sigma)
      : unit(normal / sigma), center(c) {}
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::Vector3d> point(parameters[0]);
    residual[0] = unit.dot(point - center);
    residual[1] = 0;
    if (jacobians && jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> point_j(jacobians[0]);
      point_j.row(0) = unit.transpose();
      point_j.row(1).setZero();
    }
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
// A rig frame: the R pose is the rig extrinsic applied to the L pose (propagated by delta_T_w
// across asynchronous L/R trigger offsets). Residuals are the rotation (rad) and translation
// (m, R camera frame) differences over their sigmas.
struct RigLink {
  Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d delta_t = Eigen::Vector3d::Zero();
  double rot_sigma = RigRotationSigma;
  double pos_sigma = RigPositionSigma;
  template <class T>
  bool operator()(const T* left, const T* right, const T* rig, T* r) const {
    const T ql[] = {left[3], left[0], left[1], left[2]},
            qr[] = {right[3], right[0], right[1], right[2]},
            qe[] = {rig[3], rig[0], rig[1], rig[2]},
            dq[] = {T(delta_q.w()), T(delta_q.x()), T(delta_q.y()), T(delta_q.z())};
    const T dt[] = {T(delta_t.x()), T(delta_t.y()), T(delta_t.z())};
    T ql_tilde[4], rotated_dt[3], tl_tilde[3], predicted[4], difference[4], moved[3];
    ceres::QuaternionProduct(ql, dq, ql_tilde);
    ceres::QuaternionRotatePoint(ql, dt, rotated_dt);
    for (int i = 0; i < 3; ++i)
      tl_tilde[i] = rotated_dt[i] + left[4 + i];
    ceres::QuaternionProduct(qe, ql_tilde, predicted);
    const T inverse[] = {predicted[0], -predicted[1], -predicted[2], -predicted[3]};
    ceres::QuaternionProduct(qr, inverse, difference);
    const T sign = difference[0] < T(0) ? T(-2) : T(2);
    ceres::QuaternionRotatePoint(qe, tl_tilde, moved);
    for (int i = 0; i < 3; ++i) {
      r[i] = sign * difference[i + 1] / T(rot_sigma);
      r[i + 3] = (right[4 + i] - moved[i] - rig[4 + i]) / T(pos_sigma);
    }
    return true;
  }
};
// Synchronous frame readout-rotation coupling: wr / tr = R_ext * (wl / tl), written in
// homogeneous cross-multiplied form so it stays smooth through tl = tr = 0 and handles
// both same-sign and opposite-sign sensor scan directions.
struct RigShutterRotation {
  template <class T>
  bool operator()(const T* left, const T* right, const T* cam_l, const T* cam_r, const T* rig,
                  T* r) const {
    const T* wl = left + 7;
    const T* wr = right + 7;
    const T tl = cam_l[8], tr = cam_r[8];
    const T qe[] = {rig[3], rig[0], rig[1], rig[2]};
    T predicted[3];
    ceres::QuaternionRotatePoint(qe, wl, predicted);
    const T scale = ceres::sqrt(tl * tl + tr * tr + T(1e-6));
    for (int i = 0; i < 3; ++i)
      r[i] = (tl * wr[i] - tr * predicted[i]) / (scale * T(RigShutterRotationSigma));
    return true;
  }
};
// Weak IMU angular-rate prior tying per-image readout rotation to tau * R_w2c * omega_world.
struct ImuShutterRotation {
  Eigen::Vector3d omega_world;
  template <class T>
  bool operator()(const T* image, const T* cam, T* r) const {
    const T q[] = {image[3], image[0], image[1], image[2]};
    const T w_world[] = {T(omega_world.x()), T(omega_world.y()), T(omega_world.z())};
    T w_cam[3];
    ceres::QuaternionRotatePoint(q, w_world, w_cam);
    const T tau = cam[8];
    for (int i = 0; i < 3; ++i)
      r[i] = (image[7 + i] - tau * w_cam[i]) / T(ImuShutterRotationSigma);
    return true;
  }
};
struct EqualReadoutDuration {
  template <class T>
  bool operator()(const T* cam_l, const T* cam_r, T* r) const {
    const T tl = cam_l[8], tr = cam_r[8];
    const T scale = ceres::sqrt(tl * tl + T(1e-8)) + ceres::sqrt(tr * tr + T(1e-8));
    r[0] = (tl * tl - tr * tr) / (scale * T(EqualReadoutSigma));
    return true;
  }
};
// Consecutive exposure relative rotation prior: R_b ≈ relative * R_a.
struct RelativeRotationCost {
  Eigen::Quaterniond relative;
  double sigma;
  template <class T>
  bool operator()(const T* pose_a, const T* pose_b, T* r) const {
    const T qa[] = {pose_a[3], pose_a[0], pose_a[1], pose_a[2]},
            qb[] = {pose_b[3], pose_b[0], pose_b[1], pose_b[2]},
            qrel[] = {T(relative.w()), T(relative.x()), T(relative.y()), T(relative.z())};
    T predicted[4], difference[4];
    ceres::QuaternionProduct(qrel, qa, predicted);
    const T inv_pred[] = {predicted[0], -predicted[1], -predicted[2], -predicted[3]};
    ceres::QuaternionProduct(qb, inv_pred, difference);
    const T sign = difference[0] < T(0) ? T(-2) : T(2);
    for (int i = 0; i < 3; ++i)
      r[i] = sign * difference[i + 1] / T(sigma);
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
bool RigExtrinsic(const Reconstruction& m, const std::vector<RigFrame>& frames, Pose& rig) {
  std::vector<Eigen::Quaterniond> rotations;
  std::vector<Eigen::Vector3d> translations;
  for (const auto& f : frames) {
    const auto& left = m.images[f.left];
    const auto& right = m.images[f.right];
    if (!left.registered || !right.registered)
      continue;
    const Eigen::Quaterniond q_l_tilde = (left.pose.Rotation() * f.delta_rotation).normalized();
    const Eigen::Vector3d t_l_tilde =
        left.pose.Rotation() * f.delta_translation + left.pose.Translation();
    const Eigen::Quaterniond q = (right.pose.Rotation() * q_l_tilde.conjugate()).normalized();
    rotations.push_back(q);
    translations.push_back(right.pose.Translation() - q * t_l_tilde);
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
              const std::set<uint64_t>& selected_points, ShutterMotion* shutter,
              const std::vector<RigFrame>& frames, const std::set<uint64_t>& fixed_points,
              bool fixed_rig_and_readout, const std::vector<Eigen::Vector3d>& imu_omega,
              const std::vector<RelativeRotationPrior>& relative_rotations,
              double function_tolerance) {
  const auto started = Clock::now();
  ceres::Problem problem;
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  auto* loss = new ceres::HuberLoss(2.);
  std::set<uint32_t> used_images, used_cameras;
  size_t observations = 0;
  const bool metric = !nav.empty();
  size_t variable_count = 0;
  bool has_variable_points = false;
  std::vector<std::array<double, 10>> shutter_images;
  std::array<std::array<double, 9>, 2> shutter_cameras{};
  if (shutter) {
    shutter_images.resize(m.images.size());
    for (size_t id = 0; id < m.images.size(); ++id) {
      std::copy_n(m.images[id].pose.params.data(), 7, shutter_images[id].data());
      std::copy_n(shutter->rotation[id].data(), 3, shutter_images[id].data() + 7);
    }
    for (size_t cid = 0; cid < m.cameras.size(); ++cid) {
      std::copy_n(m.cameras[cid].params.data(), 8, shutter_cameras[cid].data());
      shutter_cameras[cid][8] = shutter->readout[cid];
    }
  }
  for (auto& p : m.points)
    if (p.track.size() >= 2) {
      const uint64_t pid = uint64_t(&p - m.points.data());
      if (!selected_points.empty() && !selected_points.count(pid))
        continue;
      if (!variable_images.empty() &&
          std::none_of(p.track.begin(), p.track.end(),
                       [&](const auto& o) { return variable_images.count(o.image) != 0; }))
        continue;
      const bool point_fixed = fixed_points.count(pid) != 0;
      for (const auto& o : p.track) {
        ++observations;
        auto& image = m.images[o.image];
        const Eigen::Vector3d p_cam = image.pose.Apply(p.xyz);
        const double depth = p_cam.z();
        if (depth <= 1e-6)
          throw std::runtime_error("BA requires initial depth > 1e-6 (point " + std::to_string(pid) +
                                   ", image " + std::to_string(o.image) + ")");
        const bool need_depth_guard = !point_fixed && (!metric || (p.track.size() == 2 && depth < 1.0));
        const Eigen::Vector2d pixel = image.pixels[o.feature];
        if (shutter) {
          const double s = shutter->Fraction(m.cameras[image.camera], pixel);
          const double exposed_depth = shutter->Apply(o.image, image.camera, s, p_cam).z();
          if (exposed_depth <= 1e-6)
            throw std::runtime_error("Shutter BA requires initial exposed depth > 1e-6 (point " +
                                     std::to_string(pid) + ", image " + std::to_string(o.image) +
                                     ")");
          problem.AddResidualBlock(new ShutterReprojection(pixel, shutter->velocity[o.image], s),
                                   loss, shutter_images[o.image].data(), p.xyz.data(),
                                   shutter_cameras[image.camera].data());
          if (need_depth_guard)
            problem.AddResidualBlock(new Depth<10>(.05 * depth, 1000 / depth), nullptr,
                                     shutter_images[o.image].data(), p.xyz.data());
        } else {
          problem.AddResidualBlock(new Reprojection(pixel), loss, image.pose.params.data(),
                                   p.xyz.data(), m.cameras[image.camera].params.data());
          if (need_depth_guard)
            problem.AddResidualBlock(new Depth<7>(.05 * depth, 1000 / depth), nullptr,
                                     image.pose.params.data(), p.xyz.data());
        }
        used_images.insert(o.image);
        used_cameras.insert(image.camera);
      }
      if (point_fixed) {
        problem.SetParameterBlockConstant(p.xyz.data());
      } else {
        ordering->AddElementToGroup(p.xyz.data(), 0);
        has_variable_points = true;
      }
    }
  if (fixed_rig_and_readout && !frames.empty()) {
    for (const auto& f : frames) {
      if (!m.images[f.left].registered || !m.images[f.right].registered)
        continue;
      if (variable_images.count(f.left) || variable_images.count(f.right)) {
        used_images.insert(f.left);
        used_images.insert(f.right);
        used_cameras.insert(m.images[f.left].camera);
        used_cameras.insert(m.images[f.right].camera);
      }
    }
  }
  const double observation_build_seconds = Seconds(started);
  if (used_images.size() < 2)
    throw std::runtime_error("Insufficient BA observations");
  for (uint32_t id : used_images) {
    const bool fixed_pose =
        (!metric && id == m.gauge_image) || (!variable_images.empty() && !variable_images.count(id));
    if (shutter) {
      auto* block = shutter_images[id].data();
      if (!problem.HasParameterBlock(block))
        problem.AddParameterBlock(block, 10);
      ordering->AddElementToGroup(block, 1);
      if (fixed_pose && fixed_rig_and_readout) {
        problem.SetParameterBlockConstant(block);
      } else if (fixed_pose) {
        problem.SetManifold(block, new ceres::SubsetManifold(10, {0, 1, 2, 3, 4, 5, 6}));
      } else {
        problem.SetManifold(
            block, new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                              ceres::EuclideanManifold<6>>());
        ++variable_count;
      }
      if (!(fixed_pose && fixed_rig_and_readout)) {
        for (int i = 7; i < 10; ++i) {
          problem.SetParameterLowerBound(block, i, -ShutterRotationBound);
          problem.SetParameterUpperBound(block, i, ShutterRotationBound);
        }
      }
    } else {
      auto* pose = m.images[id].pose.params.data();
      if (!problem.HasParameterBlock(pose))
        problem.AddParameterBlock(pose, 7);
      problem.SetManifold(
          pose,
          new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>());
      ordering->AddElementToGroup(pose, 1);
      if (fixed_pose)
        problem.SetParameterBlockConstant(pose);
      else
        ++variable_count;
    }
  }
  if (!metric && used_images.count(m.gauge_image) && used_images.count(m.scale_image)) {
    if (shutter)
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Baseline, 1, 10, 10>(new Baseline{m.gauge_baseline}),
          nullptr, shutter_images[m.gauge_image].data(), shutter_images[m.scale_image].data());
    else
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Baseline, 1, 7, 7>(new Baseline{m.gauge_baseline}),
          nullptr, m.images[m.gauge_image].pose.params.data(),
          m.images[m.scale_image].pose.params.data());
  }
  for (uint32_t cid : used_cameras) {
    double* params = shutter ? shutter_cameras[cid].data() : m.cameras[cid].params.data();
    const int dim = shutter ? 9 : 8;
    if (!problem.HasParameterBlock(params))
      problem.AddParameterBlock(params, dim);
    ordering->AddElementToGroup(params, 1);
    if (fixed_rig_and_readout) {
      problem.SetParameterBlockConstant(params);
    } else if (stage == "fixed") {
      if (shutter)
        problem.SetManifold(params, new ceres::SubsetManifold(9, {0, 1, 2, 3, 4, 5, 6, 7}));
      else
        problem.SetParameterBlockConstant(params);
    } else {
      std::vector<int> fixed;
      if (stage == "focal")
        fixed = {2, 3, 6, 7};
      else if (stage == "radial")
        fixed = {2, 3};
      else if (stage != "principal")
        throw std::runtime_error("Unknown calibration stage: " + stage);
      if (!fixed.empty())
        problem.SetManifold(params, new ceres::SubsetManifold(dim, fixed));
      const auto delta = initial[cid].CalibrationBounds();
      for (int i = 0; i < 8; ++i) {
        problem.SetParameterLowerBound(params, i, initial[cid].params[i] - delta[i]);
        problem.SetParameterUpperBound(params, i, initial[cid].params[i] + delta[i]);
      }
    }
    if (shutter && !fixed_rig_and_readout) {
      problem.SetParameterLowerBound(params, 8, -.05);
      problem.SetParameterUpperBound(params, 8, .05);
    }
  }
  for (const auto& p : planes)
    if (problem.HasParameterBlock(m.points.at(p.point).xyz.data()))
      problem.AddResidualBlock(new PlaneCost(p.center, p.normal, p.sigma), loss,
                               m.points[p.point].xyz.data());
  if (!nav.empty())
    for (uint32_t id : used_images) {
      if (shutter)
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Position, 3, 10>(new Position{nav.at(id)}), loss,
            shutter_images[id].data());
      else
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Position, 3, 7>(new Position{nav.at(id)}), loss,
            m.images[id].pose.params.data());
    }
  if (shutter && !imu_omega.empty()) {
    if (imu_omega.size() != m.images.size())
      throw std::runtime_error("IMU angular velocity count differs from image count");
    ceres::LossFunction* imu_shutter_loss = nullptr;
    for (uint32_t id : used_images) {
      const bool fixed_pose = (!metric && id == m.gauge_image) ||
                              (!variable_images.empty() && !variable_images.count(id));
      if ((fixed_pose && fixed_rig_and_readout) || !imu_omega[id].allFinite())
        continue;
      if (!imu_shutter_loss)
        imu_shutter_loss = new ceres::HuberLoss(3.);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<ImuShutterRotation, 3, 10, 9>(
              new ImuShutterRotation{imu_omega[id]}),
          imu_shutter_loss, shutter_images[id].data(),
          shutter_cameras[m.images[id].camera].data());
    }
  }
  if (shutter && !fixed_rig_and_readout && used_cameras.size() == 2 &&
      m.cameras[0].width == m.cameras[1].width && m.cameras[0].height == m.cameras[1].height)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<EqualReadoutDuration, 1, 9, 9>(new EqualReadoutDuration),
        nullptr, shutter_cameras[0].data(), shutter_cameras[1].data());
  size_t relative_rotation_count = 0;
  if (!shutter && !relative_rotations.empty()) {
    ceres::LossFunction* rel_rot_loss = nullptr;
    for (const auto& rr : relative_rotations) {
      if (!used_images.count(rr.a) || !used_images.count(rr.b) || !(rr.sigma > 0))
        continue;
      const bool fixed_a = (!metric && rr.a == m.gauge_image) ||
                           (!variable_images.empty() && !variable_images.count(rr.a));
      const bool fixed_b = (!metric && rr.b == m.gauge_image) ||
                           (!variable_images.empty() && !variable_images.count(rr.b));
      if (fixed_a && fixed_b)
        continue;
      if (!rel_rot_loss)
        rel_rot_loss = new ceres::HuberLoss(2.);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<RelativeRotationCost, 3, 7, 7>(
              new RelativeRotationCost{rr.relative.normalized(), rr.sigma}),
          rel_rot_loss, m.images[rr.a].pose.params.data(), m.images[rr.b].pose.params.data());
      ++relative_rotation_count;
    }
  }
  Pose rig;
  size_t rig_frames = 0;
  ceres::LossFunction* rig_shutter_loss = nullptr;
  bool have_rig = false;
  if (!frames.empty()) {
    if (fixed_rig_and_readout && !variable_images.empty()) {
      std::vector<RigFrame> anchor_frames;
      anchor_frames.reserve(frames.size());
      for (const auto& f : frames)
        if (!variable_images.count(f.left) && !variable_images.count(f.right))
          anchor_frames.push_back(f);
      have_rig = RigExtrinsic(m, anchor_frames, rig) || RigExtrinsic(m, frames, rig);
    } else {
      have_rig = RigExtrinsic(m, frames, rig);
    }
  }
  if (have_rig) {
    // Quadratic rig constraints follow ReanchorRigOutliers so weak-texture corridor outliers
    // cannot drift off the calibrated rig.
    for (const auto& f : frames)
      if (used_images.count(f.left) && used_images.count(f.right)) {
        const double rot_sigma = f.synchronous ? RigRotationSigma : AsyncRigRotationSigma;
        const double pos_sigma = f.synchronous ? RigPositionSigma : AsyncRigPositionSigma;
        if (shutter) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<RigLink, 6, 10, 10, 7>(
                  new RigLink{f.delta_rotation.normalized(), f.delta_translation, rot_sigma,
                              pos_sigma}),
              nullptr, shutter_images[f.left].data(), shutter_images[f.right].data(),
              rig.params.data());
          if (f.synchronous && m.images[f.left].camera != m.images[f.right].camera) {
            if (!rig_shutter_loss)
              rig_shutter_loss = new ceres::HuberLoss(4);
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RigShutterRotation, 3, 10, 10, 9, 9, 7>(
                    new RigShutterRotation),
                rig_shutter_loss, shutter_images[f.left].data(), shutter_images[f.right].data(),
                shutter_cameras[m.images[f.left].camera].data(),
                shutter_cameras[m.images[f.right].camera].data(), rig.params.data());
          }
        } else {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<RigLink, 6, 7, 7, 7>(
                  new RigLink{f.delta_rotation.normalized(), f.delta_translation, rot_sigma,
                              pos_sigma}),
              nullptr, m.images[f.left].pose.params.data(), m.images[f.right].pose.params.data(),
              rig.params.data());
        }
        if (!rig_frames++) {
          problem.SetManifold(rig.params.data(),
                              new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                                         ceres::EuclideanManifold<3>>());
          ordering->AddElementToGroup(rig.params.data(), 1);
        }
      }
    if (rig_frames && fixed_rig_and_readout)
      problem.SetParameterBlockConstant(rig.params.data());
  }
  ceres::Solver::Options options;
  options.num_threads = workers;
  options.max_num_iterations = iterations;
  options.logging_type = ceres::SILENT;
  const bool global_large =
      has_variable_points && variable_images.empty() && used_images.size() >= 64;
  bool use_cuda_dense = false;
  if (global_large) {
    const size_t schur_cols = used_images.size() * (shutter ? 9 : 6) +
                              used_cameras.size() * (shutter ? 9 : 8) + (rig_frames ? 6 : 0);
    // Mixed-precision CUDA Cholesky holds ~12 B/entry on GPU (FP64 + FP32 lhs plus workspace).
    const size_t dense_bytes = size_t(12) * schur_cols * schur_cols;
    size_t free_bytes = 0, total_bytes = 0;
    if (schur_cols <= 12000 && cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        free_bytes >= 2 * dense_bytes + (size_t(512) << 20))
      use_cuda_dense = true;
  }
  const bool iterative = global_large && !use_cuda_dense;
  options.linear_solver_type = !has_variable_points
                                   ? ceres::DENSE_QR
                                   : (use_cuda_dense
                                          ? ceres::DENSE_SCHUR
                                          : (iterative ? ceres::ITERATIVE_SCHUR : ceres::SPARSE_SCHUR));
  if (use_cuda_dense) {
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.use_mixed_precision_solves = true;
  } else if (iterative) {
    options.preconditioner_type = ceres::SCHUR_JACOBI;
  }
  options.use_explicit_schur_complement = iterative && used_images.size() <= 4096;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.linear_solver_ordering = has_variable_points ? ordering : nullptr;
  // Below 1e-6 the final refinements only creep: in the shutter BA the last half of the
  // iterations moved camera centres by under 5 um and changed held-out error by 3e-5 px.
  options.function_tolerance =
      function_tolerance > 0 ? function_tolerance
                             : (iterations <= 100 && planes.empty() ? 1e-5 : 1e-6);
  Progress progress;
  options.callbacks.push_back(&progress);
  ceres::Solver::Summary summary;
  Log("BA " + stage + ": " + std::to_string(used_images.size()) + " images, " +
      std::to_string(observations) + " observations");
  struct BaSnapshot {
    std::vector<Eigen::Vector3d> points;
    std::vector<std::array<double, 7>> poses;
    std::array<std::array<double, 8>, 2> cameras{};
    std::vector<std::array<double, 10>> shutter_images;
    std::array<std::array<double, 9>, 2> shutter_cameras{};
    std::array<double, 7> rig{};
  };
  std::optional<BaSnapshot> snapshot;
  if (use_cuda_dense && options.use_mixed_precision_solves) {
    BaSnapshot snap;
    snap.points.resize(m.points.size());
    for (size_t i = 0; i < m.points.size(); ++i)
      snap.points[i] = m.points[i].xyz;
    snap.poses.resize(m.images.size());
    for (size_t i = 0; i < m.images.size(); ++i)
      snap.poses[i] = m.images[i].pose.params;
    for (size_t c = 0; c < 2; ++c)
      snap.cameras[c] = m.cameras[c].params;
    snap.shutter_images = shutter_images;
    snap.shutter_cameras = shutter_cameras;
    snap.rig = rig.params;
    snapshot = std::move(snap);
  }
  auto restore_snapshot = [&] {
    if (!snapshot)
      return;
    for (size_t i = 0; i < m.points.size(); ++i)
      m.points[i].xyz = snapshot->points[i];
    for (size_t i = 0; i < m.images.size(); ++i)
      m.images[i].pose.params = snapshot->poses[i];
    for (size_t c = 0; c < 2; ++c)
      m.cameras[c].params = snapshot->cameras[c];
    shutter_images = snapshot->shutter_images;
    shutter_cameras = snapshot->shutter_cameras;
    rig.params = snapshot->rig;
  };
  const double build_seconds = Seconds(started);
  ceres::Solve(options, &problem, &summary);
  bool mixed_precision_fallback = false;
  bool cuda_fallback_iterative = false;
  if (!summary.IsSolutionUsable() && use_cuda_dense && options.use_mixed_precision_solves) {
    Log("BA CUDA mixed-precision failed (" + summary.message + "); retrying CUDA FP64");
    restore_snapshot();
    options.use_mixed_precision_solves = false;
    mixed_precision_fallback = true;
    summary = ceres::Solver::Summary();
    ceres::Solve(options, &problem, &summary);
  }
  if (!summary.IsSolutionUsable() && use_cuda_dense) {
    Log("BA CUDA DENSE_SCHUR failed (" + summary.message + "); falling back to ITERATIVE_SCHUR");
    restore_snapshot();
    use_cuda_dense = false;
    cuda_fallback_iterative = true;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_explicit_schur_complement = used_images.size() <= 4096;
    summary = ceres::Solver::Summary();
    ceres::Solve(options, &problem, &summary);
  }
  if (!summary.IsSolutionUsable())
    throw std::runtime_error("BA failed: " + summary.BriefReport());
  if (shutter) {
    for (uint32_t id : used_images) {
      std::copy_n(shutter_images[id].data(), 7, m.images[id].pose.params.data());
      std::copy_n(shutter_images[id].data() + 7, 3, shutter->rotation[id].data());
    }
    for (uint32_t cid : used_cameras) {
      std::copy_n(shutter_cameras[cid].data(), 8, m.cameras[cid].params.data());
      shutter->readout[cid] = shutter_cameras[cid][8];
    }
  }
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
          {"relative_rotation_priors", relative_rotation_count},
          {"rolling_shutter", shutter != nullptr},
          {"observations", observations},
          {"solver", ceres::LinearSolverTypeToString(summary.linear_solver_type_used)},
          {"cuda_dense", use_cuda_dense},
          {"mixed_precision", summary.mixed_precision_solves_used},
          {"mixed_precision_fallback", mixed_precision_fallback},
          {"cuda_fallback_iterative", cuda_fallback_iterative},
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

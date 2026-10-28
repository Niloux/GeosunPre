#pragma once
#include "geometry/model.h"
#include <set>

namespace geosun {
struct PlaneConstraint {
  uint64_t point;
  Eigen::Vector3d center, normal;
  double sigma;
};
void RefinePose(const Camera& camera, Pose& pose, const std::vector<Eigen::Vector3d>& points,
                const std::vector<Eigen::Vector2d>& pixels);
// Robust cost of a reprojection error, quadratic up to 2 px.
inline double TrackCost(double error) { return error <= 2 ? .5 * error * error : 2 * (error - 1); }
// Point refinement with cameras fixed: up to six reweighted Gauss-Newton steps, each halved up
// to six times until the cost of the track falls. cost(point) is the summed TrackCost or infinity;
// linearize(observation, point, residual, jacobian) gives the pixel residual and its derivative.
template <class Cost, class Linearize>
void RefineTrackPoint(const std::vector<Observation>& track, Eigen::Vector3d& point, Cost cost_of,
                      Linearize linearize, const char* nonfinite) {
  double cost = cost_of(point);
  for (int iteration = 0; iteration < 6; ++iteration) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (const auto& o : track) {
      Eigen::Vector2d residual;
      Eigen::Matrix<double, 2, 3> j;
      linearize(o, point, residual, j);
      const double weight = std::min(1., 2 / std::max(residual.norm(), 1e-12));
      normal.noalias() += weight * j.transpose() * j;
      rhs.noalias() -= weight * j.transpose() * residual;
    }
    const Eigen::Vector3d step = normal.ldlt().solve(rhs);
    if (!step.allFinite())
      throw std::runtime_error(nonfinite);
    bool accepted = false;
    for (int backtrack = 0; backtrack < 6; ++backtrack) {
      const Eigen::Vector3d candidate = point + step * std::pow(.5, backtrack);
      const double next = cost_of(candidate);
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
void RefinePoint(const Reconstruction& model, const std::vector<Observation>& track,
                 Eigen::Vector3d& point);
// CameraPos agrees with the final camera centres to 6-19 mm (median) and at most 87 mm on the
// three test scenes; a centre off by more than the gate is a wrong registration (repeated texture
// put 展厅 R exposures 0.5-1.4 m away).
constexpr double NavigationSigma = .03, NavigationGate = .25;
// Navigation positions give the metric gauge: without them the gauge image is held and the
// gauge baseline kept.
Json Optimize(Reconstruction& model, const std::array<Camera, 2>& initial, const std::string& stage,
              int workers, int iterations, const std::set<uint32_t>& variable_images = {},
              const std::vector<PlaneConstraint>& planes = {},
              const std::vector<Eigen::Vector3d>& navigation = {},
              const std::set<uint64_t>& selected_points = {},
              const std::vector<std::vector<uint8_t>>& precise_observations = {},
              const Reconstruction* reference = nullptr, ShutterMotion* shutter = nullptr);
}  // namespace geosun

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
Json Optimize(Reconstruction& model, const std::array<Camera, 2>& initial, const std::string& stage,
              int workers, int iterations, const std::set<uint32_t>& variable_images = {},
              const std::vector<PlaneConstraint>& planes = {},
              const std::vector<Eigen::Vector3d>& navigation = {},
              const std::set<uint64_t>& selected_points = {});
}  // namespace geosun

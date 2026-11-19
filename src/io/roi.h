#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "app/runtime.h"
#include "geometry/model.h"

namespace geosun {

struct RoiParams {
  double cell_size_m = 1.0;
  uint32_t min_cell_points = 30;
  double margin_xy_m = 2.0;
  double margin_z_bottom_m = 1.0;
  double margin_z_top_m = 3.0;
  double max_top_above_p999_m = 15.0;
  int max_vertices = 64;
};

struct RoiPolygon {
  std::vector<Eigen::Vector2d> vertices_xy;  // World-frame simple non-self-intersecting 2D polygon
  double z_min = 0.0;
  double z_max = 0.0;
  Eigen::Vector3d bounds_min = Eigen::Vector3d::Zero();
  Eigen::Vector3d bounds_max = Eigen::Vector3d::Zero();
  double area_m2 = 0.0;
  uint64_t occupied_cells = 0;
  double kept_las_fraction = 0.0;
  size_t cameras_inside = 0;
  size_t cameras_total = 0;
};

inline uint64_t PackGridCell2D(int32_t ix, int32_t iy) {
  return (uint64_t(uint32_t(ix)) << 32) | uint64_t(uint32_t(iy));
}

inline std::pair<int32_t, int32_t> UnpackGridCell2D(uint64_t key) {
  return {static_cast<int32_t>(static_cast<uint32_t>(key >> 32)),
          static_cast<int32_t>(static_cast<uint32_t>(key & 0xffffffffULL))};
}

inline bool GridCellCoord2D(double x, double y, double cell_size, int32_t& ix, int32_t& iy) {
  if (!std::isfinite(x) || !std::isfinite(y) || !(cell_size > 0.0))
    return false;
  const double gx = std::floor(x / cell_size);
  const double gy = std::floor(y / cell_size);
  if (!std::isfinite(gx) || !std::isfinite(gy) ||
      gx < -1e8 || gx > 1e8 || gy < -1e8 || gy > 1e8)
    return false;
  ix = static_cast<int32_t>(gx);
  iy = static_cast<int32_t>(gy);
  return true;
}

bool PointInPolygon2D(const std::vector<Eigen::Vector2d>& poly, double x, double y);
bool IsSimpleNonSelfIntersecting(const std::vector<Eigen::Vector2d>& poly);
double PolygonArea2D(const std::vector<Eigen::Vector2d>& poly);

RoiPolygon BuildRoiPolygon2D(const std::unordered_map<uint64_t, uint32_t>& raw_cell_counts,
                             const Reconstruction& model, const RoiParams& params = {});

void FinalizeRoiZRange(RoiPolygon& roi, std::vector<float>& inside_z_samples,
                       const Reconstruction& model, const RoiParams& params = {});

Json RoiToJson(const RoiPolygon& roi, const RoiParams& params,
               const std::string& coordinate_frame);

RoiPolygon ValidateRoiJson(const Json& roi_json, const Reconstruction& model,
                           const std::string& expected_coordinate_frame);

}  // namespace geosun

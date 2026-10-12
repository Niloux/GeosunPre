#pragma once
#include <colmap/geometry/sim3.h>
#include <colmap/scene/reconstruction.h>

#include <memory>
#include <nanoflann.hpp>

#include "las_io.h"

namespace registration {
using Tile = std::array<int64_t, 3>;
using Points = std::vector<Eigen::Vector3d>;
struct Cloud {
  const Points* points;
  size_t kdtree_get_point_count() const { return points->size(); }
  double kdtree_get_pt(size_t i, size_t axis) const {
    return (*points)[i][axis];
  }
  template <class Box>
  bool kdtree_get_bbox(Box&) const {
    return false;
  }
};
using Tree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 3, size_t>;
struct Surface {
  Points points, centres, normals;
  std::vector<double> noise;
  std::vector<uint8_t> valid, coherent;
  Cloud cloud{&points};
  std::unique_ptr<Tree> tree;
  double voxel = 0.;
  void Index();
};
struct Regions {
  Eigen::Vector3d origin;
  double size;
  std::vector<Tile> tiles;
  Json report;
};
struct Residual {
  size_t nearest;
  double distance, plane;
  bool supported;
};
double Quantile(std::vector<double> values, double fraction);
Json Statistics(const std::vector<double>& values);
Json Matrix(const colmap::Sim3d& transform);
double Huber(double value, double cutoff);
std::unique_ptr<Surface> Surfaces(const Points& points, double voxel,
                                  int workers);
std::unique_ptr<Surface> CoherentSurfaces(const Points& points, double voxel,
                                          int workers);
std::vector<Residual> Residuals(const Points& points, const Surface& surface,
                                double maximum);
Regions Partition(const Points& points);
colmap::Sim3d RefineSimilarity(const Points& source, const Points& target,
                               colmap::Sim3d transform, const Regions& regions,
                               const std::vector<double>& voxels,
                               const std::vector<double>& distances,
                               int workers, Json& optimization,
                               std::unique_ptr<Surface>& surface);
struct JointResult {
  std::unique_ptr<colmap::Reconstruction> model;
  colmap::Sim3d transform;
  Json report;
};
JointResult RefineJoint(
    const colmap::Reconstruction& base, const colmap::Sim3d& initial,
    const colmap::Sim3d& partition_transform,
    const std::map<colmap::image_t, Eigen::Vector3d>& positions,
    const Points& target, const std::vector<colmap::point3D_t>& fit_ids,
    const Regions& regions, double voxel, double maximum, int workers);
}  // namespace registration
void RunRegistration(const Arguments& args);

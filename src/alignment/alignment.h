#pragma once
#include "geometry/model.h"
#include "sfm/optimization.h"

namespace geosun {
struct AlignmentLimits {
  double support_fraction = .25, median_m = .1, p90_m = .3;
  double median_px = 2, p90_px = 4;
};
Json Align(const Scene& scene, Reconstruction& model, int workers, const AlignmentLimits& limits);
std::vector<PlaneConstraint> LasPlaneConstraints(const Scene& scene, const Reconstruction& model,
                                                 int workers);
Json FilterSparseStructure(const Scene& scene, Reconstruction& model, const AlignmentLimits& limits,
                           int workers);
}  // namespace geosun

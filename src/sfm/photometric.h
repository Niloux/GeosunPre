#pragma once
#include "alignment/alignment.h"

namespace geosun {
struct PhotometricRefinement {
  Reconstruction candidate;
  Json report;
  bool accepted = false;
};
PhotometricRefinement RefinePhotometric(const Scene& scene, const Reconstruction& source,
                                        int workers, const AlignmentLimits& limits,
                                        int ba_workers = 0);
}  // namespace geosun

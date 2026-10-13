#pragma once
#include "alignment/alignment.h"

namespace geosun {
struct PhotometricRefinement {
  Reconstruction candidate;
  Json report;
  bool accepted = false;
};
PhotometricRefinement RefinePhotometric(const Scene& scene, const Reconstruction& source,
                                      int workers, const AlignmentLimits& limits);
Json RefinementResult(const Scene& scene, const Reconstruction& selected, const Json& original,
                      const Json& refinement);
void RefinePhotometric(const Arguments& args);
}  // namespace geosun

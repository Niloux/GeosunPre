#pragma once
#include "alignment/alignment.h"

namespace geosun {
struct StructureCleanup {
  Reconstruction candidate;
  Json report;
  bool accepted = false;
};
StructureCleanup CleanSparseStructure(const Scene& scene, const Reconstruction& source, int workers,
                                      const AlignmentLimits& limits,
                                      const fs::path& las_sample_cache = {});
}  // namespace geosun

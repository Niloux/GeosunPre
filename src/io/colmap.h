#pragma once
#include "geometry/model.h"
#include "geometry/supervision.h"

namespace geosun {
// A rolling-shutter corrected model is delivered with the original photos and its readout motion.
void ExportBundle(const Scene& scene, const Reconstruction& model, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report,
                  const SupervisionOptions& supervision = {});
Json CheckBundle(const fs::path& output);
}  // namespace geosun

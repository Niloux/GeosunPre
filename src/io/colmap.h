#pragma once
#include "geometry/model.h"
#include "geometry/supervision.h"

namespace geosun {
// photos is "rectified" or "original"; it only matters for rolling-shutter corrected models.
void ExportBundle(const Scene& scene, const Reconstruction& model, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report,
                  const SupervisionOptions& supervision = {},
                  const std::string& photos = "rectified");
Json CheckBundle(const fs::path& output);
}  // namespace geosun

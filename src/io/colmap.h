#pragma once
#include "geometry/model.h"

namespace geosun {
void ExportBundle(const Scene& scene, const Reconstruction& model, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report);
Json CheckBundle(const fs::path& output);
}  // namespace geosun

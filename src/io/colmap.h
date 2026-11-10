#pragma once
#include "geometry/model.h"
#include "geometry/supervision.h"

namespace geosun {
// A rolling-shutter corrected model is delivered with the original photos and its readout motion.
// With crop, each lens's photos, masks and model are cut to the window of its valid domain.
void ExportBundle(const Scene& scene, const Reconstruction& model, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report,
                  const SupervisionOptions& supervision = {}, bool crop = true);
Json CheckBundle(const fs::path& output);
// Measures a delivery's depth and normal maps against what they were not fitted to: SfM track
// points seen in each photo and the maps of neighbouring views reprojected through the poses. With
// compare, a second delivery of the same model, also on the pixels valid in both, so that dropping
// pixels does not count as an improvement. Only reads the deliveries.
// Writes an existing delivery's masks with the persistent transient components made static again
// (see StaticMasks) into the new directory output, for deliveries made without them; the delivery
// is only read. Its depth maps were made with the old masks.
Json RefineStaticMasks(const fs::path& delivery, const fs::path& input, const fs::path& las,
                       bool enhanced, const fs::path& output);
Json MeasureSupervision(const fs::path& output, const fs::path& compare);
}  // namespace geosun

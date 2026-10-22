#pragma once
#include "geometry/model.h"

namespace geosun {
struct SupervisionOptions {
  bool enabled = true;
  int max_size = 1600, boundary = 6;
  int cache_mb = 1024;
  fs::path mask_dir;
};
SupervisionOptions ParseSupervisionOptions(const Arguments& args);
// masks is the directory of the selected model's static masks, named by image index.
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& masks,
                        const SupervisionOptions& options, bool rolling = false);
// With motion, each pixel uses the camera of its own readout line (original rolling-shutter
// photos); its normal is expressed in that camera.
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model, const fs::path& masks,
                          const fs::path& output, const SupervisionOptions& options,
                          const ShutterMotion* motion = nullptr);
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary);
}  // namespace geosun

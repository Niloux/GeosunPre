#pragma once
#include "geometry/model.h"

namespace geosun {
struct SupervisionOptions {
  bool enabled = false;
  int max_size = 1600, boundary = 6;
  fs::path mask_dir;
};
SupervisionOptions ParseSupervisionOptions(const Arguments& args);
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& work,
                        const SupervisionOptions& options);
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model,
                          const fs::path& work, const fs::path& output,
                          const SupervisionOptions& options);
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary);
}  // namespace geosun

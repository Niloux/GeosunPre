#pragma once
#include "geometry/model.h"
#include <optional>

namespace geosun {
struct SupervisionOptions {
  bool enabled = true;
  // Persistent people (pictures, posters, seated) are made static in the masks; needs the LAS.
  bool static_masks = true;
  int max_size = 1600, boundary = 6;
  int cache_mb = 2048;
  fs::path mask_dir;
  // moge (default): MoGe-2 faces of the photo scaled to the LAS projection, with the geometry
  // engine of the geometry_models package; las: maps projected from the LAS.
  std::string source = "moge";
  fs::path geometry_models;
  // Cropped delivery: each lens's maps cover this window of its original photos, and the
  // original-size static and exclusion masks are cut to it.
  std::optional<std::array<CropWindow, 2>> crop;
};
SupervisionOptions ParseSupervisionOptions(const Arguments& args);
// Settles the MoGe model package once the command knows its models (given, recorded by the work
// directory, or the default) and checks that it holds the geometry engine.
void ResolveSupervisionModels(SupervisionOptions& options, const fs::path& package);
// Supervision-map camera: the longest side reduced to max_size, never enlarged.
Camera MapCamera(const Camera& source, int max_size);
// masks is the directory of the selected model's static masks, named by image index.
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& masks,
                        const SupervisionOptions& options, bool rolling = false);
// With motion, each pixel uses the camera of its own readout line (original rolling-shutter
// photos); its normal is expressed in that camera.
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model, const fs::path& masks,
                          const fs::path& output, const SupervisionOptions& options,
                          const ShutterMotion* motion = nullptr);
// Projects the LAS into every registered image and hands over its raw blended ray-distance map
// (float32, metres, 0 where no LAS record landed) on the MapCamera(max_size) grid, with no mask,
// hole filling or occlusion handling. scratch holds the temporary LAS coordinate cache.
void ProjectLasDepths(const fs::path& las, const LasHeader& header, const Reconstruction& model,
                      const ShutterMotion* motion, int max_size, int cache_mb,
                      const fs::path& scratch,
                      const std::function<void(uint32_t, cv::Mat&&)>& consume);
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary);
}  // namespace geosun

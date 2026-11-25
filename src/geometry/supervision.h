#pragma once
#include "geometry/model.h"
#include <memory>
#include <optional>

namespace geosun {
struct SupervisionOptions {
  bool enabled = true;
  // Persistent people (pictures, posters, seated) are made static in the masks; needs the LAS.
  bool static_masks = true;
  int max_size = 1024, boundary = 4;
  int cache_mb = 4096;
  fs::path mask_dir;
  fs::path geometry_models;
  // Cropped delivery: each lens's maps cover this window of its original photos, and the
  // original-size static and exclusion masks are cut to it.
  std::optional<std::array<CropWindow, 2>> crop;
};

class LasSurfelCache {
 public:
  struct Implementation;

  LasSurfelCache(const fs::path& las, const LasHeader& header, const fs::path& scratch,
                 int cache_mb, const Reconstruction* model = nullptr);
  ~LasSurfelCache();
  LasSurfelCache(const LasSurfelCache&) = delete;
  LasSurfelCache& operator=(const LasSurfelCache&) = delete;

  void Finish();
  Implementation& Impl() { return *impl_; }

 private:
  std::unique_ptr<Implementation> impl_;
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
// Exports pure MoGe-2 SSI depth and OpenCV camera-frame normal maps. With motion, faces are
// rectified along omega and fused normals are rotated back to each pixel's readout-line camera.
Json ExportSupervision(const Scene& scene, const Reconstruction& model, const fs::path& masks,
                       const fs::path& output, const SupervisionOptions& options,
                       const ShutterMotion* motion = nullptr,
                       LasSurfelCache* surfel_cache = nullptr);
// Projects the LAS into every registered image and hands over its single-pass nearest tangent-disk
// ray-distance map (float32, metres, 0 where no LAS record landed) on the MapCamera(max_size) grid,
// with no mask, hole filling or occlusion handling.
void ProjectLasDepths(LasSurfelCache& cache, const LasHeader& header, const Reconstruction& model,
                      const ShutterMotion* motion, int max_size,
                      const std::function<void(uint32_t, cv::Mat&&)>& consume);
void CheckSupervision(const fs::path& output, const Reconstruction& model, const Json& photos,
                      const Json& summary);
}  // namespace geosun

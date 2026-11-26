#pragma once
#include "geometry/model.h"
#include <functional>

namespace geosun {
class LasSurfelCache;

// Re-decides the transient (person) components, sky detections, and single-view photometric
// outliers (moving shadows and specular glare) of the masks using the LAS projection:
// - A transient component is located in the world (median LAS point) and made static again when
//   flagged in a sizeable share of views over a long time (picture, poster, seated person).
// - A sky (128) component whose eroded core has dense LAS surface depth is a false positive
//   (ceiling, lightbox, display panel) and returns to static (255); if the scene as a whole lacks
//   open sky after removing surface components, remaining indoor sky false positives are cleared.
// - Jointly calibrates per-lens 3-channel KB8 radial vignetting, per-frame relative exposure EVs,
//   and temporally-smoothed 3-channel white-balance gains via Schur-complement Huber IRLS over
//   fused LAS surface tracks and SfM tracks (with 10% holdout validation), plans headroom-safe
//   homogenization, and masks single-view moving shadows, specular glare, and closed-loop
//   overexposure/post-clipping outliers to transient (0).
// Masks are read from source(image) and written, for the registered images, to destination(image).
// Returns the report with summary statistics and one row per transient candidate.
Json StaticMasks(const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
                 const LasHeader& header, LasSurfelCache& cache,
                 const std::function<fs::path(uint32_t)>& source,
                 const std::function<fs::path(uint32_t)>& destination);
// The decision rules, recorded beside the masks.
Json StaticMaskRules();
}  // namespace geosun

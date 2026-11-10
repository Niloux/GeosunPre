#pragma once
#include "geometry/model.h"
#include <functional>

namespace geosun {
// Re-decides the transient (person) components of the masks by persistence: where a candidate is
// in the world (median LAS point), and how many of the views that see that place flagged it
// transient, over how many seconds. Flagged in a sizeable share of the views over a long time - a
// picture of people, a poster, a person seated all along - it is static again; a walking person,
// their shadow or the operator is flagged only where and while they are. Masks are read from
// source(image) and written, for the registered images, to destination(image); scratch holds the
// temporary LAS coordinate cache. Returns the report with one row per candidate.
Json StaticMasks(const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
                 const LasHeader& header, const fs::path& las,
                 const std::function<fs::path(uint32_t)>& source,
                 const std::function<fs::path(uint32_t)>& destination, const fs::path& scratch);
// The decision rules, recorded beside the masks.
Json StaticMaskRules();
}  // namespace geosun

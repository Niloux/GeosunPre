#pragma once
#include "io/scene.h"
#include <set>

namespace geosun {
Json MakeMasks(const Scene& scene, const fs::path& work, const fs::path& models);
Json MakeFeatures(const Scene& scene, const fs::path& work, const fs::path& models);
Json MakeMatches(const Scene& scene, const fs::path& work, const fs::path& models,
                 const std::set<uint32_t>& repair = {});
fs::path MaskPath(const fs::path& work, uint32_t image);
}  // namespace geosun

#pragma once
#include "app/runtime.h"
#include <cstdint>

namespace geosun {

struct AdaptiveTrainerInput {
  size_t registered_images = 0;
  uint64_t geom_points = 0;
  double vram_gb = 0.0;
  fs::path template_path;
};

struct AdaptiveTrainerResult {
  Json config;
  Json summary;
  uint64_t ply_cap = 2500000;
};

double QueryGpuVramGiB();
Json LoadTrainerTemplate(const fs::path& template_path = {});
AdaptiveTrainerResult BuildAdaptiveTrainerConfig(const AdaptiveTrainerInput& input);
Json WriteTrainerConfigForBundle(const fs::path& output, double vram_gb = 0.0,
                                 const fs::path& template_path = {});

}  // namespace geosun

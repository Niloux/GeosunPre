#include "io/trainer_config.h"
#include "frontend/inference.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace geosun {
// Compiled from configs/trainer.json.
extern const char DefaultTrainerConfig[];

namespace {
uint64_t RoundToNearest(double value, uint64_t step) {
  return static_cast<uint64_t>(std::llround(value / static_cast<double>(step))) * step;
}
}  // namespace

double QueryGpuVramGiB() {
  const auto gpu = GpuInfo();
  const double total_bytes = gpu.at("memory_total").get<double>();
  if (!std::isfinite(total_bytes) || total_bytes <= 0.0)
    throw std::runtime_error("Invalid GPU total memory reported by CUDA");
  return total_bytes / (1024.0 * 1024.0 * 1024.0);
}

Json LoadTrainerTemplate(const fs::path& template_path) {
  Json config =
      template_path.empty() ? Json::parse(DefaultTrainerConfig) : ReadJson(template_path);
  if (!config.is_object() || !config.contains("optimization") ||
      !config.at("optimization").is_object() || !config.contains("dataset") ||
      !config.at("dataset").is_object()) {
    throw std::runtime_error("Trainer config template must contain optimization and dataset objects");
  }
  const auto& opt = config.at("optimization");
  if (!opt.contains("iterations") || !opt.at("iterations").is_number_integer() ||
      opt.at("iterations").get<int64_t>() < 1) {
    throw std::runtime_error("Trainer config template must define positive integer optimization.iterations");
  }
  return config;
}

AdaptiveTrainerResult BuildAdaptiveTrainerConfig(const AdaptiveTrainerInput& input) {
  if (input.registered_images == 0)
    throw std::runtime_error("Adaptive trainer config requires at least one registered image");
  if (!std::isfinite(input.vram_gb) || input.vram_gb <= 0.0)
    throw std::runtime_error("Adaptive trainer config requires positive finite VRAM GiB");

  Json config = LoadTrainerTemplate(input.template_path);
  auto& opt = config.at("optimization");
  const auto& ds = config.at("dataset");

  const bool eval_all = opt.value("eval_all", false);
  const int test_every = ds.value("test_every", 8);
  size_t train_images = input.registered_images;
  if (!eval_all && test_every > 1 && input.registered_images > 1) {
    const size_t step = static_cast<size_t>(test_every);
    const size_t test_count = (input.registered_images + step - 1) / step;
    train_images = std::max<size_t>(1, input.registered_images - test_count);
  }

  const uint64_t base_iters = opt.at("iterations").get<uint64_t>();
  const double raw_iters =
      static_cast<double>(base_iters) * std::pow(static_cast<double>(train_images) / 550.0, 0.6);
  const uint64_t target_iters = RoundToNearest(std::clamp(raw_iters, 20000.0, 120000.0), 1000ULL);
  const double ratio = static_cast<double>(target_iters) / static_cast<double>(base_iters);

  auto scale_step = [&](uint64_t value) -> uint64_t {
    if (value == base_iters)
      return target_iters;
    const long long scaled = std::llround(static_cast<double>(value) * ratio);
    return scaled > 0 ? static_cast<uint64_t>(scaled) : 0ULL;
  };
  auto scale_field = [&](const char* key, bool clamp_to_one) {
    if (!opt.contains(key))
      return;
    const int64_t raw = opt.at(key).get<int64_t>();
    if (raw < 0)
      return;
    uint64_t scaled = scale_step(static_cast<uint64_t>(raw));
    if (clamp_to_one)
      scaled = std::max<uint64_t>(1ULL, scaled);
    opt[key] = scaled;
  };
  auto scale_list = [&](const char* key) {
    if (!opt.contains(key) || !opt.at(key).is_array())
      return;
    std::set<uint64_t> unique;
    for (const auto& item : opt.at(key)) {
      if (const uint64_t scaled = scale_step(item.get<uint64_t>()); scaled > 0)
        unique.insert(scaled);
    }
    Json out = Json::array();
    for (const uint64_t step : unique)
      out.push_back(step);
    opt[key] = std::move(out);
  };

  opt["iterations"] = target_iters;
  scale_field("start_refine", false);
  scale_field("stop_refine", false);
  scale_field("reset_every", true);
  scale_field("refine_every", true);
  scale_field("morton_reorder_interval", false);
  scale_field("sh_degree_interval", true);
  scale_field("grow_until_iter", false);
  scale_field("color_variance_warmup_iters", false);
  scale_field("exposure_correction_grid_start_iter", false);
  scale_list("eval_steps");
  scale_list("save_steps");
  opt["steps_scaler"] = 1.0;
  opt["image_count_scaler"] = 1.0;
  opt["image_count_scaler_total"] = 1.0;

  const uint64_t cap_geom =
      input.geom_points > 0
          ? static_cast<uint64_t>(std::llround(3.0 * static_cast<double>(input.geom_points)))
          : std::numeric_limits<uint64_t>::max();
  const uint64_t cap_views = static_cast<uint64_t>(train_images) * 8000ULL;
  const uint64_t cap_vram = RoundToNearest(
      std::clamp((input.vram_gb - 2.5) * 1800000.0, 4000000.0, 24000000.0), 100000ULL);
  const uint64_t demand = std::min(cap_geom, cap_views);
  const uint64_t max_cap = RoundToNearest(
      std::clamp(static_cast<double>(demand), 1500000.0, static_cast<double>(cap_vram)),
      100000ULL);
  opt["max_cap"] = max_cap;

  const uint64_t ply_cap = std::min<uint64_t>(
      2500000ULL,
      std::max<uint64_t>(100000ULL,
                         static_cast<uint64_t>(std::llround(0.75 * static_cast<double>(max_cap)))));

  Json summary = {
      {"vram_gb", input.vram_gb},
      {"registered_images", input.registered_images},
      {"train_images", train_images},
      {"geom_points", input.geom_points},
      {"iterations", target_iters},
      {"schedule_scale", ratio},
      {"max_cap", max_cap},
      {"cap_geom", input.geom_points > 0 ? Json(cap_geom) : Json(nullptr)},
      {"cap_views", cap_views},
      {"cap_vram", cap_vram},
      {"ply_cap", ply_cap},
  };
  return {std::move(config), std::move(summary), ply_cap};
}

Json WriteTrainerConfigForBundle(const fs::path& output, double vram_gb,
                                 const fs::path& template_path) {
  const auto manifest = ReadJson(output / "manifest.json");
  const size_t registered_images =
      manifest.at("registration").at("registered_images").get<size_t>();
  uint64_t geom_points = 0;
  if (manifest.contains("initialization") && !manifest.at("initialization").is_null()) {
    const auto& init = manifest.at("initialization");
    if (init.contains("roi_voxel_points_before_cap"))
      geom_points = init.at("roi_voxel_points_before_cap").get<uint64_t>();
    else if (init.contains("voxel_points_before_cap"))
      geom_points = init.at("voxel_points_before_cap").get<uint64_t>();
    else if (init.contains("points"))
      geom_points = init.at("points").get<uint64_t>();
  }
  const double resolved_vram = vram_gb > 0.0 ? vram_gb : QueryGpuVramGiB();
  const auto built =
      BuildAdaptiveTrainerConfig({registered_images, geom_points, resolved_vram, template_path});
  const auto config_path = output / "trainer.json";
  WriteJson(config_path, built.config);
  Json report = built.summary;
  report["file"] = config_path.u8string();
  return report;
}

}  // namespace geosun

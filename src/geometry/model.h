#pragma once
#include "io/scene.h"

namespace geosun {
constexpr uint64_t NoPoint = UINT64_MAX;
struct Observation {
  uint32_t image, feature;
};
struct ModelImage {
  uint32_t camera = 0;
  Pose pose;
  bool registered = false;
  std::vector<Eigen::Vector2d> pixels;
  std::vector<uint64_t> points;
};
struct ModelPoint {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  std::array<uint8_t, 3> color{127, 127, 127};
  double error = 0;
  std::vector<Observation> track;
};
struct Reconstruction {
  std::array<Camera, 2> cameras;
  std::vector<ModelImage> images;
  std::vector<ModelPoint> points;
  uint32_t gauge_image = 0, scale_image = 1;
  double gauge_baseline = 1;
  size_t Registered() const;
  size_t Observations() const;
  void Apply(const Similarity& transform);
  void RebuildReferences();
  void Filter(double max_error = 6, double min_angle_degrees = .5);
  Json Quality(const Scene& scene, double median_limit, double p90_limit) const;
};
void WriteModel(const fs::path& path, const Reconstruction& model);
Reconstruction ReadModel(const fs::path& path);
}  // namespace geosun

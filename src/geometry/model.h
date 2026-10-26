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
// Rolling-shutter motion of every image. The sensor reads out along one image axis, rows (y) or
// columns (x); a pixel at coordinate c on that axis was exposed at the readout fraction
// s = (c - size / 2) / size from the reference (middle-line) pose. Its camera point is
// (I + s [rotation]x) (v - s readout velocity), where v is the reference camera point, rotation the
// camera-frame rotation over one full readout and velocity the camera-frame velocity in m/s.
// The readout time is shared by each lens; its sign carries the line order.
struct ShutterMotion {
  std::vector<Eigen::Vector3d> rotation, velocity;
  std::array<double, 2> readout{};
  int axis = 1;  // 1: rows, 0: columns
  double Fraction(const Camera& camera, const Eigen::Vector2d& pixel) const {
    const double size = axis ? camera.height : camera.width;
    return (pixel[axis] - size / 2) / size;
  }
  // Camera point of reference camera point v for an observation at readout fraction s.
  Eigen::Vector3d Apply(uint32_t image, uint32_t camera, double s, const Eigen::Vector3d& v) const {
    const Eigen::Vector3d b = v - s * readout[camera] * velocity[image];
    return b + s * rotation[image].cross(b);
  }
  // Inverse of Apply for the same readout fraction.
  Eigen::Vector3d Reference(uint32_t image, uint32_t camera, double s,
                            const Eigen::Vector3d& moved) const {
    Eigen::Matrix3d a = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d w = s * rotation[image];
    a << 1, -w.z(), w.y(), w.z(), 1, -w.x(), -w.y(), w.x(), 1;
    return a.inverse() * moved + s * readout[camera] * velocity[image];
  }
};
// Pixel window of a lens's photos; see ValidDomainCrop.
struct CropWindow {
  int x = 0, y = 0, width = 0, height = 0;
  Json ToJson() const { return {{"x", x}, {"y", y}, {"width", width}, {"height", height}}; }
};
struct Reconstruction {
  std::array<Camera, 2> cameras;
  std::vector<ModelImage> images;
  std::vector<ModelPoint> points;
  uint32_t gauge_image = 0, scale_image = 1;
  double gauge_baseline = 1;
  size_t Registered() const;
  size_t Observations() const;
  Json Registration(const Scene& scene) const;
  void Apply(const Similarity& transform);
  void RebuildReferences();
  void Filter(double max_error = 6, double min_angle_degrees = .5);
  // Reprojection uses the rolling-shutter projection when motion is given.
  Json Quality(const Scene& scene, double median_limit, double p90_limit,
               const ShutterMotion* motion = nullptr) const;
};
void WriteModel(const fs::path& path, const Reconstruction& model);
Reconstruction ReadModel(const fs::path& path);
}  // namespace geosun

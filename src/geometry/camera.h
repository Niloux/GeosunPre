#pragma once
#include <array>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "app/runtime.h"

namespace geosun {
constexpr double Pi = 3.14159265358979323846;
// Valid fisheye domain shared by masks, features and supervision.
constexpr double FieldOfViewDegrees = 160;
// Per-image support required for registration, calibration and visual acceptance.
constexpr size_t MinimumImageObservations = 32, MinimumImageCells = 8;
struct Camera {
  std::string label;
  int width = 0, height = 0;
  std::array<double, 8> params{};
  std::array<double, 2> principal_offset{};
  static Camera FromJson(const Json& value);
  Json ToJson() const;
  double Radius(double fov = FieldOfViewDegrees) const;
  bool Valid(double x, double y, double fov = FieldOfViewDegrees) const;
  cv::Mat Mask(double fov = FieldOfViewDegrees) const;
  Eigen::Vector2d Project(const Eigen::Vector3d& xyz) const;
  Eigen::Matrix<double, 2, 3> ProjectionJacobian(const Eigen::Vector3d& xyz) const;
  Eigen::Vector3d Ray(const Eigen::Vector2d& pixel) const;
  // Largest change from the initial calibration allowed in BA and checked by acceptance.
  std::array<double, 8> CalibrationBounds() const;
  // Cell index in the 8x8 image grid counted against MinimumImageCells.
  int CoverageCell(const Eigen::Vector2d& pixel) const;
};
struct Pose {
  // World-to-camera rotation in Eigen order x,y,z,w, followed by translation.
  std::array<double, 7> params{0, 0, 0, 1, 0, 0, 0};
  Eigen::Quaterniond Rotation() const;
  Eigen::Vector3d Translation() const;
  Eigen::Vector3d Center() const;
  Eigen::Vector3d Apply(const Eigen::Vector3d& point) const;
  Eigen::Vector3d operator*(const Eigen::Vector3d& point) const { return Apply(point); }
  void Set(const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation);
};
struct Similarity {
  double scale = 1;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d Apply(const Eigen::Vector3d& point) const;
  Eigen::Vector3d operator*(const Eigen::Vector3d& point) const { return Apply(point); }
  Similarity operator*(const Similarity& other) const {
    return {scale * other.scale, rotation * other.rotation, Apply(other.translation)};
  }
  Json ToJson() const;
  static Similarity FromJson(const Json& value);
};
Similarity FitSimilarity(const std::vector<Eigen::Vector3d>& source,
                         const std::vector<Eigen::Vector3d>& target);
double Quantile(std::vector<double> values, double fraction);
Json Statistics(const std::vector<double>& values);
}  // namespace geosun

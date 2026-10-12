#pragma once
#include <array>
#include <opencv2/core.hpp>

#include "runtime.h"

struct Camera {
  std::string label;
  int width = 0, height = 0;
  std::array<double, 8> params{};
  std::array<double, 2> principal_offset{};
  static Camera FromJson(const Json& value);
  Json ToJson() const;
  cv::Matx33d K() const;
  cv::Vec4d D() const;
  double Radius(double fov = 160.) const;
  bool Valid(double x, double y, double fov = 160.) const;
  cv::Mat Mask(double fov = 160.) const;
  std::vector<cv::Point2d> Project(const std::vector<cv::Point3d>& xyz) const;
  std::vector<cv::Point3d> Rays(const std::vector<cv::Point2d>& pixels) const;
};
cv::Mat ReadImage(const fs::path& path, int mode);
void WriteImage(const fs::path& path, const cv::Mat& image);
cv::Mat BicubicResize(const cv::Mat& image, int width, int height);
Json PrepareScene(const fs::path& input, const fs::path& work);
Json ImageHeader(const fs::path& path);

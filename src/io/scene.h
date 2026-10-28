#pragma once
#include "geometry/camera.h"

namespace geosun {
struct Exposure {
  uint32_t id = 0, camera = 0;
  std::string name;
  fs::path source;
  int frame = 0, week = 0;
  double seconds = 0, pose_seconds = 0;
  Eigen::Vector3d navigation_raw;
  Json source_file;
};
struct Scene {
  fs::path input, project, las;
  std::array<Camera, 2> cameras;
  std::vector<Exposure> images;
  Json sources;
  bool enhanced = false;  // Image/enhanced copies replace same-name originals
  Json ToJson() const;
  static Scene FromJson(const Json& value);
};
Scene ReadScene(const fs::path& input, const fs::path& las = {}, bool enhanced = false);
cv::Mat ReadImage(const fs::path& path, int mode);
void WriteImage(const fs::path& path, const cv::Mat& image);
struct ResizeCoefficients {
  int start;
  std::vector<int32_t> values;
};
std::vector<ResizeCoefficients> ResizeWeights(int input, int output);
Json ImageHeader(const fs::path& path);
std::vector<Eigen::Vector3d> NavigationPositions(const Scene& scene);
// Every navigation row as (GPS seconds of week, position in the LAS frame), in time order.
std::vector<std::pair<double, Eigen::Vector3d>> NavigationTrack(const Scene& scene);
// Trajectory velocity in the LAS frame (m/s) at every exposure, from +-50 ms of navigation.
std::vector<Eigen::Vector3d> NavigationVelocities(const Scene& scene);
// (L, R) image indices of each frame number exposed by both lenses when every such pair was
// exposed within 2 ms; empty when the lenses are not synchronized.
std::vector<std::array<uint32_t, 2>> SynchronousFrames(const Scene& scene);
}  // namespace geosun

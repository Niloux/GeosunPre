#pragma once
#include "frontend/preprocessing.h"
#include "geometry/camera.h"

namespace geosun {
// MoGe-2 runs on square pinhole faces of a fisheye lens: a front face and, for a wide lens, a
// ring of tilted faces. Each face is estimated on its own and gathered onto the supervision grid.
struct MonocularFace {
  Eigen::Matrix3d rotation;  // lens camera ray -> face frame
  cv::Mat source;            // CV_32FC2 face pixel -> prefiltered photo pixel
  cv::Mat valid;             // CV_8U, 255 where the face pixel sees the lens's valid domain
};
// One face covering a grid pixel, at face pixel coordinates (pixel centres at integers).
struct FaceSample {
  float x = 0, y = 0, weight = 0;
  uint8_t face = 0;
};
struct MonocularLens {
  static constexpr int MaxSamples = 3;
  Camera source, grid;
  int size = 0;
  double focal = 0, tilt_degrees = 0, prefilter = 1;
  std::vector<MonocularFace> faces;
  // Up to MaxSamples faces per grid pixel by descending weight; count 0 outside the domain.
  std::vector<std::array<FaceSample, MaxSamples>> samples;
  std::vector<uint8_t> counts;
  Json ToJson() const;
};
// Faces of size x size pixels and 90 degree field of view covering every valid ray of the grid
// camera with at least 5 degrees to a face edge; source is the photo calibration.
MonocularLens PlanMonocularLens(const Camera& source, const Camera& grid, int size);
struct FaceEstimate {
  cv::Mat distance;  // CV_32F distance along the lens ray in MoGe's metric scale, 0 where invalid
  cv::Mat normal;    // CV_32FC3 unit normal in the lens camera frame, facing the camera
  double shift = 0, metric_scale = 0;
  size_t valid = 0;
  Json ToJson() const;
};
class MonocularEstimator {
  Models& models_;
  ImagePreprocessor pixels_;
  int size_;

 public:
  explicit MonocularEstimator(Models& models);
  int FaceSize() const { return size_; }
  // rgb is the full photo of the lens; faces[i].source addresses it after prefiltering.
  std::vector<FaceEstimate> Estimate(const MonocularLens& lens, const cv::Mat& rgb);
  // The prefiltered faces as RGB8, for inspection.
  static std::vector<cv::Mat> Faces(const MonocularLens& lens, const cv::Mat& rgb);
};
// Per-face scales making overlapping faces agree on the shared rays, face 0 (front) fixed at 1:
// least squares over the median log distance ratio of every overlapping pair.
struct FaceAlignment {
  std::vector<double> scales;
  Json pairs;  // per pair: pixels, median and MAD of the log ratio, residual after alignment
};
FaceAlignment AlignFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces);
struct MonocularMaps {
  cv::Mat distance;  // CV_32F on the grid camera, 0 where no face holds a valid estimate
  cv::Mat normal;    // CV_32FC3 unit normal, zero where invalid
};
// Weighted blend in log distance; scales[i] multiplies face i. A face contributes to a grid pixel
// only where all four bilinear neighbours of its sample are valid.
MonocularMaps GatherFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                          const std::vector<double>& scales);
// Writes the faces and gathered maps of one photo into a new directory for inspection.
Json CheckMonocular(const fs::path& input, const fs::path& models, const std::string& image,
                    const fs::path& output, int max_size);
}  // namespace geosun

#pragma once
#include "frontend/preprocessing.h"
#include "geometry/camera.h"
#include <functional>

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
  cv::Mat rays;  // CV_32FC3 unit grid ray per pixel, zero outside the valid domain
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
  // images are the RGB8 faces of one photo from Faces.
  std::vector<FaceEstimate> Estimate(const MonocularLens& lens, const std::vector<cv::Mat>& images);
  // rgb is the full photo of the lens; faces[i].source addresses it after prefiltering. Runs on
  // the CPU and may be called from another thread.
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
// Weighted blend in log distance; log_scale(sample) is added to the log distance of that face
// sample. A face contributes to a grid pixel only where all four bilinear neighbours of its sample
// are valid.
MonocularMaps GatherFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                          const std::function<double(const FaceSample&)>& log_scale);
// MoGe's depth errors vary within a face (near objects against the room behind them), so every
// face gets a smooth log-scale field on a Cells x Cells grid of bilinear cells, fitted to LAS ray
// distances (las: CV_16U millimetres on the grid, 0 where invalid) together with the face
// overlaps. On a 2-pixel lattice of the grid each covering face contributes one LAS residual and
// each pair of covering faces one overlap residual; neighbouring nodes are tied by a first
// difference. The iteratively reweighted Cauchy least squares in log distance starts from one
// scale per face. The LAS sets the low frequencies of the depth; MoGe keeps its detail.
struct FaceFit {
  static constexpr int Cells = 12;
  std::vector<double> scales;  // exp of each field's mean, for reports
  std::vector<std::vector<double>> fields;  // (Cells + 1)^2 log-scale nodes per face, row-major
  Json faces;  // per face: LAS anchors, inlier fraction, residual median and MAD, field range
  size_t anchors = 0, inliers = 0;
  bool accepted = false;  // enough agreeing LAS anchors to give the photo metric depth
  double LogScale(const FaceSample& sample, int size) const;
  Json ToJson() const;
};
FaceFit FitFaceScales(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                      const cv::Mat& las);
// Supervision maps of one photo in the LAS supervision encoding: uint16 millimetre ray distance
// and RGB8 normals (B, G, R = z, y, x), zero where invalid. allowed (CV_8U, nonzero valid) holds
// the static mask with its semantic margin removed. Both maps drop MoGe's blended depth steps;
// depth further needs an accepted fit, the 0.05-65.535 m range and LAS agreement where LAS holds
// depth, normals an incidence of at least min_incidence. Returns per-photo statistics.
Json EncodeMonocular(const MonocularLens& lens, const MonocularMaps& maps, const cv::Mat& las,
                     const cv::Mat& allowed, bool depth, double min_incidence, cv::Mat& depth_mm,
                     cv::Mat& normal_rgb);
// Writes the faces and gathered maps of one photo into a new directory for inspection.
Json CheckMonocular(const fs::path& input, const fs::path& models, const std::string& image,
                    const fs::path& output, int max_size);
}  // namespace geosun

#pragma once
#include "frontend/inference.h"
#include "geometry/camera.h"
#include "geometry/model.h"

namespace geosun {
// MoGe-2 runs on square pinhole faces of a fisheye lens: a front face and, for a wide lens, rings
// of tilted faces. Each face is estimated on its own and gathered onto the supervision grid.
struct MonocularFace {
  Eigen::Matrix3d rotation;  // lens camera ray -> face frame
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
  double half_degrees = 0, focal = 0, prefilter = 1;
  int prefilter_width = 0, prefilter_height = 0;
  double prefilter_sx = 1, prefilter_sy = 1;
  std::vector<double> ring_tilts;  // degrees
  std::vector<MonocularFace> faces;
  // Up to MaxSamples faces per grid pixel by descending weight; count 0 outside the domain.
  std::vector<std::array<FaceSample, MaxSamples>> samples;
  std::vector<uint8_t> counts;
  cv::Mat rays;  // CV_32FC3 unit grid ray per pixel, zero outside the valid domain
  cv::Mat Prefilter(const cv::Mat& rgb) const;
  int SamplesAtRay(const Eigen::Vector3d& ray_ref,
                   std::array<FaceSample, MaxSamples>& out) const;
  Json ToJson() const;
};
// Half field of view of the faces covering the lens: wide enough for scene context and shared
// across depth and normals in one inference pass per face.
constexpr double FaceHalfDegrees = 45;
// Faces of size x size pixels and the given field of view covering every valid ray of the grid
// camera with at least 5 degrees to a face edge; source is the photo calibration.
MonocularLens PlanMonocularLens(const Camera& source, const Camera& grid, int size,
                                double half_degrees);
struct FaceEstimate {
  cv::Mat distance;  // CV_32F distance along the lens ray in MoGe's metric scale, 0 where invalid
  cv::Mat normal;    // CV_32FC3 unit normal in the lens camera frame, facing the camera
  double shift = 0, metric_scale = 0;
  size_t valid = 0;
  Json ToJson() const;
};
class MonocularEstimator {
  struct Workspace;
  Models& models_;
  int size_;
  std::unique_ptr<Workspace> workspace_;

 public:
  explicit MonocularEstimator(Models& models);
  ~MonocularEstimator();
  int FaceSize() const { return size_; }
  // prefiltered is the RGB8 photo produced by lens.Prefilter(rgb); GPU resampling rotates faces
  // along omega when rolling is true.
  std::vector<FaceEstimate> Estimate(const MonocularLens& lens, const cv::Mat& prefiltered,
                                     const Eigen::Vector3d& omega = Eigen::Vector3d::Zero(),
                                     int axis = 1, bool rolling = false);
};
// Decouples metric LAS depth from MoGe-2 dense normals on the supervision grid:
// 1. Detects depth/normal edges on each 90-degree MoGe face and dilates them on the grid to
//    mask LAS edge bleeding and calibration micro-offsets.
// 2. Gathers rolling-shutter rectified MoGe normals onto the grid (rotating back to readout-line
//    camera frame when omega is nonzero) and clears severe MoGe normal hallucinations where
//    confident planar LAS surfels disagree by > 35 degrees off MoGe edges.
// 3. Runs per-face deterministic RANSAC on inverse depth (q_las ~= s * q_moge + t) over clean
//    lattice anchors to filter close flying points and far glass/mirror penetrations, checking
//    multi-face consensus and rescuing solid planar surfaces whose LAS normal agrees with MoGe.
// 4. Keeps raw millimetre LAS depth for inliers (without warping) and MoGe normals inside allowed
//    pixels passing min_incidence (zeroing normals only on unrescued glass penetrations or
//    severe planar normal disagreements).
Json CleanLasAndEncodeNormals(const MonocularLens& lens,
                              const std::vector<FaceEstimate>& faces,
                              const cv::Mat& las,
                              const cv::Mat& las_normal,
                              const cv::Mat& allowed,
                              const Eigen::Vector3d* omega,
                              int axis,
                              double min_incidence,
                              cv::Mat& depth_mm,
                              cv::Mat& normal_rgb);
}  // namespace geosun

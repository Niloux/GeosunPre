#pragma once
#include "frontend/inference.h"
#include "geometry/camera.h"
#include "geometry/model.h"
#include <functional>

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
struct MonocularMaps {
  cv::Mat distance;  // CV_32F on the grid camera, 0 where no face holds a valid estimate
  cv::Mat normal;    // CV_32FC3 unit normal, zero where invalid
};
// Weighted blend in log distance; log_scale(sample) is added to the log distance of that face
// sample. A face contributes to a grid pixel only where all four bilinear neighbours of its sample
// are valid. When omega is nonzero, pixels un-rotate to s = 0 for lookup and normals rotate back
// into the readout-line camera frame.
MonocularMaps GatherFaces(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                          const std::function<double(const FaceSample&)>& log_scale,
                          const Eigen::Vector3d* omega = nullptr, int axis = 1);
// MoGe's depth errors vary within a face (near objects against the room behind them), so every
// face gets a smooth log-scale field on a Cells x Cells grid of bilinear cells, fitted to LAS ray
// distances (las: CV_16U millimetres on the grid, 0 where invalid) together with the face
// overlaps. On a 2-pixel lattice of the grid each covering face contributes one LAS residual where
// the LAS is trusted (las_normal, the LAS normal encoding, within 15 degrees of the face's MoGe
// normal) and each pair of covering faces one overlap residual; neighbouring nodes are tied by a
// first difference. The iteratively reweighted Cauchy least squares in log distance starts from one
// scale per face. The LAS sets the low frequencies of the depth; MoGe keeps its detail.
struct FaceFit {
  static constexpr int Cells = 12;
  std::vector<double> scales;  // exp of each field's mean, for reports
  std::vector<std::vector<double>> fields;  // (Cells + 1)^2 log-scale nodes per face, row-major
  Json faces;  // per face: LAS anchors, inlier fraction, residual median and MAD, field range
  size_t las_samples = 0;  // lattice LAS depths over valid face samples, trusted or not
  size_t anchors = 0, inliers = 0;  // trusted LAS residuals and those within 0.05
  bool accepted = false;  // enough agreeing LAS anchors to give the photo metric depth
  double LogScale(const FaceSample& sample, int size) const;
  Json ToJson() const;
};
FaceFit FitFaceScales(const MonocularLens& lens, const std::vector<FaceEstimate>& faces,
                      const cv::Mat& las, const cv::Mat& las_normal,
                      const Eigen::Vector3d* omega = nullptr, int axis = 1);
// Supervision maps of one photo in the LAS supervision encoding: uint16 millimetre ray distance
// and RGB8 normals (B, G, R = z, y, x) from maps on the grid, zero where invalid. allowed (CV_8U,
// nonzero valid) holds the static mask with its semantic margin removed. Depth drops MoGe's blended
// depth steps and needs an accepted fit, the 0.05-65.535 m range and agreement with the LAS where
// the LAS is trusted (as in FitFaceScales); untrusted LAS depth never vetoes. Normals need an
// incidence of at least min_incidence. Returns per-photo statistics.
Json EncodeMonocular(const MonocularLens& lens, const MonocularMaps& maps, const cv::Mat& las,
                     const cv::Mat& las_normal, const cv::Mat& allowed, bool depth,
                     double min_incidence, cv::Mat& depth_mm, cv::Mat& normal_rgb,
                     cv::Mat& verified);
// The 4 x 4 cells of the cross-view check holding any nonzero pixel (CV_8U in, CV_8U cells out).
cv::Mat ConflictCells(const cv::Mat& pixels);
// Free-space check between views of the delivered MoGe depth under output/depths. Each photo is
// paired with the views sharing most tracks (the relation made symmetric). On a 4-pixel lattice a
// depth point of one view that lies clearly in front of the depth the other view sees along the
// same ray contradicts one of the two maps, and both cells count the contradicting view. A cell
// whose depth the trusted LAS confirmed (verified, from ConflictCells, by image index) keeps it;
// any other loses it when at least two views contradict it. rows (by image index, null if
// unregistered) get the rewritten file size, depth count and the dropped pixels. Returns the
// parameters and totals.
Json FilterViewConflicts(const fs::path& output, const Scene& scene, const Reconstruction& model,
                         const ShutterMotion* motion, int max_size,
                         const std::vector<cv::Mat>& verified, std::vector<Json>& rows,
                         std::vector<cv::Mat>* held_depths = nullptr);
}  // namespace geosun

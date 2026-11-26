#pragma once
#include "geometry/model.h"
#include <array>
#include <vector>

namespace geosun {
// Precomputed per-lens geometry on the coarse photometric calibration grid (PhotoGridSize).
struct PhotometricCamGrid {
  Camera grid_cam;
  int width = 0, height = 0;
  double scale_x = 1.0, scale_y = 1.0;
  std::vector<Eigen::Vector3d> rays;
  std::vector<float> rho;  // normalized KB8 incidence angle theta / (80 deg)
  cv::Mat domain;
};

// Per-image coarse grid buffers; rgb_lin is strictly in RGB channel order (0=R, 1=G, 2=B) in
// un-devignetted IEC linear sRGB space.
struct PhotometricViewData {
  cv::Mat rgb_lin;       // CV_32FC3, RGB
  cv::Mat lum_srgb;      // CV_32FC1, blurred sRGB luminance
  cv::Mat max_srgb;      // CV_32FC1, unblurred max(R, G, B) in [0, 1]
  cv::Mat grad_srgb;     // CV_32FC1, sRGB gradient magnitude
  cv::Mat depth;         // CV_32FC1, cleaned LAS ray depth (m)
  cv::Mat depth_smooth;  // CV_8UC1
  cv::Mat static_mask;   // CV_8UC1
};

struct PhotometricCalibration {
  bool calibrated = false;
  bool wb_calibrated = false;
  // [camera 0..1][channel R/G/B][r^2, r^4, r^6] log2 vignetting polynomial coefficients:
  // V_{c,k}(rho) = exp2(c0*rho^2 + c1*rho^4 + c2*rho^6).
  std::array<std::array<std::array<double, 3>, 3>, 2> vig_coeffs_rgb{};
  std::array<int, 2> vig_order{3, 3};
  double reference_ev = 0.0;
  std::vector<double> exposure_ev;
  std::vector<double> pixel_ev_shift;
  std::vector<double> residual_ev;
  std::vector<uint8_t> exposure_valid;
  std::vector<std::array<double, 3>> wb_gains;  // RGB gains normalized to unit luminance
  Json report;
};

// Evaluates V_{c,k}(rho) = exp2(c0*r^2 + c1*r^4 + c2*r^6) clamped to [0.40, 1.05].
double EvaluateVignettingGain(const std::array<double, 3>& coeffs, int order, double rho);

// Solves per-lens 3-channel KB8 radial vignetting, per-frame relative exposure EV, and
// temporally-smoothed 3-channel white-balance gains via exact Schur-complement Huber IRLS over
// fused LAS surface tracks and verified SfM tracks, with 10% spatial-block holdout validation.
PhotometricCalibration SolvePhotometricCalibration(
    const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
    const std::vector<uint32_t>& written, const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& translations, const std::vector<Eigen::Vector3d>& centers,
    const std::vector<double>& seconds, const std::array<PhotometricCamGrid, 2>& grids,
    const std::vector<PhotometricViewData>& views, int workers);

// Plans headroom-safe pixel homogenization and runs a closed-loop forward check to flag any
// newly clipped or darkened-saturated regions into post_clip_masks.
void PlanSafeHomogenization(const Reconstruction& model, const std::vector<uint32_t>& written,
                            const std::array<PhotometricCamGrid, 2>& grids,
                            const std::vector<PhotometricViewData>& views,
                            const std::vector<cv::Mat>& clip_masks,
                            PhotometricCalibration& cal, std::vector<cv::Mat>& post_clip_masks);

// Decision constants recorded in static_masks.json for cache invalidation.
Json PhotometricRules();
}  // namespace geosun

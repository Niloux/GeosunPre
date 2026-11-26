#include "geometry/static_masks.h"
#include "geometry/supervision.h"
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace geosun {
namespace {
constexpr int MapSize = 1024, MinArea = 16, MinSamples = 12;
// Masks of all images are kept at 1/ThumbScale of the photo to look up what each view masked.
constexpr int ThumbScale = 6;
// A candidate is static when the place it covers is flagged transient in a sizeable share of the
// views that see that place, over a long time: a picture, poster or seated person is flagged in
// every view that really sees it, a walking person only while passing, the operator only where the
// rig is. Visibility is geometric only (no occlusion test: views from the next room count), which
// is why the share of a true picture is about 0.4 and the minimum is low.
constexpr double MinRatio = .2, MinSpan = 40;
constexpr int MinSeen = 8;
// Views see the place from 0.8 m to MaxRange within this share of the lens's valid radius.
constexpr double MaxRange = 12, ValidShare = .8;
// A static candidate this close (fraction of the image width) to one that stays masked stays
// masked: a shadow darkens the floor in every view the same way, so it must follow its person.
constexpr double ShadowReach = .08;
// Sky verification: erode each 1024-grid sky component so LAS edge spread on trees or roofs does
// not enter thin outdoor gaps, then check whether the interior has dense LAS depth returns.
constexpr int SkyCoreErosion = 2;
constexpr double SkyLasShare = .5, SkyImageMinShare = .005, SkySceneMinImageShare = .05;
constexpr size_t SkySceneMinImages = 8;
constexpr int SkyEdgeGuidedRadius4k = 12;
constexpr double SkyEdgeAlphaThreshold = .35;
constexpr int SkyCanopyPoreMaxArea4k = 2500;
constexpr double SkyCanopyFoliageBorderShare = .65;
// Multi-view photometric transient detection (moving shadows and specular glare).
constexpr int PhotoGridSize = 256;
constexpr int PhotoNeighbors = 20;
constexpr double PhotoMaxNeighborDistM = 10.0;
constexpr double PhotoMinTimeGapS = 1.5;
constexpr double PhotoMinDepthM = 0.35, PhotoMaxDepthM = 12.0, PhotoMaxDepthGrad = 0.35;
constexpr int PhotoMinGainSamples = 150, PhotoMinSeen = 6, PhotoGrowMinSeen = 4;
constexpr double PhotoMinPairGain = 0.50, PhotoMaxPairGain = 2.00;
constexpr double PhotoMinRefPerceptual = 0.12;
constexpr int PhotoMinCompArea = 25;
constexpr double PhotoMaxImageMaskShare = 0.20;
constexpr double PhotoDarkShare = 0.70, PhotoDarkRatio = 0.78, PhotoDarkDiff = 0.06;
constexpr double PhotoDarkMaxGrad = 0.20;
constexpr double PhotoDarkGrowShare = 0.65, PhotoDarkGrowRatio = 0.84, PhotoDarkGrowDiff = 0.04;
constexpr double PhotoDarkGrowMaxGrad = 0.25;
constexpr int PhotoDarkGrowRadius = 4, PhotoDarkDilateRadius = 1;
constexpr double PhotoBrightShare = 0.70, PhotoBrightRatio = 1.30, PhotoBrightDiff = 0.10;
constexpr double PhotoBrightMaxGrad = 0.25;
constexpr double PhotoBrightGrowShare = 0.65, PhotoBrightGrowRatio = 1.20, PhotoBrightGrowDiff = 0.07;
constexpr double PhotoBrightGrowMaxGrad = 0.30;
constexpr int PhotoBrightGrowRadius = 3, PhotoBrightDilateRadius = 1;
constexpr int PhotoPersonShadowReach = 24, PhotoPersonMinSeen = 5, PhotoPersonMinCompArea = 12;
constexpr double PhotoPersonDarkShare = 0.70, PhotoPersonDarkRatio = 0.88, PhotoPersonDarkDiff = 0.04;
constexpr double PhotoPersonDarkMaxGrad = 0.15;
// Global multi-view relative exposure, white balance, homogenization, and overexposure clipping.
constexpr int ExposureNeighbors = 25, ExposureMinSamples = 40, ExposureIters = 15;
constexpr double ExposureMinSrgb = 0.10, ExposureMaxSrgb = 0.78;
constexpr double ExposureMaxPairEv = 3.5, ExposureMaxPairMad = 0.45, ExposureHuberDeltaEv = 0.15;
constexpr double ExposureMinComponentShare = 0.90, ExposureMaxMedianResidualEv = 0.15;
constexpr double WbMinChSrgb = 0.04, WbMaxChSrgb = 0.84, WbPriorWeight = 0.5, WbMaxAbsEv = 0.35;
constexpr double WbMaxPairEv = 1.0, WbHuberDeltaEv = 0.08;
constexpr int WbIters = 5;
constexpr double HomogKdeBandwidthEv = 0.35, HomogMaxBrightenEv = 1.20, HomogMaxDarkenEv = 1.40;
constexpr double HomogMinP50Srgb = 0.25;
constexpr int PhotoClipNeighbors = 32, PhotoClipNearestQuota = 12;
constexpr double PhotoClipLargeEvDiff = 1.20, PhotoClipMaxNeighborDistM = 25.0;
constexpr double PhotoClipMaxDepthM = 30.0;
constexpr double PhotoClipMinLum = 0.92, PhotoClipMinMaxCh = 0.96, PhotoClipMinEvDiff = 0.35;
constexpr double PhotoClipRefMaxLum = 0.86, PhotoClipRefMaxCh = 0.92;
constexpr int PhotoClipMinSeen = 2, PhotoClipMinCompArea = 12;
constexpr double PhotoClipShare = 0.60, PhotoClipGrowLum = 0.88, PhotoClipGrowMaxCh = 0.94;
constexpr double PhotoClipHighEvBoost = 2.0, PhotoClipHighEvMinLum = 0.95;
constexpr double PhotoClipHighEvMinMaxCh = 0.985, PhotoClipHighEvMinDepthM = 7.0;
constexpr double PhotoClipBloomEvBoost = 1.5;
constexpr int PhotoClipBloomCloseRadius = 4, PhotoClipHighEvDilateRadius = 3;
constexpr int PhotoClipDilateRadius = 2;
constexpr double PhotoMaxClipMaskShare = 0.45;

inline float SrgbToLin(float c) {
  const float x = std::clamp(c, 0.0f, 1.0f);
  return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

struct Candidate {
  uint32_t image = 0;
  int label = 0, area = 0;
  double x = 0, y = 0;
  bool located = false;
  Eigen::Vector3d place = Eigen::Vector3d::Zero();
  size_t seen = 0, flagged = 0;
  double span = 0;
  std::string reason;
};
struct SkyComponent {
  int label = 0, area = 0, core_area = 0, core_las = 0;
  bool surface = false;
};
struct SkyImage {
  bool has_raw_sky = false;
  int valid_grid_pixels = 0;
  int open_sky_grid_pixels = 0;
  std::vector<SkyComponent> components;
};
struct PhotoCamGrid {
  Camera grid_cam;
  int width = 0, height = 0;
  double scale_x = 1.0, scale_y = 1.0;
  std::vector<Eigen::Vector3d> rays;
  std::vector<float> rho;
  cv::Mat domain;
};
struct PhotoView {
  cv::Mat lum_lin;
  cv::Mat rgb_lin;
  cv::Mat lum_srgb;
  cv::Mat max_srgb;
  cv::Mat grad_srgb;
  cv::Mat p_lin, p_min3, p_max3;
  cv::Mat depth;
  cv::Mat depth_smooth;
  cv::Mat static_mask;
  cv::Mat raw_sky_small;
  cv::Mat person_small;
  cv::Mat transient_mask;
  bool capped = false;
  bool clip_capped = false;
};
struct Context {
  const Scene& scene;
  const Reconstruction& model;
  const ShutterMotion* motion = nullptr;
  const std::function<fs::path(uint32_t)> &source, &destination;
  std::array<cv::Mat, 2> domains, full_domains;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<Eigen::Vector3d> translations, centers;
  std::vector<double> seconds;
  std::vector<cv::Mat> thumbs;  // transient share of each view, 1/ThumbScale
  std::vector<cv::Mat> las_interior;  // 1024-grid interior LAS returns for outdoor edge locking
  std::vector<std::vector<Candidate>> candidates;
  std::array<int, 2> full_domain_grid_pixels{0, 0};
  std::vector<SkyImage> skies;
  std::array<PhotoCamGrid, 2> photo_grids;
  std::vector<PhotoView> photo_views;
  std::array<std::array<double, 3>, 2> vig_coeffs{};
  double photo_seconds = 0.0;
  bool keep_sky = true;
  bool exposure_calibrated = false;
  double exposure_ref_ev = 0.0;
  std::vector<double> exposure_ev;
  std::vector<double> pixel_ev_shift;
  std::vector<double> residual_ev;
  std::vector<uint8_t> exposure_valid;
  Json exposure_report = Json::object();
  bool wb_calibrated = false;
  std::vector<std::array<double, 3>> wb_gains;
  Json wb_report = Json::object();
};
struct Labelled {
  cv::Mat mask, transient_map, labels, stats, centroids;
  int count = 0;
};
void CleanAndDownsampleLasDepth(const cv::Mat& depth_1024, int gw, int gh, cv::Mat& depth_out,
                                cv::Mat& smooth_out) {
  const int h1024 = depth_1024.rows, w1024 = depth_1024.cols;
  cv::Mat las_inf(h1024, w1024, CV_32FC1, cv::Scalar(1e9f));
  for (int y = 0; y < h1024; ++y) {
    const float* src = depth_1024.ptr<float>(y);
    float* dst = las_inf.ptr<float>(y);
    for (int x = 0; x < w1024; ++x) {
      const float d = src[x];
      if (std::isfinite(d) && d >= 0.20f && d <= 60.0f)
        dst[x] = d;
    }
  }
  cv::Mat las_min;
  cv::erode(las_inf, las_min, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
  cv::Mat clean(h1024, w1024, CV_32FC1, cv::Scalar(0.f));
  for (int y = 0; y < h1024; ++y) {
    const float* src = las_inf.ptr<float>(y);
    const float* mmin = las_min.ptr<float>(y);
    float* dst = clean.ptr<float>(y);
    for (int x = 0; x < w1024; ++x) {
      const float d = src[x];
      if (d < 1e8f && d <= mmin[x] * 1.08f + 0.08f)
        dst[x] = d;
    }
  }

  cv::Mat pooled(gh, gw, CV_32FC1, cv::Scalar(0.f));
  cv::Mat bimodal = cv::Mat::zeros(gh, gw, CV_8UC1);
  std::vector<float> block_vals;
  for (int gy = 0; gy < gh; ++gy) {
    const int y0 = (gy * h1024) / gh;
    const int y1 = std::max(y0 + 1, ((gy + 1) * h1024) / gh);
    float* dst = pooled.ptr<float>(gy);
    uint8_t* bm = bimodal.ptr<uint8_t>(gy);
    for (int gx = 0; gx < gw; ++gx) {
      const int x0 = (gx * w1024) / gw;
      const int x1 = std::max(x0 + 1, ((gx + 1) * w1024) / gw);
      block_vals.clear();
      float d_min = 1e9f, d_max = 0.f;
      for (int sy = y0; sy < y1; ++sy) {
        const float* row = clean.ptr<float>(sy);
        for (int sx = x0; sx < x1; ++sx) {
          const float v = row[sx];
          if (v > 0.f) {
            block_vals.push_back(v);
            d_min = std::min(d_min, v);
            d_max = std::max(d_max, v);
          }
        }
      }
      if (block_vals.empty())
        continue;
      if (d_max - d_min > std::max(0.30f, 0.15f * d_min))
        bm[gx] = 1;
      const float thresh = d_min * 1.06f + 0.06f;
      double sum = 0.0;
      int cnt = 0;
      for (float v : block_vals) {
        if (v <= thresh) {
          sum += v;
          ++cnt;
        }
      }
      if (cnt > 0)
        dst[gx] = float(sum / double(cnt));
    }
  }

  // Conservative 1-pixel hole fill where >= 5 neighbors agree smoothly.
  depth_out = pooled.clone();
  float nbrs[8];
  for (int y = 1; y < gh - 1; ++y) {
    const float* r_prev = pooled.ptr<float>(y - 1);
    const float* r_cur = pooled.ptr<float>(y);
    const float* r_next = pooled.ptr<float>(y + 1);
    float* dst = depth_out.ptr<float>(y);
    for (int x = 1; x < gw - 1; ++x) {
      if (r_cur[x] > 0.f)
        continue;
      int cnt = 0;
      float d_min = 1e9f, d_max = 0.f;
      const float vals[8] = {r_prev[x - 1], r_prev[x], r_prev[x + 1], r_cur[x - 1],
                             r_cur[x + 1],  r_next[x - 1], r_next[x], r_next[x + 1]};
      for (float v : vals) {
        if (v > 0.f) {
          nbrs[cnt++] = v;
          d_min = std::min(d_min, v);
          d_max = std::max(d_max, v);
        }
      }
      if (cnt >= 5 && d_max - d_min <= std::max(0.10f, 0.06f * d_min)) {
        std::nth_element(nbrs, nbrs + cnt / 2, nbrs + cnt);
        dst[x] = nbrs[cnt / 2];
      }
    }
  }

  cv::Mat gx_mat, gy_mat;
  cv::Sobel(depth_out, gx_mat, CV_32F, 1, 0, 3);
  cv::Sobel(depth_out, gy_mat, CV_32F, 0, 1, 3);
  smooth_out = cv::Mat::zeros(gh, gw, CV_8UC1);
  for (int y = 1; y < gh - 1; ++y) {
    const float* r_prev = depth_out.ptr<float>(y - 1);
    const float* r_cur = depth_out.ptr<float>(y);
    const float* r_next = depth_out.ptr<float>(y + 1);
    const uint8_t* bm = bimodal.ptr<uint8_t>(y);
    const float* gxp = gx_mat.ptr<float>(y);
    const float* gyp = gy_mat.ptr<float>(y);
    uint8_t* sp = smooth_out.ptr<uint8_t>(y);
    for (int x = 1; x < gw - 1; ++x) {
      if (bm[x])
        continue;
      const float d = r_cur[x];
      if (d < float(PhotoMinDepthM) || d > float(PhotoMaxDepthM))
        continue;
      bool all_valid = true;
      float d_min = d, d_max = d;
      const float* rows[3] = {r_prev, r_cur, r_next};
      for (int ry = 0; ry < 3 && all_valid; ++ry) {
        for (int dx = -1; dx <= 1; ++dx) {
          const float v = rows[ry][x + dx];
          if (v <= 0.f) {
            all_valid = false;
            break;
          }
          d_min = std::min(d_min, v);
          d_max = std::max(d_max, v);
        }
      }
      if (!all_valid)
        continue;
      if (d_max - d_min > std::max(0.15f, 0.10f * d))
        continue;
      if (std::hypot(gxp[x], gyp[x]) / std::max(d, 0.2f) < float(PhotoMaxDepthGrad))
        sp[x] = 255;
    }
  }
}
void BuildPhotoView(Context& context, uint32_t id, const cv::Mat& full_mask,
                    const cv::Mat& depth_1024) {
  const uint32_t cid = context.model.images[id].camera;
  const auto& grid = context.photo_grids[cid];
  PhotoView view;
  CleanAndDownsampleLasDepth(depth_1024, grid.width, grid.height, view.depth, view.depth_smooth);

  cv::Mat bgr = ReadImage(context.scene.images[id].source, cv::IMREAD_REDUCED_COLOR_8);
  if (bgr.empty() || bgr.channels() != 3 || bgr.cols < grid.width || bgr.rows < grid.height)
    bgr = ReadImage(context.scene.images[id].source, cv::IMREAD_COLOR);
  if (bgr.empty() || bgr.channels() != 3)
    throw std::runtime_error("Invalid photo channels for photometric mask: " +
                             context.scene.images[id].name);

  cv::Mat small_bgr, rgb_f;
  cv::resize(bgr, small_bgr, {grid.width, grid.height}, 0, 0, cv::INTER_AREA);
  view.max_srgb.create(grid.height, grid.width, CV_32FC1);
  for (int y = 0; y < grid.height; ++y) {
    const cv::Vec3b* sp = small_bgr.ptr<cv::Vec3b>(y);
    float* mp = view.max_srgb.ptr<float>(y);
    for (int x = 0; x < grid.width; ++x)
      mp[x] = float(std::max({sp[x][0], sp[x][1], sp[x][2]})) / 255.0f;
  }
  small_bgr.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);
  cv::GaussianBlur(rgb_f, rgb_f, {3, 3}, 0.7);

  view.lum_srgb.create(grid.height, grid.width, CV_32FC1);
  view.lum_lin.create(grid.height, grid.width, CV_32FC1);
  view.rgb_lin.create(grid.height, grid.width, CV_32FC3);
  for (int y = 0; y < grid.height; ++y) {
    const cv::Vec3f* sp = rgb_f.ptr<cv::Vec3f>(y);
    float* ps = view.lum_srgb.ptr<float>(y);
    float* pl = view.lum_lin.ptr<float>(y);
    cv::Vec3f* rl = view.rgb_lin.ptr<cv::Vec3f>(y);
    for (int x = 0; x < grid.width; ++x) {
      const float b = std::clamp(sp[x][0], 0.f, 1.f);
      const float g = std::clamp(sp[x][1], 0.f, 1.f);
      const float r = std::clamp(sp[x][2], 0.f, 1.f);
      const float bl = SrgbToLin(b), gl = SrgbToLin(g), rl_v = SrgbToLin(r);
      ps[x] = 0.114f * b + 0.587f * g + 0.299f * r;
      pl[x] = 0.0722f * bl + 0.7152f * gl + 0.2126f * rl_v;
      rl[x] = {bl, gl, rl_v};
    }
  }
  cv::Mat gx, gy;
  cv::Sobel(view.lum_srgb, gx, CV_32F, 1, 0, 3);
  cv::Sobel(view.lum_srgb, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, view.grad_srgb);

  const cv::Mat& full_domain = context.full_domains[cid];
  cv::Mat static_resized, sky_resized, person_resized;
  cv::resize((full_mask == 255) & full_domain, static_resized, {grid.width, grid.height}, 0, 0,
             cv::INTER_AREA);
  view.static_mask = (static_resized >= 252) & grid.domain;
  cv::resize((full_mask == 128) & full_domain, sky_resized, {grid.width, grid.height}, 0, 0,
             cv::INTER_AREA);
  view.raw_sky_small = (sky_resized >= 128) & grid.domain;
  cv::resize((full_mask == 0) & context.domains[cid], person_resized, {grid.width, grid.height}, 0,
             0, cv::INTER_AREA);
  view.person_small = (person_resized > 0) & grid.domain;

  context.photo_views[id] = std::move(view);
}
// Transient components of an image's mask, on the given grid.
Labelled Label(const Context& context, uint32_t id, cv::Size grid) {
  const auto& name = context.scene.images[id].name;
  const auto& camera = context.model.cameras[context.model.images[id].camera];
  Labelled out;
  out.mask = ReadImage(context.source(id), cv::IMREAD_GRAYSCALE);
  if (out.mask.cols != camera.width || out.mask.rows != camera.height)
    throw std::runtime_error("Mask size differs from its camera: " + name);
  const cv::Mat& domain = context.domains[context.model.images[id].camera];
  cv::resize((out.mask == 0) & domain, out.transient_map, grid, 0, 0, cv::INTER_NEAREST);
  out.count = cv::connectedComponentsWithStats(out.transient_map, out.labels, out.stats,
                                               out.centroids, 8);
  return out;
}
// Pass 1, per image: where each transient candidate is in the world (median LAS point of it).
void Collect(Context& context, uint32_t id, const cv::Mat& depth) {
  const auto& model = context.model;
  const auto& camera = model.cameras[model.images[id].camera];
  auto labelled = Label(context, id, depth.size());
  BuildPhotoView(context, id, labelled.mask, depth);
  const double scale_x = double(camera.width) / depth.cols, scale_y = double(camera.height) / depth.rows;
  const cv::Mat& domain = context.domains[model.images[id].camera];
  cv::Mat transient;
  cv::resize((labelled.mask == 0) & domain, transient,
             {camera.width / ThumbScale, camera.height / ThumbScale}, 0, 0, cv::INTER_AREA);
  context.thumbs[id] = transient >= 80;
  std::vector<Candidate> found;
  for (int k = 1; k < labelled.count; ++k) {
    Candidate c;
    c.image = id;
    c.label = k;
    c.area = labelled.stats.at<int>(k, cv::CC_STAT_AREA);
    if (c.area < MinArea)
      continue;
    c.x = labelled.centroids.at<double>(k, 0);
    c.y = labelled.centroids.at<double>(k, 1);
    const cv::Rect box(labelled.stats.at<int>(k, cv::CC_STAT_LEFT),
                       labelled.stats.at<int>(k, cv::CC_STAT_TOP),
                       labelled.stats.at<int>(k, cv::CC_STAT_WIDTH),
                       labelled.stats.at<int>(k, cv::CC_STAT_HEIGHT));
    std::vector<cv::Point> pixels;
    cv::findNonZero(labelled.labels(box) == k, pixels);
    std::array<std::vector<double>, 3> coordinates;
    for (const auto& p : pixels) {
      const double d = depth.at<float>(p.y + box.y, p.x + box.x);
      if (d <= 0)
        continue;
      const Eigen::Vector2d pixel{(p.x + box.x + .5) * scale_x, (p.y + box.y + .5) * scale_y};
      const auto ray = camera.Ray(pixel);
      const Eigen::Vector3d moved = ray * d;
      const Eigen::Vector3d cam_point =
          context.motion ? context.motion->Reference(
                               id, model.images[id].camera,
                               context.motion->Fraction(camera, pixel), moved)
                         : moved;
      const Eigen::Vector3d world =
          context.rotations[id].transpose() * (cam_point - context.translations[id]);
      for (int a = 0; a < 3; ++a)
        coordinates[a].push_back(world[a]);
    }
    if (coordinates[0].size() < size_t(MinSamples) || coordinates[0].size() < pixels.size() * 3 / 10) {
      c.reason = "no_las";
    } else {
      for (int a = 0; a < 3; ++a) {
        auto& values = coordinates[a];
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        c.place[a] = values[values.size() / 2];
      }
      c.located = true;
    }
    found.push_back(std::move(c));
  }
  context.candidates[id] = std::move(found);
  const cv::Mat raw_sky = (labelled.mask == 128);
  if (cv::countNonZero(raw_sky) > 0) {
    SkyImage sky_info;
    sky_info.has_raw_sky = true;
    sky_info.valid_grid_pixels = context.full_domain_grid_pixels[model.images[id].camera];
    cv::Mat sky_down;
    cv::resize(raw_sky, sky_down, depth.size(), 0, 0, cv::INTER_AREA);
    cv::Mat sky_labels, sky_stats, sky_centroids;
    const int sky_count =
        cv::connectedComponentsWithStats(sky_down > 0, sky_labels, sky_stats, sky_centroids, 8);
    const cv::Mat sky_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * SkyCoreErosion + 1, 2 * SkyCoreErosion + 1});
    cv::Mat valid_las = (depth > 0.f);
    cv::Mat las_core;
    cv::erode(valid_las, las_core, sky_kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
              cv::Scalar(0));
    context.las_interior[id] = std::move(las_core);
    for (int k = 1; k < sky_count; ++k) {
      SkyComponent comp;
      comp.label = k;
      comp.area = sky_stats.at<int>(k, cv::CC_STAT_AREA);
      const cv::Rect box(sky_stats.at<int>(k, cv::CC_STAT_LEFT),
                         sky_stats.at<int>(k, cv::CC_STAT_TOP),
                         sky_stats.at<int>(k, cv::CC_STAT_WIDTH),
                         sky_stats.at<int>(k, cv::CC_STAT_HEIGHT));
      if (comp.area >= MinArea && box.width > 2 * SkyCoreErosion &&
          box.height > 2 * SkyCoreErosion) {
        const cv::Mat patch = (sky_labels(box) == k);
        cv::Mat core;
        cv::erode(patch, core, sky_kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
                  cv::Scalar(0));
        comp.core_area = cv::countNonZero(core);
        if (comp.core_area >= MinArea) {
          std::vector<cv::Point> core_pixels;
          cv::findNonZero(core, core_pixels);
          for (const auto& p : core_pixels)
            if (depth.at<float>(p.y + box.y, p.x + box.x) > 0.f)
              ++comp.core_las;
          if (comp.core_las >= MinSamples &&
              double(comp.core_las) >= SkyLasShare * double(comp.core_area))
            comp.surface = true;
        }
      }
      if (!comp.surface)
        sky_info.open_sky_grid_pixels += comp.area;
      sky_info.components.push_back(comp);
    }
    context.skies[id] = std::move(sky_info);
  }
}
// Pass 2: how many of the views that see a candidate's place flagged it, and over what time.
void Judge(Context& context, Candidate& c) {
  const auto& model = context.model;
  double first = context.seconds[c.image], last = first;
  for (uint32_t b = 0; b < model.images.size(); ++b) {
    if (b == c.image || !model.images[b].registered)
      continue;
    const Eigen::Vector3d seen = context.rotations[b] * c.place + context.translations[b];
    const auto& other = model.cameras[model.images[b].camera];
    if (seen.z() <= .3 || seen.norm() < .8 || seen.norm() > MaxRange)
      continue;
    const auto pixel = other.Project(seen);
    const double dx = (pixel.x() - other.params[2]) / other.params[0],
                 dy = (pixel.y() - other.params[3]) / other.params[1];
    if (dx * dx + dy * dy > std::pow(other.Radius(), 2) * ValidShare * ValidShare || pixel.x() < 0 ||
        pixel.y() < 0 || pixel.x() >= other.width || pixel.y() >= other.height)
      continue;
    ++c.seen;
    const cv::Mat& thumb = context.thumbs[b];
    const int tx = int(pixel.x() / ThumbScale), ty = int(pixel.y() / ThumbScale);
    bool flagged = false;
    for (int y = std::max(0, ty - 1); y <= std::min(thumb.rows - 1, ty + 1) && !flagged; ++y)
      for (int x = std::max(0, tx - 1); x <= std::min(thumb.cols - 1, tx + 1); ++x)
        if (thumb.at<uint8_t>(y, x)) {
          flagged = true;
          break;
        }
    if (flagged) {
      ++c.flagged;
      first = std::min(first, context.seconds[b]);
      last = std::max(last, context.seconds[b]);
    }
  }
  c.span = last - first;
  if (c.seen < size_t(MinSeen))
    c.reason = "few_views";
  else if (double(c.flagged) / double(c.seen) >= MinRatio && c.span >= MinSpan)
    c.reason = "static";
  else
    c.reason = "dynamic";
}
struct ApplyCounts {
  uint64_t person_unmasked = 0;
  uint64_t sky_cleared = 0;
  uint64_t sky_edge_to_sky = 0;
  uint64_t sky_edge_to_static = 0;
  uint64_t photo_dark = 0;
  uint64_t photo_bright = 0;
  uint64_t photo_clipped = 0;
};
// Smooth local per-channel mean and variance over seed_mask != 0 using 4x block accumulation
// (exact squared-sum accumulation before smoothing avoids downsampling variance bias), blending
// to global seed statistics where local seed support falls below 64 pixels.
void LocalColorStats(const std::array<cv::Mat, 3>& ch, const cv::Mat& seed_mask, int box_size,
                     float min_var, std::array<cv::Mat, 3>& mean_out,
                     std::array<cv::Mat, 3>& var_out) {
  const int rows = seed_mask.rows, cols = seed_mask.cols;
  constexpr int ds = 4;
  const int rows_ds = (rows + ds - 1) / ds, cols_ds = (cols + ds - 1) / ds;
  cv::Mat ds_w = cv::Mat::zeros(rows_ds, cols_ds, CV_64FC1);
  std::array<cv::Mat, 3> ds_c, ds_c2;
  for (int c = 0; c < 3; ++c) {
    ds_c[c] = cv::Mat::zeros(rows_ds, cols_ds, CV_64FC1);
    ds_c2[c] = cv::Mat::zeros(rows_ds, cols_ds, CV_64FC1);
  }
  std::array<double, 3> g_sum{}, g_sum2{};
  size_t g_count = 0;
  for (int y = 0; y < rows; ++y) {
    const int by = y / ds;
    const uint8_t* m = seed_mask.ptr<uint8_t>(y);
    const float* p0 = ch[0].ptr<float>(y);
    const float* p1 = ch[1].ptr<float>(y);
    const float* p2 = ch[2].ptr<float>(y);
    double* wp = ds_w.ptr<double>(by);
    double *c0p = ds_c[0].ptr<double>(by), *c1p = ds_c[1].ptr<double>(by),
           *c2p = ds_c[2].ptr<double>(by);
    double *s0p = ds_c2[0].ptr<double>(by), *s1p = ds_c2[1].ptr<double>(by),
           *s2p = ds_c2[2].ptr<double>(by);
    for (int x = 0; x < cols; ++x)
      if (m[x]) {
        const double v0 = p0[x], v1 = p1[x], v2 = p2[x];
        g_sum[0] += v0;
        g_sum[1] += v1;
        g_sum[2] += v2;
        g_sum2[0] += v0 * v0;
        g_sum2[1] += v1 * v1;
        g_sum2[2] += v2 * v2;
        ++g_count;
        const int bx = x / ds;
        wp[bx] += 1.0;
        c0p[bx] += v0;
        c1p[bx] += v1;
        c2p[bx] += v2;
        s0p[bx] += v0 * v0;
        s1p[bx] += v1 * v1;
        s2p[bx] += v2 * v2;
      }
  }
  std::array<float, 3> g_mean{.5f, .5f, .5f}, g_var{.25f, .25f, .25f};
  if (g_count > 0) {
    for (int c = 0; c < 3; ++c) {
      const double m = g_sum[c] / double(g_count);
      g_mean[c] = float(m);
      g_var[c] = std::max(min_var, float(g_sum2[c] / double(g_count) - m * m));
    }
  }
  const int box_ds = std::max(3, ((box_size + ds / 2) / ds) | 1);
  const cv::Size ksize_ds(box_ds, box_ds);
  cv::Mat sw;
  cv::boxFilter(ds_w, sw, CV_64F, ksize_ds, {-1, -1}, false, cv::BORDER_REFLECT_101);
  for (int c = 0; c < 3; ++c) {
    cv::Mat sm, sm2;
    cv::boxFilter(ds_c[c], sm, CV_64F, ksize_ds, {-1, -1}, false, cv::BORDER_REFLECT_101);
    cv::boxFilter(ds_c2[c], sm2, CV_64F, ksize_ds, {-1, -1}, false, cv::BORDER_REFLECT_101);
    cv::Mat mean_ds(rows_ds, cols_ds, CV_32FC1), var_ds(rows_ds, cols_ds, CV_32FC1);
    for (int by = 0; by < rows_ds; ++by) {
      const double* swp = sw.ptr<double>(by);
      const double* smp = sm.ptr<double>(by);
      const double* sm2p = sm2.ptr<double>(by);
      float* mo = mean_ds.ptr<float>(by);
      float* vo = var_ds.ptr<float>(by);
      for (int bx = 0; bx < cols_ds; ++bx) {
        const double cnt = swp[bx];
        if (cnt >= 64.0) {
          const double m = smp[bx] / cnt;
          mo[bx] = float(m);
          vo[bx] = std::max(min_var, float(sm2p[bx] / cnt - m * m));
        } else if (cnt > 1.0) {
          const double m_loc = smp[bx] / cnt;
          const float v_loc = std::max(min_var, float(sm2p[bx] / cnt - m_loc * m_loc));
          const float blend = float(cnt / 64.0);
          mo[bx] = blend * float(m_loc) + (1.f - blend) * g_mean[c];
          vo[bx] = std::max(min_var, blend * v_loc + (1.f - blend) * g_var[c]);
        } else {
          mo[bx] = g_mean[c];
          vo[bx] = g_var[c];
        }
      }
    }
    cv::resize(mean_ds, mean_out[c], {cols, rows}, 0, 0, cv::INTER_LINEAR);
    cv::resize(var_ds, var_out[c], {cols, rows}, 0, 0, cv::INTER_LINEAR);
  }
}
void GuidedFilterCoefficients(const std::array<cv::Mat, 3>& ch, const cv::Mat& p, int radius,
                              float eps, std::array<cv::Mat, 3>& ma, cv::Mat& mb) {
  const cv::Size ksize(2 * radius + 1, 2 * radius + 1);
  auto box = [&](const cv::Mat& m) {
    cv::Mat out;
    cv::boxFilter(m, out, CV_32F, ksize, {-1, -1}, true, cv::BORDER_REFLECT_101);
    return out;
  };
  std::array<cv::Mat, 3> mean_i, cov_ip;
  const cv::Mat mean_p = box(p);
  for (int c = 0; c < 3; ++c) {
    mean_i[c] = box(ch[c]);
    cov_ip[c] = box(ch[c].mul(p)) - mean_i[c].mul(mean_p);
  }
  const cv::Mat v00 = box(ch[0].mul(ch[0])) - mean_i[0].mul(mean_i[0]) + eps;
  const cv::Mat v01 = box(ch[0].mul(ch[1])) - mean_i[0].mul(mean_i[1]);
  const cv::Mat v02 = box(ch[0].mul(ch[2])) - mean_i[0].mul(mean_i[2]);
  const cv::Mat v11 = box(ch[1].mul(ch[1])) - mean_i[1].mul(mean_i[1]) + eps;
  const cv::Mat v12 = box(ch[1].mul(ch[2])) - mean_i[1].mul(mean_i[2]);
  const cv::Mat v22 = box(ch[2].mul(ch[2])) - mean_i[2].mul(mean_i[2]) + eps;
  const int rows = p.rows, cols = p.cols;
  std::array<cv::Mat, 3> a;
  for (int c = 0; c < 3; ++c)
    a[c].create(rows, cols, CV_32FC1);
  cv::Mat b(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    const float *r00 = v00.ptr<float>(y), *r01 = v01.ptr<float>(y), *r02 = v02.ptr<float>(y),
                *r11 = v11.ptr<float>(y), *r12 = v12.ptr<float>(y), *r22 = v22.ptr<float>(y);
    const float *c0 = cov_ip[0].ptr<float>(y), *c1 = cov_ip[1].ptr<float>(y),
                *c2 = cov_ip[2].ptr<float>(y), *mp = mean_p.ptr<float>(y);
    const float *m0 = mean_i[0].ptr<float>(y), *m1 = mean_i[1].ptr<float>(y),
                *m2 = mean_i[2].ptr<float>(y);
    float *a0p = a[0].ptr<float>(y), *a1p = a[1].ptr<float>(y), *a2p = a[2].ptr<float>(y),
          *bp = b.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      const double d00 = r00[x], d01 = r01[x], d02 = r02[x], d11 = r11[x], d12 = r12[x],
                   d22 = r22[x];
      const double adj00 = d11 * d22 - d12 * d12;
      const double adj01 = d02 * d12 - d01 * d22;
      const double adj02 = d01 * d12 - d02 * d11;
      const double adj11 = d00 * d22 - d02 * d02;
      const double adj12 = d02 * d01 - d00 * d12;
      const double adj22 = d00 * d11 - d01 * d01;
      const double det = d00 * adj00 + d01 * adj01 + d02 * adj02;
      const double inv = det > 1e-12 ? 1.0 / det : 0.0;
      const double k0 = c0[x], k1 = c1[x], k2 = c2[x];
      const double av0 = (adj00 * k0 + adj01 * k1 + adj02 * k2) * inv;
      const double av1 = (adj01 * k0 + adj11 * k1 + adj12 * k2) * inv;
      const double av2 = (adj02 * k0 + adj12 * k1 + adj22 * k2) * inv;
      a0p[x] = float(av0);
      a1p[x] = float(av1);
      a2p[x] = float(av2);
      bp[x] = float(double(mp[x]) - av0 * m0[x] - av1 * m1[x] - av2 * m2[x]);
    }
  }
  ma = {box(a[0]), box(a[1]), box(a[2])};
  mb = box(b);
}
// 3-channel Fast Guided Filter: solves linear model coefficients at 2x downsample when radius >= 6,
// then evaluates at full resolution strictly inside the boundary band.
cv::Mat GuidedFilterColor(const std::array<cv::Mat, 3>& ch, const cv::Mat& p, const cv::Mat& band,
                          int radius, float eps) {
  const int rows = p.rows, cols = p.cols;
  std::array<cv::Mat, 3> ma;
  cv::Mat mb;
  if (radius >= 6 && cols >= 64 && rows >= 64) {
    const cv::Size sub_size(cols / 2, rows / 2);
    const int r_sub = std::max(2, (radius + 1) / 2);
    std::array<cv::Mat, 3> ch_sub;
    for (int c = 0; c < 3; ++c)
      cv::resize(ch[c], ch_sub[c], sub_size, 0, 0, cv::INTER_AREA);
    cv::Mat p_sub;
    cv::resize(p, p_sub, sub_size, 0, 0, cv::INTER_AREA);
    std::array<cv::Mat, 3> ma_sub;
    cv::Mat mb_sub;
    GuidedFilterCoefficients(ch_sub, p_sub, r_sub, eps, ma_sub, mb_sub);
    for (int c = 0; c < 3; ++c)
      cv::resize(ma_sub[c], ma[c], {cols, rows}, 0, 0, cv::INTER_LINEAR);
    cv::resize(mb_sub, mb, {cols, rows}, 0, 0, cv::INTER_LINEAR);
  } else {
    GuidedFilterCoefficients(ch, p, radius, eps, ma, mb);
  }
  cv::Mat q = cv::Mat::zeros(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    const uint8_t* bp = band.ptr<uint8_t>(y);
    const float *p0 = ma[0].ptr<float>(y), *p1 = ma[1].ptr<float>(y), *p2 = ma[2].ptr<float>(y),
                *pb = mb.ptr<float>(y);
    const float *i0 = ch[0].ptr<float>(y), *i1 = ch[1].ptr<float>(y), *i2 = ch[2].ptr<float>(y);
    float* qo = q.ptr<float>(y);
    for (int x = 0; x < cols; ++x)
      if (bp[x])
        qo[x] = std::clamp(p0[x] * i0[x] + p1[x] * i1[x] + p2[x] * i2[x] + pb[x], 0.f, 1.f);
  }
  return q;
}
// Refines coarse SAM3 128/255 sky edges using the full-resolution photo, locking interior LAS
// returns as foreground and clearing small sunlit sky pores trapped inside dark tree canopies.
void RefineOutdoorSkyBoundary(const Context& context, uint32_t id, cv::Mat& mask,
                              ApplyCounts& counts) {
  const uint32_t cam_idx = context.model.images[id].camera;
  const auto& camera = context.model.cameras[cam_idx];
  const cv::Mat sky = (mask == 128);
  const cv::Rect sky_box = cv::boundingRect(sky);
  if (sky_box.empty())
    return;
  const double scale = std::clamp(std::max(camera.width, camera.height) / 4000.0, .5, 2.0);
  const int r_in = std::max(6, int(std::lround(18.0 * scale)));
  const int r_out = std::max(10, int(std::lround(28.0 * scale)));
  const int r_fg = std::max(16, int(std::lround(44.0 * scale)));
  const int r_gf = std::max(4, int(std::lround(SkyEdgeGuidedRadius4k * scale)));
  const int r_canopy = std::max(24, int(std::lround(120.0 * scale)));
  const int r_stat = std::max(32, int(std::lround(128.0 * scale)));
  const int max_pore_area = std::max(16, int(std::lround(SkyCanopyPoreMaxArea4k * scale * scale)));
  const int pad = r_canopy + r_gf + 4;
  const int x0 = std::max(0, sky_box.x - pad), y0 = std::max(0, sky_box.y - pad);
  const int x1 = std::min(camera.width, sky_box.x + sky_box.width + pad);
  const int y1 = std::min(camera.height, sky_box.y + sky_box.height + pad);
  const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
  const cv::Mat sky_roi = sky(roi);
  cv::Mat sky_padded, dist_in;
  cv::copyMakeBorder(sky_roi, sky_padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::distanceTransform(sky_padded, dist_in, cv::DIST_L2, cv::DIST_MASK_5);
  const cv::Mat sky_core = dist_in(cv::Rect(1, 1, roi.width, roi.height)) > float(r_in);
  if (cv::countNonZero(sky_core) < 64)
    return;
  const cv::Mat bgr = ReadImage(context.scene.images[id].source, cv::IMREAD_COLOR);
  if (bgr.cols != camera.width || bgr.rows != camera.height || bgr.channels() != 3)
    throw std::runtime_error("Photo size differs from its camera: " + context.scene.images[id].name);
  cv::Mat las_lock = cv::Mat::zeros(roi.size(), CV_8UC1);
  if (!context.las_interior[id].empty()) {
    cv::Mat las_full;
    cv::resize(context.las_interior[id], las_full, {camera.width, camera.height}, 0, 0,
               cv::INTER_NEAREST);
    las_lock = las_full(roi).clone();
  }
  cv::Mat mask_roi = mask(roi);
  const cv::Mat dom_roi = context.full_domains[cam_idx](roi);
  const cv::Mat editable = (dom_roi != 0) & (mask_roi != 0) & (las_lock == 0);
  cv::Mat dist_out;
  cv::distanceTransform(sky_roi == 0, dist_out, cv::DIST_L2, cv::DIST_MASK_5);
  const cv::Mat sky_outer = dist_out <= float(r_out);
  const cv::Mat sky_fg_shell = dist_out <= float(r_fg);
  const cv::Mat sky_canopy = dist_out <= float(r_canopy);
  cv::Mat fg_seed =
      ((sky_fg_shell & ~sky_outer) & (mask_roi == 255)) | (sky_outer & (mask_roi == 255) & (las_lock != 0));
  if (cv::countNonZero(fg_seed) < 64)
    fg_seed = (mask_roi == 255) & (dom_roi != 0);
  if (cv::countNonZero(fg_seed) < 64)
    return;
  cv::Mat bgr_f;
  bgr(roi).convertTo(bgr_f, CV_32FC3, 1.0 / 255.0);
  std::array<cv::Mat, 3> ch, mu_sky, var_sky, mu_fg, var_fg;
  cv::split(bgr_f, ch.data());
  LocalColorStats(ch, sky_core, 2 * r_stat + 1, .03f * .03f, mu_sky, var_sky);
  LocalColorStats(ch, fg_seed, 2 * r_stat + 1, .05f * .05f, mu_fg, var_fg);
  cv::Mat sky_f, p_prior;
  sky_roi.convertTo(sky_f, CV_32FC1, 1.0 / 255.0);
  cv::boxFilter(sky_f, p_prior, CV_32F, {2 * r_gf + 1, 2 * r_gf + 1}, {-1, -1}, true,
                cv::BORDER_REFLECT_101);
  const cv::Mat band = (sky_outer & ~sky_core) & editable;
  cv::Mat p0(roi.size(), CV_32FC1);
  for (int y = 0; y < roi.height; ++y) {
    const uint8_t* bp = band.ptr<uint8_t>(y);
    const uint8_t* lp = las_lock.ptr<uint8_t>(y);
    const uint8_t* mp = mask_roi.ptr<uint8_t>(y);
    const uint8_t* dp = dom_roi.ptr<uint8_t>(y);
    const uint8_t* sp = sky_roi.ptr<uint8_t>(y);
    const float* pp = p_prior.ptr<float>(y);
    const float *c0 = ch[0].ptr<float>(y), *c1 = ch[1].ptr<float>(y), *c2 = ch[2].ptr<float>(y);
    const float *ms0 = mu_sky[0].ptr<float>(y), *ms1 = mu_sky[1].ptr<float>(y),
                *ms2 = mu_sky[2].ptr<float>(y);
    const float *vs0 = var_sky[0].ptr<float>(y), *vs1 = var_sky[1].ptr<float>(y),
                *vs2 = var_sky[2].ptr<float>(y);
    const float *mf0 = mu_fg[0].ptr<float>(y), *mf1 = mu_fg[1].ptr<float>(y),
                *mf2 = mu_fg[2].ptr<float>(y);
    const float *vf0 = var_fg[0].ptr<float>(y), *vf1 = var_fg[1].ptr<float>(y),
                *vf2 = var_fg[2].ptr<float>(y);
    float* out = p0.ptr<float>(y);
    for (int x = 0; x < roi.width; ++x) {
      if (bp[x]) {
        const float d0s = c0[x] - ms0[x], d1s = c1[x] - ms1[x], d2s = c2[x] - ms2[x];
        const float d0f = c0[x] - mf0[x], d1f = c1[x] - mf1[x], d2f = c2[x] - mf2[x];
        const float d_sky2 =
            std::clamp(d0s * d0s / vs0[x] + d1s * d1s / vs1[x] + d2s * d2s / vs2[x], 0.f, 36.f);
        const float d_fg2 =
            std::clamp(d0f * d0f / vf0[x] + d1f * d1f / vf1[x] + d2f * d2f / vf2[x], 0.f, 36.f);
        const float prior = std::clamp(pp[x], .15f, .85f);
        const float logit =
            std::clamp(std::log(prior / (1.f - prior)) + .5f * (d_fg2 - d_sky2), -20.f, 20.f);
        out[x] = 1.f / (1.f + std::exp(-logit));
      } else if (lp[x] || !mp[x] || !dp[x]) {
        out[x] = 0.f;
      } else {
        out[x] = sp[x] ? 1.f : 0.f;
      }
    }
  }
  const cv::Mat alpha_sky = GuidedFilterColor(ch, p0, band, r_gf, 1e-3f);
  for (int y = 0; y < roi.height; ++y) {
    const uint8_t* bp = band.ptr<uint8_t>(y);
    const float* ap = alpha_sky.ptr<float>(y);
    uint8_t* mp = mask_roi.ptr<uint8_t>(y);
    for (int x = 0; x < roi.width; ++x) {
      if (!bp[x])
        continue;
      const uint8_t next = ap[x] >= float(SkyEdgeAlphaThreshold) ? 128 : 255;
      if (mp[x] == 255 && next == 128)
        ++counts.sky_edge_to_sky;
      else if (mp[x] == 128 && next == 255)
        ++counts.sky_edge_to_static;
      mp[x] = next;
    }
  }
  // Tree canopy sky pores near the open sky, guarded by a dark-foliage border so glass facades are
  // never touched.
  const cv::Mat canopy_zone = (sky_canopy & ~sky_outer) & editable & (mask_roi == 255);
  cv::Mat pore_cand = cv::Mat::zeros(roi.size(), CV_8UC1);
  for (int y = 0; y < roi.height; ++y) {
    const uint8_t* cz = canopy_zone.ptr<uint8_t>(y);
    const float *c0 = ch[0].ptr<float>(y), *c1 = ch[1].ptr<float>(y), *c2 = ch[2].ptr<float>(y);
    const float *ms0 = mu_sky[0].ptr<float>(y), *ms1 = mu_sky[1].ptr<float>(y),
                *ms2 = mu_sky[2].ptr<float>(y);
    const float *vs0 = var_sky[0].ptr<float>(y), *vs1 = var_sky[1].ptr<float>(y),
                *vs2 = var_sky[2].ptr<float>(y);
    uint8_t* po = pore_cand.ptr<uint8_t>(y);
    for (int x = 0; x < roi.width; ++x) {
      if (!cz[x])
        continue;
      const float b = c0[x], g = c1[x], r = c2[x];
      const float lum = .114f * b + .587f * g + .299f * r;
      const float chroma = std::max({b, g, r}) - std::min({b, g, r});
      if (chroma > .12f)
        continue;
      const float mu_l = .114f * ms0[x] + .587f * ms1[x] + .299f * ms2[x];
      const float var_l =
          .114f * .114f * vs0[x] + .587f * .587f * vs1[x] + .299f * .299f * vs2[x];
      if (lum < std::max(.70f, mu_l - 1.5f * std::sqrt(var_l)))
        continue;
      const float d0 = b - ms0[x], d1 = g - ms1[x], d2 = r - ms2[x];
      const float d_sky2 = d0 * d0 / vs0[x] + d1 * d1 / vs1[x] + d2 * d2 / vs2[x];
      if (d_sky2 <= 4.f)
        po[x] = 255;
    }
  }
  if (cv::countNonZero(pore_cand) == 0)
    return;
  cv::Mat comp_labels, comp_stats, comp_centroids;
  const int comp_count =
      cv::connectedComponentsWithStats(pore_cand, comp_labels, comp_stats, comp_centroids, 8);
  cv::Mat pore_to_sky = cv::Mat::zeros(roi.size(), CV_8UC1);
  const cv::Mat k9 = cv::Mat::ones(9, 9, CV_8UC1), k3 = cv::Mat::ones(3, 3, CV_8UC1);
  for (int k = 1; k < comp_count; ++k) {
    const int area = comp_stats.at<int>(k, cv::CC_STAT_AREA);
    if (area < 4 || area > max_pore_area)
      continue;
    const int bx = std::max(0, comp_stats.at<int>(k, cv::CC_STAT_LEFT) - 5);
    const int by = std::max(0, comp_stats.at<int>(k, cv::CC_STAT_TOP) - 5);
    const int bx1 = std::min(
        roi.width, comp_stats.at<int>(k, cv::CC_STAT_LEFT) + comp_stats.at<int>(k, cv::CC_STAT_WIDTH) + 5);
    const int by1 = std::min(
        roi.height, comp_stats.at<int>(k, cv::CC_STAT_TOP) + comp_stats.at<int>(k, cv::CC_STAT_HEIGHT) + 5);
    const cv::Rect pbox(bx, by, bx1 - bx, by1 - by);
    const cv::Mat comp_patch = (comp_labels(pbox) == k);
    cv::Mat outer9;
    cv::dilate(comp_patch, outer9, k9);
    const cv::Mat border = outer9 & ~comp_patch & (mask_roi(pbox) == 255);
    int border_total = 0, border_foliage = 0;
    for (int py = 0; py < pbox.height; ++py) {
      const uint8_t* br = border.ptr<uint8_t>(py);
      const float* b_ptr = ch[0].ptr<float>(pbox.y + py) + pbox.x;
      const float* g_ptr = ch[1].ptr<float>(pbox.y + py) + pbox.x;
      const float* r_ptr = ch[2].ptr<float>(pbox.y + py) + pbox.x;
      for (int px = 0; px < pbox.width; ++px)
        if (br[px]) {
          ++border_total;
          const float b = b_ptr[px], g = g_ptr[px], r = r_ptr[px];
          const float lum = .114f * b + .587f * g + .299f * r;
          if (lum < .45f && (2.f * g - r - b > .03f || lum < .28f))
            ++border_foliage;
        }
    }
    if (border_total < 12 ||
        double(border_foliage) < SkyCanopyFoliageBorderShare * double(border_total))
      continue;
    cv::Mat outer3;
    cv::dilate(comp_patch, outer3, k3);
    const cv::Mat halo_cand = outer3 & ~comp_patch & editable(pbox) & (mask_roi(pbox) == 255);
    cv::Mat halo = cv::Mat::zeros(pbox.size(), CV_8UC1);
    for (int py = 0; py < pbox.height; ++py) {
      const uint8_t* hp = halo_cand.ptr<uint8_t>(py);
      const float* b_ptr = ch[0].ptr<float>(pbox.y + py) + pbox.x;
      const float* g_ptr = ch[1].ptr<float>(pbox.y + py) + pbox.x;
      const float* r_ptr = ch[2].ptr<float>(pbox.y + py) + pbox.x;
      uint8_t* ho = halo.ptr<uint8_t>(py);
      for (int px = 0; px < pbox.width; ++px)
        if (hp[px] && .114f * b_ptr[px] + .587f * g_ptr[px] + .299f * r_ptr[px] >= .65f)
          ho[px] = 255;
    }
    pore_to_sky(pbox) |= comp_patch | halo;
  }
  const cv::Mat new_pore_sky = pore_to_sky & (mask_roi == 255);
  counts.sky_edge_to_sky += uint64_t(cv::countNonZero(new_pore_sky));
  mask_roi.setTo(128, new_pore_sky);
}
bool ProjectWorldToPhoto(const Context& context, uint32_t j, const Eigen::Vector3d& world,
                         double& out_u, double& out_v, double& out_dist, float& out_rho) {
  const uint32_t cid_j = context.model.images[j].camera;
  const auto& full_cam = context.model.cameras[cid_j];
  const auto& grid_j = context.photo_grids[cid_j];
  const Eigen::Vector3d v = context.rotations[j] * world + context.translations[j];
  Eigen::Vector3d p = v;
  Eigen::Vector2d px;
  if (context.motion) {
    double s = 0.0;
    for (int it = 0; it < 2; ++it) {
      p = context.motion->Apply(j, cid_j, s, v);
      if (p.norm() <= 0.25)
        return false;
      px = full_cam.Project(p);
      s = context.motion->Fraction(full_cam, px);
    }
    p = context.motion->Apply(j, cid_j, s, v);
  }
  const double dist = p.norm();
  if (dist <= 0.25)
    return false;
  const double theta = std::acos(std::clamp(p.z() / dist, -1.0, 1.0));
  if (theta >= 79.0 * Pi / 180.0)
    return false;
  px = full_cam.Project(p);
  out_u = px.x() * grid_j.scale_x - 0.5;
  out_v = px.y() * grid_j.scale_y - 0.5;
  if (out_u < 1.5 || out_u >= double(grid_j.width) - 2.5 || out_v < 1.5 ||
      out_v >= double(grid_j.height) - 2.5)
    return false;
  out_dist = dist;
  out_rho = float(theta / (80.0 * Pi / 180.0));
  return true;
}
inline float SampleBilinear(const cv::Mat& img, double u, double v) {
  const int x0 = int(std::floor(u)), y0 = int(std::floor(v));
  const float fx = float(u - x0), fy = float(v - y0);
  const float* r0 = img.ptr<float>(y0);
  const float* r1 = img.ptr<float>(y0 + 1);
  return (1.f - fy) * ((1.f - fx) * r0[x0] + fx * r0[x0 + 1]) +
         fy * ((1.f - fx) * r1[x0] + fx * r1[x0 + 1]);
}
inline cv::Vec3f SampleBilinearVec3(const cv::Mat& img, double u, double v) {
  const int x0 = int(std::floor(u)), y0 = int(std::floor(v));
  const float fx = float(u - x0), fy = float(v - y0);
  const cv::Vec3f* r0 = img.ptr<cv::Vec3f>(y0);
  const cv::Vec3f* r1 = img.ptr<cv::Vec3f>(y0 + 1);
  return r0[x0] * ((1.f - fx) * (1.f - fy)) + r0[x0 + 1] * (fx * (1.f - fy)) +
         r1[x0] * ((1.f - fx) * fy) + r1[x0 + 1] * (fx * fy);
}
inline double MedianValue(std::vector<double>& v) {
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}
void FitVignetting(Context& context, const std::vector<uint32_t>& written) {
  struct VigSample {
    double dr2, dr4, dr6, dl;
  };
  for (uint32_t cid = 0; cid < 2; ++cid) {
    std::vector<uint32_t> cam_imgs;
    for (const uint32_t id : written)
      if (context.model.images[id].camera == cid)
        cam_imgs.push_back(id);
    if (cam_imgs.size() < 4)
      continue;
    const auto& grid = context.photo_grids[cid];
    std::vector<VigSample> samples;
    const size_t stride = std::max<size_t>(1, cam_imgs.size() / 80);
    for (size_t idx = 0; idx < cam_imgs.size(); idx += stride) {
      const uint32_t i = cam_imgs[idx];
      std::vector<std::pair<double, uint32_t>> near;
      for (const uint32_t j : cam_imgs) {
        if (j == i)
          continue;
        const double d = (context.centers[j] - context.centers[i]).norm();
        if (d > 0.5 && d < 6.0)
          near.emplace_back(d, j);
      }
      std::sort(near.begin(), near.end());
      if (near.size() > 4)
        near.resize(4);
      if (near.empty())
        continue;

      const auto& view_i = context.photo_views[i];
      const auto& cam = context.model.cameras[cid];
      struct SrcPx {
        Eigen::Vector3d world;
        float lum, rho;
      };
      std::vector<SrcPx> src_pts;
      for (int y = 0; y < grid.height; ++y) {
        const uint8_t* m = view_i.static_mask.ptr<uint8_t>(y);
        const uint8_t* sm = view_i.depth_smooth.ptr<uint8_t>(y);
        const float* srgb = view_i.lum_srgb.ptr<float>(y);
        const float* grad = view_i.grad_srgb.ptr<float>(y);
        const float* lum = view_i.lum_lin.ptr<float>(y);
        const float* dep = view_i.depth.ptr<float>(y);
        for (int x = 0; x < grid.width; ++x) {
          if (!m[x] || !sm[x] || srgb[x] <= 0.12f || srgb[x] >= 0.88f || grad[x] >= 0.12f ||
              lum[x] <= 1e-4f)
            continue;
          const size_t gidx = size_t(y) * size_t(grid.width) + size_t(x);
          const Eigen::Vector3d moved = grid.rays[gidx] * double(dep[x]);
          const Eigen::Vector3d cam_pt =
              context.motion
                  ? context.motion->Reference(
                        i, cid,
                        context.motion->Fraction(
                            cam, {(x + 0.5) / grid.scale_x, (y + 0.5) / grid.scale_y}),
                        moved)
                  : moved;
          const Eigen::Vector3d world =
              context.rotations[i].transpose() * (cam_pt - context.translations[i]);
          src_pts.push_back({world, lum[x], grid.rho[gidx]});
        }
      }
      if (src_pts.size() < 300)
        continue;

      std::vector<VigSample> pair_samples;
      for (const auto& [dist_c, j] : near) {
        (void)dist_c;
        const auto& view_j = context.photo_views[j];
        pair_samples.clear();
        for (const auto& sp : src_pts) {
          double u = 0.0, v = 0.0, d_proj = 0.0;
          float rho_j = 0.f;
          if (!ProjectWorldToPhoto(context, j, sp.world, u, v, d_proj, rho_j))
            continue;
          const int ui = int(std::lround(u)), vi = int(std::lround(v));
          if (!view_j.static_mask.at<uint8_t>(vi, ui))
            continue;
          const float d_j = view_j.depth.at<float>(vi, ui);
          if (d_j <= 0.4f || std::abs(d_proj - double(d_j)) > std::max(0.06, 0.04 * d_proj))
            continue;
          const float s_j = SampleBilinear(view_j.lum_srgb, u, v);
          if (s_j <= 0.12f || s_j >= 0.88f)
            continue;
          const float l_j = SampleBilinear(view_j.lum_lin, u, v);
          if (l_j <= 1e-4f)
            continue;
          const double r2i = double(sp.rho) * double(sp.rho);
          const double r2j = double(rho_j) * double(rho_j);
          pair_samples.push_back(
              {r2i - r2j, r2i * r2i - r2j * r2j, r2i * r2i * r2i - r2j * r2j * r2j,
               std::log(double(sp.lum)) - std::log(double(l_j))});
        }
        if (pair_samples.size() < 300)
          continue;
        const size_t step = std::max<size_t>(1, pair_samples.size() / 200);
        double m2 = 0.0, m4 = 0.0, m6 = 0.0, ml = 0.0;
        size_t cnt = 0;
        for (size_t k = 0; k < pair_samples.size(); k += step) {
          m2 += pair_samples[k].dr2;
          m4 += pair_samples[k].dr4;
          m6 += pair_samples[k].dr6;
          ml += pair_samples[k].dl;
          ++cnt;
        }
        m2 /= double(cnt);
        m4 /= double(cnt);
        m6 /= double(cnt);
        ml /= double(cnt);
        for (size_t k = 0; k < pair_samples.size(); k += step) {
          samples.push_back({pair_samples[k].dr2 - m2, pair_samples[k].dr4 - m4,
                             pair_samples[k].dr6 - m6, pair_samples[k].dl - ml});
        }
      }
    }
    if (samples.size() < 2000)
      continue;
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    std::vector<double> w(samples.size(), 1.0);
    for (int iter = 0; iter < 5; ++iter) {
      Eigen::Matrix3d ata = Eigen::Matrix3d::Zero();
      Eigen::Vector3d atb = Eigen::Vector3d::Zero();
      for (size_t k = 0; k < samples.size(); ++k) {
        const Eigen::Vector3d a(samples[k].dr2, samples[k].dr4, samples[k].dr6);
        const double wk = w[k];
        ata.noalias() += wk * (a * a.transpose());
        atb.noalias() += wk * samples[k].dl * a;
      }
      ata.diagonal().array() += 1e-9;
      c = ata.ldlt().solve(atb);
      for (size_t k = 0; k < samples.size(); ++k) {
        const double res = std::abs(c[0] * samples[k].dr2 + c[1] * samples[k].dr4 +
                                    c[2] * samples[k].dr6 - samples[k].dl);
        w[k] = res < 0.15 ? 1.0 : 0.15 / std::max(res, 1e-6);
      }
    }
    if (c.allFinite())
      context.vig_coeffs[cid] = {c[0], c[1], c[2]};
  }

  const cv::Mat k3 = cv::Mat::ones(3, 3, CV_8UC1);
  for (const uint32_t id : written) {
    const uint32_t cid = context.model.images[id].camera;
    const auto& grid = context.photo_grids[cid];
    const auto& vc = context.vig_coeffs[cid];
    auto& view = context.photo_views[id];
    view.p_lin.create(grid.height, grid.width, CV_32FC1);
    constexpr float inv_gamma = 1.f / 2.2f;
    for (int y = 0; y < grid.height; ++y) {
      float* lp = view.lum_lin.ptr<float>(y);
      cv::Vec3f* rp = view.rgb_lin.ptr<cv::Vec3f>(y);
      float* pp = view.p_lin.ptr<float>(y);
      for (int x = 0; x < grid.width; ++x) {
        const double r2 = double(grid.rho[size_t(y) * size_t(grid.width) + size_t(x)]);
        const double rr = r2 * r2;
        const float vig = float(std::clamp(
            std::exp(vc[0] * rr + vc[1] * rr * rr + vc[2] * rr * rr * rr), 0.35, 1.35));
        lp[x] /= vig;
        rp[x] /= vig;
        pp[x] = std::pow(std::max(lp[x], 0.f), inv_gamma);
      }
    }
    cv::erode(view.p_lin, view.p_min3, k3);
    cv::dilate(view.p_lin, view.p_max3, k3);
  }
}
cv::Mat FilterComponents(const cv::Mat& raw, int min_area) {
  cv::Mat labels, stats, centroids;
  const int count = cv::connectedComponentsWithStats(raw, labels, stats, centroids, 8);
  cv::Mat out = cv::Mat::zeros(raw.size(), CV_8UC1);
  for (int k = 1; k < count; ++k) {
    if (stats.at<int>(k, cv::CC_STAT_AREA) >= min_area)
      out.setTo(255, labels == k);
  }
  return out;
}
cv::Mat GrowHysteresisMask(const cv::Mat& seed, const cv::Mat& soft_raw, int grow_radius,
                           int dilate_radius) {
  if (cv::countNonZero(seed) == 0)
    return {};
  cv::Mat allowed;
  if (grow_radius >= 0) {
    cv::Mat reach;
    cv::dilate(seed, reach,
               cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                         {2 * grow_radius + 1, 2 * grow_radius + 1}));
    allowed = (soft_raw > 0) & (reach > 0);
  } else {
    allowed = soft_raw > 0;
  }
  cv::Mat grown = seed.clone();
  std::vector<cv::Point> queue;
  cv::findNonZero(seed, queue);
  const int h = seed.rows, w = seed.cols;
  for (size_t head = 0; head < queue.size(); ++head) {
    const cv::Point pt = queue[head];
    for (int dy = -1; dy <= 1; ++dy) {
      const int ny = pt.y + dy;
      if (ny < 0 || ny >= h)
        continue;
      const uint8_t* ap = allowed.ptr<uint8_t>(ny);
      uint8_t* gp = grown.ptr<uint8_t>(ny);
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy)
          continue;
        const int nx = pt.x + dx;
        if (nx >= 0 && nx < w && ap[nx] && !gp[nx]) {
          gp[nx] = 255;
          queue.emplace_back(nx, ny);
        }
      }
    }
  }
  if (dilate_radius > 0) {
    cv::Mat out;
    cv::dilate(grown, out,
               cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                         {2 * dilate_radius + 1, 2 * dilate_radius + 1}));
    return out;
  }
  return grown;
}
void SolveRelativeExposures(Context& context, const std::vector<uint32_t>& written, int workers) {
  context.exposure_calibrated = false;
  context.exposure_ref_ev = 0.0;
  context.exposure_ev.assign(context.model.images.size(), 0.0);
  context.pixel_ev_shift.assign(context.model.images.size(), 0.0);
  context.residual_ev.assign(context.model.images.size(), 0.0);
  context.exposure_valid.assign(context.model.images.size(), 0);
  context.wb_calibrated = false;
  context.wb_gains.assign(context.model.images.size(), {1.0, 1.0, 1.0});
  context.exposure_report = {{"calibrated", false},
                             {"edges", 0},
                             {"cross_lens_edges", 0},
                             {"components", 0},
                             {"largest_component_images", 0},
                             {"residual_median_ev", 0.0},
                             {"residual_p90_ev", 0.0},
                             {"ev_min", 0.0},
                             {"ev_max", 0.0},
                             {"reference_ev", 0.0},
                             {"homogenized_images", 0},
                             {"exact_zero_residual_images", 0},
                             {"residual_ev_min", 0.0},
                             {"residual_ev_max", 0.0},
                             {"pixel_shift_min", 0.0},
                             {"pixel_shift_max", 0.0}};
  context.wb_report = {{"calibrated", false},
                       {"edges", 0},
                       {"max_abs_rg_ev", 0.0},
                       {"max_abs_bg_ev", 0.0},
                       {"residual_rg_median_ev", 0.0},
                       {"residual_bg_median_ev", 0.0}};
  const size_t n = written.size();
  if (n < 2)
    return;

  struct Edge {
    size_t a = 0, b = 0;
    double med = 0.0, w = 1.0;
    double med_rg = 0.0, med_bg = 0.0, w_wb = 0.0;
  };
  std::vector<std::vector<Edge>> slot_edges(n);
  const float wb_min_lin = SrgbToLin(float(WbMinChSrgb));
  const float wb_max_lin = SrgbToLin(float(WbMaxChSrgb));

  ParallelFor(n, workers, 1, [&](size_t slot_i) {
    const uint32_t i = written[slot_i];
    const uint32_t cid_i = context.model.images[i].camera;
    const auto& cam_i = context.model.cameras[cid_i];
    const auto& grid_i = context.photo_grids[cid_i];
    const auto& view_i = context.photo_views[i];
    const Eigen::Vector3d fwd_i = context.rotations[i].row(2).transpose();

    std::vector<std::pair<double, size_t>> cand;
    for (size_t slot_j = 0; slot_j < n; ++slot_j) {
      if (slot_j == slot_i)
        continue;
      const uint32_t j = written[slot_j];
      const double dist = (context.centers[j] - context.centers[i]).norm();
      if (dist >= PhotoMaxNeighborDistM)
        continue;
      const Eigen::Vector3d fwd_j = context.rotations[j].row(2).transpose();
      if (fwd_i.dot(fwd_j) <= -0.35)
        continue;
      cand.emplace_back(dist, slot_j);
    }
    if (cand.empty())
      return;
    std::sort(cand.begin(), cand.end());
    if (cand.size() > size_t(ExposureNeighbors))
      cand.resize(size_t(ExposureNeighbors));

    struct SrcPt {
      Eigen::Vector3d world;
      float lum, b, g, r;
    };
    std::vector<SrcPt> all_pts;
    all_pts.reserve(size_t(grid_i.width) * size_t(grid_i.height) / 4);
    for (int y = 0; y < grid_i.height; ++y) {
      const uint8_t* mp = view_i.static_mask.ptr<uint8_t>(y);
      const uint8_t* sp = view_i.depth_smooth.ptr<uint8_t>(y);
      const float* sr = view_i.lum_srgb.ptr<float>(y);
      const float* lp = view_i.lum_lin.ptr<float>(y);
      const cv::Vec3f* rp = view_i.rgb_lin.ptr<cv::Vec3f>(y);
      const float* dp = view_i.depth.ptr<float>(y);
      for (int x = 0; x < grid_i.width; ++x) {
        if (!mp[x] || !sp[x] || sr[x] <= float(ExposureMinSrgb) ||
            sr[x] >= float(ExposureMaxSrgb) || lp[x] <= 1e-4f)
          continue;
        const size_t gidx = size_t(y) * size_t(grid_i.width) + size_t(x);
        const Eigen::Vector3d moved = grid_i.rays[gidx] * double(dp[x]);
        const Eigen::Vector3d cam_pt =
            context.motion
                ? context.motion->Reference(
                      i, cid_i,
                      context.motion->Fraction(
                          cam_i, {(x + 0.5) / grid_i.scale_x, (y + 0.5) / grid_i.scale_y}),
                      moved)
                : moved;
        const Eigen::Vector3d world =
            context.rotations[i].transpose() * (cam_pt - context.translations[i]);
        all_pts.push_back({world, lp[x], rp[x][0], rp[x][1], rp[x][2]});
      }
    }
    if (all_pts.size() < size_t(ExposureMinSamples))
      return;

    const size_t step = std::max<size_t>(1, all_pts.size() / 600);
    std::vector<SrcPt> pts;
    pts.reserve(640);
    for (size_t k = 0; k < all_pts.size(); k += step)
      pts.push_back(all_pts[k]);

    std::vector<double> log2_r, log2_rg, log2_bg, devs;
    log2_r.reserve(pts.size());
    log2_rg.reserve(pts.size());
    log2_bg.reserve(pts.size());
    devs.reserve(pts.size());
    for (const auto& [dist_c, slot_j] : cand) {
      (void)dist_c;
      const uint32_t j = written[slot_j];
      const auto& view_j = context.photo_views[j];
      log2_r.clear();
      log2_rg.clear();
      log2_bg.clear();
      for (const auto& pt : pts) {
        double u = 0.0, v = 0.0, d_proj = 0.0;
        float rho_j = 0.f;
        if (!ProjectWorldToPhoto(context, j, pt.world, u, v, d_proj, rho_j))
          continue;
        const int ui = int(std::lround(u)), vi = int(std::lround(v));
        if (!view_j.static_mask.at<uint8_t>(vi, ui))
          continue;
        const float d_j = view_j.depth.at<float>(vi, ui);
        if (d_j <= 0.30f || std::abs(d_proj - double(d_j)) > std::max(0.08, 0.05 * d_proj))
          continue;
        const float s_samp = SampleBilinear(view_j.lum_srgb, u, v);
        if (s_samp <= float(ExposureMinSrgb) || s_samp >= float(ExposureMaxSrgb))
          continue;
        const float l_samp = SampleBilinear(view_j.lum_lin, u, v);
        if (l_samp <= 1e-4f)
          continue;
        log2_r.push_back(std::log2(double(pt.lum) / double(l_samp)));
        if (std::min({pt.b, pt.g, pt.r}) > wb_min_lin &&
            std::max({pt.b, pt.g, pt.r}) < wb_max_lin) {
          const cv::Vec3f c_samp = SampleBilinearVec3(view_j.rgb_lin, u, v);
          if (std::min({c_samp[0], c_samp[1], c_samp[2]}) > wb_min_lin &&
              std::max({c_samp[0], c_samp[1], c_samp[2]}) < wb_max_lin) {
            log2_rg.push_back(std::log2((double(pt.r) / double(pt.g)) /
                                        (double(c_samp[2]) / double(c_samp[1]))));
            log2_bg.push_back(std::log2((double(pt.b) / double(pt.g)) /
                                        (double(c_samp[0]) / double(c_samp[1]))));
          }
        }
      }
      if (log2_r.size() < size_t(ExposureMinSamples))
        continue;
      const double med = MedianValue(log2_r);
      if (!std::isfinite(med) || std::abs(med) > ExposureMaxPairEv)
        continue;
      devs.resize(log2_r.size());
      for (size_t k = 0; k < log2_r.size(); ++k)
        devs[k] = std::abs(log2_r[k] - med);
      const double mad = MedianValue(devs);
      if (!std::isfinite(mad) || mad > ExposureMaxPairMad)
        continue;
      const double w = std::min(double(log2_r.size()), 400.0) / std::max(mad * mad + 0.02, 0.02);

      Edge edge{slot_i, slot_j, med, w, 0.0, 0.0, 0.0};
      if (log2_rg.size() >= size_t(ExposureMinSamples)) {
        const double med_rg = MedianValue(log2_rg);
        const double med_bg = MedianValue(log2_bg);
        if (std::isfinite(med_rg) && std::isfinite(med_bg) && std::abs(med_rg) <= WbMaxPairEv &&
            std::abs(med_bg) <= WbMaxPairEv) {
          edge.med_rg = med_rg;
          edge.med_bg = med_bg;
          edge.w_wb = std::min(double(log2_rg.size()), 400.0) / 100.0;
        }
      }
      slot_edges[slot_i].push_back(edge);
    }
  });

  std::vector<Edge> edges;
  std::vector<std::vector<size_t>> adj(n);
  size_t cross_lens_edges = 0, wb_edge_count = 0;
  for (size_t s = 0; s < n; ++s) {
    for (const auto& e : slot_edges[s]) {
      edges.push_back(e);
      adj[e.a].push_back(e.b);
      adj[e.b].push_back(e.a);
      if (context.model.images[written[e.a]].camera != context.model.images[written[e.b]].camera)
        ++cross_lens_edges;
      if (e.w_wb > 0.0)
        ++wb_edge_count;
    }
  }
  if (edges.empty())
    return;

  std::vector<int> comp_id(n, -1);
  std::vector<std::vector<size_t>> components;
  std::vector<size_t> queue;
  for (size_t s = 0; s < n; ++s) {
    if (comp_id[s] >= 0)
      continue;
    const int cid = int(components.size());
    comp_id[s] = cid;
    queue.clear();
    queue.push_back(s);
    for (size_t head = 0; head < queue.size(); ++head) {
      const size_t u = queue[head];
      for (const size_t v : adj[u]) {
        if (comp_id[v] < 0) {
          comp_id[v] = cid;
          queue.push_back(v);
        }
      }
    }
    components.push_back(queue);
  }

  std::vector<std::vector<Edge>> comp_edges(components.size());
  for (const auto& e : edges)
    comp_edges[size_t(comp_id[e.a])].push_back(e);

  size_t largest_comp = 0;
  bool all_finite = true, all_wb_finite = wb_edge_count > 0;
  std::vector<size_t> local_idx(n, 0);
  std::vector<double> wb_r_ev(context.model.images.size(), 0.0);
  std::vector<double> wb_b_ev(context.model.images.size(), 0.0);
  auto solve_wb_channel = [&](const std::vector<size_t>& comp, const std::vector<Edge>& c_edges,
                              bool is_red, Eigen::VectorXd& out_vec) -> bool {
    const int k_dim = int(comp.size());
    out_vec = Eigen::VectorXd::Zero(k_dim);
    std::vector<double> hw(c_edges.size(), 1.0);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(c_edges.size() * 4 + size_t(k_dim));
    for (int iter = 0; iter < WbIters; ++iter) {
      triplets.clear();
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k_dim);
      for (size_t ei = 0; ei < c_edges.size(); ++ei) {
        const auto& ed = c_edges[ei];
        if (ed.w_wb <= 0.0)
          continue;
        const int a = int(local_idx[ed.a]), b = int(local_idx[ed.b]);
        const double wij = ed.w_wb * hw[ei];
        const double obs = is_red ? ed.med_rg : ed.med_bg;
        triplets.emplace_back(a, a, wij);
        triplets.emplace_back(b, b, wij);
        triplets.emplace_back(a, b, -wij);
        triplets.emplace_back(b, a, -wij);
        rhs[a] += wij * obs;
        rhs[b] -= wij * obs;
      }
      for (int d = 0; d < k_dim; ++d)
        triplets.emplace_back(d, d, WbPriorWeight);
      Eigen::SparseMatrix<double> mat(k_dim, k_dim);
      mat.setFromTriplets(triplets.begin(), triplets.end());
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(mat);
      if (solver.info() != Eigen::Success)
        return false;
      const Eigen::VectorXd sol = solver.solve(rhs);
      if (solver.info() != Eigen::Success || !sol.allFinite())
        return false;
      std::vector<double> tmp(sol.data(), sol.data() + k_dim);
      const double med = MedianValue(tmp);
      out_vec = sol.array() - med;
      for (size_t ei = 0; ei < c_edges.size(); ++ei) {
        const auto& ed = c_edges[ei];
        if (ed.w_wb <= 0.0)
          continue;
        const double obs = is_red ? ed.med_rg : ed.med_bg;
        const double res = std::abs((out_vec[Eigen::Index(local_idx[ed.a])] -
                                     out_vec[Eigen::Index(local_idx[ed.b])]) -
                                    obs);
        hw[ei] = res < WbHuberDeltaEv ? 1.0 : WbHuberDeltaEv / std::max(res, 1e-6);
      }
    }
    return out_vec.allFinite();
  };

  for (size_t c = 0; c < components.size(); ++c) {
    const auto& comp = components[c];
    largest_comp = std::max(largest_comp, comp.size());
    if (comp.size() < 2 || comp_edges[c].empty())
      continue;
    const int k_dim = int(comp.size());
    for (size_t lk = 0; lk < comp.size(); ++lk)
      local_idx[comp[lk]] = lk;

    Eigen::VectorXd e_vec = Eigen::VectorXd::Zero(k_dim);
    std::vector<double> hw(comp_edges[c].size(), 1.0);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(comp_edges[c].size() * 4 + size_t(k_dim));
    for (int iter = 0; iter < ExposureIters; ++iter) {
      triplets.clear();
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k_dim);
      for (size_t ei = 0; ei < comp_edges[c].size(); ++ei) {
        const auto& ed = comp_edges[c][ei];
        const int a = int(local_idx[ed.a]);
        const int b = int(local_idx[ed.b]);
        const double wij = ed.w * hw[ei];
        triplets.emplace_back(a, a, wij);
        triplets.emplace_back(b, b, wij);
        triplets.emplace_back(a, b, -wij);
        triplets.emplace_back(b, a, -wij);
        rhs[a] += wij * ed.med;
        rhs[b] -= wij * ed.med;
      }
      for (int d = 0; d < k_dim; ++d)
        triplets.emplace_back(d, d, d == 0 ? 1.0 + 1e-9 : 1e-9);
      Eigen::SparseMatrix<double> mat(k_dim, k_dim);
      mat.setFromTriplets(triplets.begin(), triplets.end());
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(mat);
      if (solver.info() != Eigen::Success) {
        all_finite = false;
        break;
      }
      const Eigen::VectorXd sol = solver.solve(rhs);
      if (solver.info() != Eigen::Success || !sol.allFinite()) {
        all_finite = false;
        break;
      }
      e_vec = sol.array() - sol.mean();
      for (size_t ei = 0; ei < comp_edges[c].size(); ++ei) {
        const auto& ed = comp_edges[c][ei];
        const double res =
            std::abs((e_vec[Eigen::Index(local_idx[ed.a])] -
                      e_vec[Eigen::Index(local_idx[ed.b])]) -
                     ed.med);
        hw[ei] = res < ExposureHuberDeltaEv ? 1.0 : ExposureHuberDeltaEv / std::max(res, 1e-6);
      }
    }
    if (!e_vec.allFinite()) {
      all_finite = false;
      continue;
    }
    Eigen::VectorXd r_vec, b_vec;
    const bool wb_ok = all_wb_finite && solve_wb_channel(comp, comp_edges[c], true, r_vec) &&
                       solve_wb_channel(comp, comp_edges[c], false, b_vec);
    all_wb_finite = all_wb_finite && wb_ok;
    for (size_t lk = 0; lk < comp.size(); ++lk) {
      const uint32_t img_id = written[comp[lk]];
      context.exposure_ev[img_id] = std::clamp(e_vec[Eigen::Index(lk)], -12.0, 12.0);
      context.exposure_valid[img_id] = 1;
      if (wb_ok) {
        const double er = std::clamp(r_vec[Eigen::Index(lk)], -WbMaxAbsEv, WbMaxAbsEv);
        const double eb = std::clamp(b_vec[Eigen::Index(lk)], -WbMaxAbsEv, WbMaxAbsEv);
        wb_r_ev[img_id] = er;
        wb_b_ev[img_id] = eb;
        const double gr = std::exp2(-er), gg = 1.0, gb = std::exp2(-eb);
        const double norm = 0.2126 * gr + 0.7152 * gg + 0.0722 * gb;
        context.wb_gains[img_id] = {gr / norm, gg / norm, gb / norm};
      }
    }
  }

  std::vector<double> residuals, rg_res, bg_res;
  residuals.reserve(edges.size());
  rg_res.reserve(edges.size());
  bg_res.reserve(edges.size());
  for (const auto& ed : edges) {
    const uint32_t ia = written[ed.a], ib = written[ed.b];
    if (context.exposure_valid[ia] && context.exposure_valid[ib]) {
      residuals.push_back(std::abs((context.exposure_ev[ia] - context.exposure_ev[ib]) - ed.med));
      if (all_wb_finite && ed.w_wb > 0.0) {
        rg_res.push_back(std::abs((wb_r_ev[ia] - wb_r_ev[ib]) - ed.med_rg));
        bg_res.push_back(std::abs((wb_b_ev[ia] - wb_b_ev[ib]) - ed.med_bg));
      }
    }
  }
  double median_res = 0.0, p90_res = 0.0;
  if (!residuals.empty()) {
    median_res = MedianValue(residuals);
    const size_t p_idx = std::min(residuals.size() - 1, size_t(0.9 * double(residuals.size())));
    std::nth_element(residuals.begin(), residuals.begin() + p_idx, residuals.end());
    p90_res = residuals[p_idx];
  }
  double ev_min = 0.0, ev_max = 0.0;
  bool has_ev = false;
  std::vector<double> valid_evs;
  for (const uint32_t id : written) {
    if (!context.exposure_valid[id])
      continue;
    const double ev = context.exposure_ev[id];
    valid_evs.push_back(ev);
    if (!has_ev) {
      ev_min = ev_max = ev;
      has_ev = true;
    } else {
      ev_min = std::min(ev_min, ev);
      ev_max = std::max(ev_max, ev);
    }
  }
  if (has_ev) {
    if (ev_max - ev_min <= 1e-6 || valid_evs.size() < 2) {
      context.exposure_ref_ev = ev_min;
    } else {
      constexpr int kKdeBins = 400;
      const double inv_2bw2 = 0.5 / (HomogKdeBandwidthEv * HomogKdeBandwidthEv);
      double best_score = -1.0, best_ev = ev_min;
      for (int b = 0; b < kKdeBins; ++b) {
        const double g = ev_min + (ev_max - ev_min) * (double(b) / double(kKdeBins - 1));
        double score = 0.0;
        for (const double e : valid_evs) {
          const double d = g - e;
          score += std::exp(-d * d * inv_2bw2);
        }
        if (score > best_score) {
          best_score = score;
          best_ev = g;
        }
      }
      context.exposure_ref_ev = best_ev;
    }
  }
  const size_t min_comp_images =
      size_t(std::ceil(ExposureMinComponentShare * double(written.size())));
  context.exposure_calibrated = all_finite && has_ev && largest_comp >= min_comp_images &&
                                !residuals.empty() && median_res <= ExposureMaxMedianResidualEv;
  context.wb_calibrated = context.exposure_calibrated && all_wb_finite && !rg_res.empty();
  double max_abs_rg = 0.0, max_abs_bg = 0.0;
  if (context.wb_calibrated) {
    for (const uint32_t id : written) {
      if (context.exposure_valid[id]) {
        max_abs_rg = std::max(max_abs_rg, std::abs(wb_r_ev[id]));
        max_abs_bg = std::max(max_abs_bg, std::abs(wb_b_ev[id]));
      }
    }
  }
  context.exposure_report = {{"calibrated", context.exposure_calibrated},
                             {"edges", edges.size()},
                             {"cross_lens_edges", cross_lens_edges},
                             {"components", components.size()},
                             {"largest_component_images", largest_comp},
                             {"residual_median_ev", median_res},
                             {"residual_p90_ev", p90_res},
                             {"ev_min", ev_min},
                             {"ev_max", ev_max},
                             {"reference_ev", context.exposure_ref_ev}};
  context.wb_report = {{"calibrated", context.wb_calibrated},
                       {"edges", wb_edge_count},
                       {"max_abs_rg_ev", max_abs_rg},
                       {"max_abs_bg_ev", max_abs_bg},
                       {"residual_rg_median_ev", rg_res.empty() ? 0.0 : MedianValue(rg_res)},
                       {"residual_bg_median_ev", bg_res.empty() ? 0.0 : MedianValue(bg_res)}};
}
void ComputeHomogenizationPlan(Context& context, const std::vector<uint32_t>& written) {
  if (!context.exposure_calibrated)
    return;
  size_t homog_cnt = 0, exact_zero_cnt = 0;
  double res_min = 0.0, res_max = 0.0, shift_min = 0.0, shift_max = 0.0;
  bool first = true;
  std::vector<float> vals;
  for (const uint32_t id : written) {
    if (!context.exposure_valid[id])
      continue;
    const auto& view = context.photo_views[id];
    const auto& grid = context.photo_grids[context.model.images[id].camera];
    vals.clear();
    vals.reserve(size_t(grid.width) * size_t(grid.height));
    const bool has_t = !view.transient_mask.empty();
    for (int y = 0; y < grid.height; ++y) {
      const uint8_t* mp = view.static_mask.ptr<uint8_t>(y);
      const uint8_t* tp = has_t ? view.transient_mask.ptr<uint8_t>(y) : nullptr;
      const float* sr = view.lum_srgb.ptr<float>(y);
      for (int x = 0; x < grid.width; ++x)
        if (mp[x] && (!tp || (tp[x] & 4) == 0))
          vals.push_back(sr[x]);
    }
    if (vals.empty()) {
      for (int y = 0; y < grid.height; ++y) {
        const uint8_t* mp = view.static_mask.ptr<uint8_t>(y);
        const float* sr = view.lum_srgb.ptr<float>(y);
        for (int x = 0; x < grid.width; ++x)
          if (mp[x])
            vals.push_back(sr[x]);
      }
    }
    double p50 = 0.5;
    if (!vals.empty()) {
      const size_t i50 = vals.size() / 2;
      std::nth_element(vals.begin(), vals.begin() + i50, vals.end());
      p50 = double(vals[i50]);
    }
    const double want = context.exposure_ev[id] - context.exposure_ref_ev;
    double shift = 0.0;
    if (want > 0.0) {
      const double t50 = std::min(p50, HomogMinP50Srgb);
      const double md50 =
          p50 > t50 ? std::log2(std::max<double>(SrgbToLin(float(p50)), 1e-6) /
                                std::max<double>(SrgbToLin(float(t50)), 1e-6))
                    : 0.0;
      shift = std::min({want, HomogMaxDarkenEv, std::max(0.0, md50)});
    } else {
      shift = std::max(want, -HomogMaxBrightenEv);
    }
    const double residual = want - shift;
    context.pixel_ev_shift[id] = shift;
    context.residual_ev[id] = residual;
    ++homog_cnt;
    if (std::abs(residual) < 0.01)
      ++exact_zero_cnt;
    if (first) {
      res_min = res_max = residual;
      shift_min = shift_max = shift;
      first = false;
    } else {
      res_min = std::min(res_min, residual);
      res_max = std::max(res_max, residual);
      shift_min = std::min(shift_min, shift);
      shift_max = std::max(shift_max, shift);
    }
  }
  context.exposure_report["homogenized_images"] = homog_cnt;
  context.exposure_report["exact_zero_residual_images"] = exact_zero_cnt;
  context.exposure_report["residual_ev_min"] = res_min;
  context.exposure_report["residual_ev_max"] = res_max;
  context.exposure_report["pixel_shift_min"] = shift_min;
  context.exposure_report["pixel_shift_max"] = shift_max;
}
void DetectPhotometricTransients(Context& context, const std::vector<uint32_t>& written,
                                 int workers) {
  const auto started = Clock::now();
  if (written.size() < size_t(PhotoMinSeen + 1))
    return;
  struct OpenCvThreadsGuard {
    int previous = cv::getNumThreads();
    explicit OpenCvThreadsGuard(int threads) { cv::setNumThreads(threads); }
    ~OpenCvThreadsGuard() { cv::setNumThreads(previous); }
  } opencv_threads_guard(1);

  for (const uint32_t id : written) {
    auto& view = context.photo_views[id];
    if (!context.keep_sky && !view.raw_sky_small.empty())
      view.static_mask |= view.raw_sky_small;
    view.raw_sky_small.release();
  }
  FitVignetting(context, written);
  SolveRelativeExposures(context, written, workers);

  const cv::Mat k3 = cv::Mat::ones(3, 3, CV_8UC1);
  const cv::Mat person_reach_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, {2 * PhotoPersonShadowReach + 1, 2 * PhotoPersonShadowReach + 1});

  ParallelFor(written.size(), workers, 1, [&](size_t idx) {
    const uint32_t i = written[idx];
    const uint32_t cid_i = context.model.images[i].camera;
    const auto& cam_i = context.model.cameras[cid_i];
    const auto& grid_i = context.photo_grids[cid_i];
    const auto& view_i = context.photo_views[i];
    const Eigen::Vector3d fwd_i = context.rotations[i].row(2).transpose();

    std::vector<std::pair<double, uint32_t>> cand;
    for (const uint32_t j : written) {
      if (j == i)
        continue;
      const double dist = (context.centers[j] - context.centers[i]).norm();
      if (dist >= PhotoMaxNeighborDistM)
        continue;
      const Eigen::Vector3d fwd_j = context.rotations[j].row(2).transpose();
      if (fwd_i.dot(fwd_j) <= -0.35)
        continue;
      cand.emplace_back(dist, j);
    }
    std::sort(cand.begin(), cand.end());
    std::vector<uint32_t> nbrs;
    nbrs.reserve(size_t(PhotoNeighbors));
    std::vector<uint8_t> used(cand.size(), 0);
    for (size_t k = 0; k < cand.size() && nbrs.size() < size_t(PhotoNeighbors); ++k) {
      const uint32_t j = cand[k].second;
      if (context.model.images[j].camera != cid_i ||
          std::abs(context.seconds[j] - context.seconds[i]) >= PhotoMinTimeGapS) {
        nbrs.push_back(j);
        used[k] = 1;
      }
    }
    for (size_t k = 0; k < cand.size() && nbrs.size() < size_t(PhotoNeighbors); ++k) {
      if (!used[k])
        nbrs.push_back(cand[k].second);
    }
    cv::Mat dark_mask, bright_mask;
    if (nbrs.size() >= size_t(PhotoMinSeen)) {
      cv::Mat p_min3_i, p_max3_i;
      cv::erode(view_i.p_lin, p_min3_i, k3);
      cv::dilate(view_i.p_lin, p_max3_i, k3);

      bool enable_person_near = false;
      for (const auto& c : context.candidates[i]) {
        if (c.reason != "static") {
          enable_person_near = true;
          break;
        }
      }
      cv::Mat person_near;
      if (enable_person_near && !view_i.person_small.empty() &&
          cv::countNonZero(view_i.person_small) > 0) {
        cv::dilate(view_i.person_small, person_near, person_reach_kernel);
      }

      struct ValidPx {
        int x, y;
        float p_i, p_min3_i, p_max3_i, lum_i, srgb_i, grad_i;
        uint8_t near_person;
        Eigen::Vector3d world;
      };
      std::vector<ValidPx> valid;
      valid.reserve(size_t(grid_i.width) * size_t(grid_i.height) / 2);
      for (int y = 0; y < grid_i.height; ++y) {
        const uint8_t* mp = view_i.static_mask.ptr<uint8_t>(y);
        const uint8_t* sp = view_i.depth_smooth.ptr<uint8_t>(y);
        const uint8_t* np = person_near.empty() ? nullptr : person_near.ptr<uint8_t>(y);
        const float* pp = view_i.p_lin.ptr<float>(y);
        const float* pminp = p_min3_i.ptr<float>(y);
        const float* pmaxp = p_max3_i.ptr<float>(y);
        const float* lp = view_i.lum_lin.ptr<float>(y);
        const float* sr = view_i.lum_srgb.ptr<float>(y);
        const float* gr = view_i.grad_srgb.ptr<float>(y);
        const float* dp = view_i.depth.ptr<float>(y);
        for (int x = 0; x < grid_i.width; ++x) {
          if (!mp[x] || !sp[x])
            continue;
          const size_t gidx = size_t(y) * size_t(grid_i.width) + size_t(x);
          const Eigen::Vector3d moved = grid_i.rays[gidx] * double(dp[x]);
          const Eigen::Vector3d cam_pt =
              context.motion
                  ? context.motion->Reference(
                        i, cid_i,
                        context.motion->Fraction(
                            cam_i, {(x + 0.5) / grid_i.scale_x, (y + 0.5) / grid_i.scale_y}),
                        moved)
                  : moved;
          const Eigen::Vector3d world =
              context.rotations[i].transpose() * (cam_pt - context.translations[i]);
          valid.push_back({x, y, pp[x], pminp[x], pmaxp[x], lp[x], sr[x], gr[x],
                           uint8_t(np && np[x] ? 1 : 0), world});
        }
      }
      if (!valid.empty()) {
        std::vector<uint16_t> seen(valid.size(), 0), dark_global(valid.size(), 0),
            dark_person(valid.size(), 0), dark_soft(valid.size(), 0),
            bright_global(valid.size(), 0), bright_soft(valid.size(), 0);
        struct Hit {
          size_t k;
          float p_lo, p_hi;
        };
        std::vector<Hit> hits;
        std::vector<float> ratios;
        hits.reserve(valid.size());
        ratios.reserve(valid.size());

        for (const uint32_t j : nbrs) {
          const auto& view_j = context.photo_views[j];
          hits.clear();
          ratios.clear();
          for (size_t k = 0; k < valid.size(); ++k) {
            const auto& px = valid[k];
            double u = 0.0, v = 0.0, d_proj = 0.0;
            float rho_j = 0.f;
            if (!ProjectWorldToPhoto(context, j, px.world, u, v, d_proj, rho_j))
              continue;
            const int ui = int(std::lround(u)), vi = int(std::lround(v));
            if (!view_j.static_mask.at<uint8_t>(vi, ui))
              continue;
            const float d_j = view_j.depth.at<float>(vi, ui);
            if (d_j <= 0.30f || std::abs(d_proj - double(d_j)) > std::max(0.08, 0.05 * d_proj))
              continue;
            const float l_samp = SampleBilinear(view_j.lum_lin, u, v);
            const float s_samp = SampleBilinear(view_j.lum_srgb, u, v);
            const float p_lo = SampleBilinear(view_j.p_min3, u, v);
            const float p_hi = SampleBilinear(view_j.p_max3, u, v);
            hits.push_back({k, p_lo, p_hi});
            if (px.srgb_i > 0.10f && px.srgb_i < 0.90f && s_samp > 0.10f && s_samp < 0.90f)
              ratios.push_back(px.lum_i / std::max(l_samp, 1e-5f));
          }
          if (ratios.size() < size_t(PhotoMinGainSamples))
            continue;
          const size_t mid = ratios.size() / 2;
          std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
          const float gain = ratios[mid];
          if (gain < float(PhotoMinPairGain) || gain > float(PhotoMaxPairGain))
            continue;
          const float gain_p = std::pow(gain, 1.f / 2.2f);
          for (const auto& h : hits) {
            const size_t k = h.k;
            ++seen[k];
            const auto& px = valid[k];
            const float p_lo = h.p_lo * gain_p;
            const float p_hi = h.p_hi * gain_p;
            if (p_lo > float(PhotoMinRefPerceptual)) {
              if (px.grad_i < float(PhotoDarkMaxGrad) &&
                  px.p_max3_i < p_lo * float(PhotoDarkRatio) &&
                  px.p_max3_i < p_lo - float(PhotoDarkDiff))
                ++dark_global[k];
              if (px.near_person && px.grad_i < float(PhotoPersonDarkMaxGrad) &&
                  px.p_i < p_lo * float(PhotoPersonDarkRatio) &&
                  px.p_i < p_lo - float(PhotoPersonDarkDiff))
                ++dark_person[k];
              if (px.grad_i < float(PhotoDarkGrowMaxGrad) &&
                  px.p_i < p_lo * float(PhotoDarkGrowRatio) &&
                  px.p_i < p_lo - float(PhotoDarkGrowDiff))
                ++dark_soft[k];
            }
            if (p_hi > float(PhotoMinRefPerceptual)) {
              if (px.grad_i < float(PhotoBrightMaxGrad) &&
                  px.p_min3_i > p_hi * float(PhotoBrightRatio) &&
                  px.p_min3_i > p_hi + float(PhotoBrightDiff))
                ++bright_global[k];
              if (px.grad_i < float(PhotoBrightGrowMaxGrad) &&
                  px.p_i > p_hi * float(PhotoBrightGrowRatio) &&
                  px.p_i > p_hi + float(PhotoBrightGrowDiff))
                ++bright_soft[k];
            }
          }
        }

        cv::Mat dark_global_raw = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
        cv::Mat dark_person_raw = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
        cv::Mat dark_soft_map = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
        cv::Mat bright_seed_raw = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
        cv::Mat bright_soft_map = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
        for (size_t k = 0; k < valid.size(); ++k) {
          const int x = valid[k].x, y = valid[k].y;
          const uint16_t s = seen[k];
          if (s < uint16_t(PhotoGrowMinSeen))
            continue;
          if (double(dark_soft[k]) >= PhotoDarkGrowShare * double(s))
            dark_soft_map.at<uint8_t>(y, x) = 255;
          if (double(bright_soft[k]) >= PhotoBrightGrowShare * double(s))
            bright_soft_map.at<uint8_t>(y, x) = 255;
          if (s >= uint16_t(PhotoPersonMinSeen) &&
              double(dark_person[k]) >= PhotoPersonDarkShare * double(s))
            dark_person_raw.at<uint8_t>(y, x) = 255;
          if (s < uint16_t(PhotoMinSeen))
            continue;
          if (double(dark_global[k]) >= PhotoDarkShare * double(s))
            dark_global_raw.at<uint8_t>(y, x) = 255;
          if (double(bright_global[k]) >= PhotoBrightShare * double(s))
            bright_seed_raw.at<uint8_t>(y, x) = 255;
        }

        const cv::Mat dark_seed =
            FilterComponents(dark_global_raw, PhotoMinCompArea) |
            FilterComponents(dark_person_raw, PhotoPersonMinCompArea);
        dark_mask = GrowHysteresisMask(dark_seed, dark_soft_map, PhotoDarkGrowRadius,
                                       PhotoDarkDilateRadius);
        bright_mask = GrowHysteresisMask(FilterComponents(bright_seed_raw, PhotoMinCompArea),
                                         bright_soft_map, PhotoBrightGrowRadius,
                                         PhotoBrightDilateRadius);
      }
    }

    cv::Mat clip_mask;
    if (context.exposure_calibrated && context.exposure_valid[i]) {
      const double ev_i = context.exposure_ev[i];
      struct ClipNbrCand {
        double dist = 0.0, dev = 0.0;
        uint32_t j = 0;
      };
      std::vector<ClipNbrCand> clip_cand;
      for (const uint32_t j : written) {
        if (j == i || !context.exposure_valid[j])
          continue;
        const double dev = ev_i - context.exposure_ev[j];
        if (dev < PhotoClipMinEvDiff)
          continue;
        const double dist = (context.centers[j] - context.centers[i]).norm();
        if (dist >= PhotoClipMaxNeighborDistM)
          continue;
        const Eigen::Vector3d fwd_j = context.rotations[j].row(2).transpose();
        if (fwd_i.dot(fwd_j) <= -0.35)
          continue;
        clip_cand.push_back({dist, dev, j});
      }
      std::sort(clip_cand.begin(), clip_cand.end(),
                [](const ClipNbrCand& a, const ClipNbrCand& b) { return a.dist < b.dist; });
      std::vector<uint32_t> clip_nbrs;
      clip_nbrs.reserve(size_t(PhotoClipNeighbors));
      std::vector<uint8_t> clip_used(clip_cand.size(), 0);
      for (size_t k = 0; k < clip_cand.size() && clip_nbrs.size() < size_t(PhotoClipNearestQuota);
           ++k) {
        clip_nbrs.push_back(clip_cand[k].j);
        clip_used[k] = 1;
      }
      for (size_t k = 0; k < clip_cand.size() && clip_nbrs.size() < size_t(PhotoClipNeighbors);
           ++k) {
        if (!clip_used[k] && clip_cand[k].dev >= PhotoClipLargeEvDiff) {
          clip_nbrs.push_back(clip_cand[k].j);
          clip_used[k] = 1;
        }
      }
      for (size_t k = 0; k < clip_cand.size() && clip_nbrs.size() < size_t(PhotoClipNeighbors);
           ++k) {
        if (!clip_used[k])
          clip_nbrs.push_back(clip_cand[k].j);
      }

      cv::Mat clip_seed_raw = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
      cv::Mat clip_soft_map = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
      struct ClipPx {
        int x = 0, y = 0;
        Eigen::Vector3d world;
      };
      std::vector<ClipPx> clip_cands;
      const bool high_ev_frame = (ev_i - context.exposure_ref_ev) >= PhotoClipHighEvBoost;
      for (int y = 0; y < grid_i.height; ++y) {
        const uint8_t* mp = view_i.static_mask.ptr<uint8_t>(y);
        const float* sr = view_i.lum_srgb.ptr<float>(y);
        const float* msr = view_i.max_srgb.ptr<float>(y);
        const float* dp = view_i.depth.ptr<float>(y);
        uint8_t* csoft = clip_soft_map.ptr<uint8_t>(y);
        uint8_t* cseed = clip_seed_raw.ptr<uint8_t>(y);
        for (int x = 0; x < grid_i.width; ++x) {
          if (!mp[x])
            continue;
          const float lum = sr[x], mx = msr[x], d = dp[x];
          if (lum >= float(PhotoClipGrowLum) || mx >= float(PhotoClipGrowMaxCh))
            csoft[x] = 255;
          if (high_ev_frame &&
              (lum >= float(PhotoClipHighEvMinLum) || mx >= float(PhotoClipHighEvMinMaxCh)) &&
              (d <= 0.f || d >= float(PhotoClipHighEvMinDepthM))) {
            cseed[x] = 255;
          }
          if (!clip_nbrs.empty() && d >= 0.30f && d <= float(PhotoClipMaxDepthM) &&
              (lum >= float(PhotoClipMinLum) || mx >= float(PhotoClipMinMaxCh))) {
            const size_t gidx = size_t(y) * size_t(grid_i.width) + size_t(x);
            const Eigen::Vector3d moved = grid_i.rays[gidx] * double(d);
            const Eigen::Vector3d cam_pt =
                context.motion
                    ? context.motion->Reference(
                          i, cid_i,
                          context.motion->Fraction(
                              cam_i, {(x + 0.5) / grid_i.scale_x, (y + 0.5) / grid_i.scale_y}),
                          moved)
                    : moved;
            const Eigen::Vector3d world =
                context.rotations[i].transpose() * (cam_pt - context.translations[i]);
            clip_cands.push_back({x, y, world});
          }
        }
      }
      if (!clip_cands.empty() && !clip_nbrs.empty()) {
        std::vector<uint16_t> darker_seen(clip_cands.size(), 0), unclipped(clip_cands.size(), 0);
        for (const uint32_t j : clip_nbrs) {
          const auto& view_j = context.photo_views[j];
          for (size_t k = 0; k < clip_cands.size(); ++k) {
            double u = 0.0, v = 0.0, d_proj = 0.0;
            float rho_j = 0.f;
            if (!ProjectWorldToPhoto(context, j, clip_cands[k].world, u, v, d_proj, rho_j))
              continue;
            const int ui = int(std::lround(u)), vi = int(std::lround(v));
            if (!view_j.static_mask.at<uint8_t>(vi, ui))
              continue;
            const float d_j = view_j.depth.at<float>(vi, ui);
            if (d_j <= 0.30f || std::abs(d_proj - double(d_j)) > std::max(0.15, 0.08 * d_proj))
              continue;
            ++darker_seen[k];
            const float s_samp = SampleBilinear(view_j.lum_srgb, u, v);
            const float max_samp = SampleBilinear(view_j.max_srgb, u, v);
            if (s_samp < float(PhotoClipRefMaxLum) && max_samp < float(PhotoClipRefMaxCh))
              ++unclipped[k];
          }
        }
        for (size_t k = 0; k < clip_cands.size(); ++k) {
          if (darker_seen[k] >= uint16_t(PhotoClipMinSeen) &&
              double(unclipped[k]) >= PhotoClipShare * double(darker_seen[k])) {
            clip_seed_raw.at<uint8_t>(clip_cands[k].y, clip_cands[k].x) = 255;
          }
        }
      }
      clip_mask = GrowHysteresisMask(FilterComponents(clip_seed_raw, PhotoClipMinCompArea),
                                     clip_soft_map, -1, 0);
      if (!clip_mask.empty()) {
        const bool bloom_frame = (ev_i - context.exposure_ref_ev) >= PhotoClipBloomEvBoost;
        if (bloom_frame) {
          cv::morphologyEx(
              clip_mask, clip_mask, cv::MORPH_CLOSE,
              cv::getStructuringElement(
                  cv::MORPH_ELLIPSE,
                  {2 * PhotoClipBloomCloseRadius + 1, 2 * PhotoClipBloomCloseRadius + 1}));
          clip_mask &= view_i.static_mask;
        }
        const int dilate_r = bloom_frame ? PhotoClipHighEvDilateRadius : PhotoClipDilateRadius;
        if (dilate_r > 0) {
          cv::dilate(clip_mask, clip_mask,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               {2 * dilate_r + 1, 2 * dilate_r + 1}));
        }
      }
    }

    const int static_count = cv::countNonZero(view_i.static_mask);
    if (static_count > 0 && (!dark_mask.empty() || !bright_mask.empty())) {
      cv::Mat db_mask = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
      if (!dark_mask.empty())
        db_mask |= dark_mask;
      if (!bright_mask.empty())
        db_mask |= bright_mask;
      const int db_count = cv::countNonZero(db_mask & view_i.static_mask);
      if (double(db_count) > PhotoMaxImageMaskShare * double(static_count)) {
        context.photo_views[i].capped = true;
        dark_mask.release();
        bright_mask.release();
      }
    }
    if (static_count > 0 && !clip_mask.empty()) {
      const int clip_count = cv::countNonZero(clip_mask & view_i.static_mask);
      if (double(clip_count) > PhotoMaxClipMaskShare * double(static_count)) {
        context.photo_views[i].clip_capped = true;
        clip_mask.release();
      }
    }
    if (dark_mask.empty() && bright_mask.empty() && clip_mask.empty())
      return;

    cv::Mat combined = cv::Mat::zeros(grid_i.height, grid_i.width, CV_8UC1);
    if (!dark_mask.empty())
      combined.setTo(1, dark_mask > 0);
    if (!bright_mask.empty()) {
      for (int y = 0; y < grid_i.height; ++y) {
        const uint8_t* bp = bright_mask.ptr<uint8_t>(y);
        uint8_t* cp = combined.ptr<uint8_t>(y);
        for (int x = 0; x < grid_i.width; ++x)
          if (bp[x])
            cp[x] |= 2;
      }
    }
    if (!clip_mask.empty()) {
      for (int y = 0; y < grid_i.height; ++y) {
        const uint8_t* clp = clip_mask.ptr<uint8_t>(y);
        uint8_t* cp = combined.ptr<uint8_t>(y);
        for (int x = 0; x < grid_i.width; ++x)
          if (clp[x])
            cp[x] |= 4;
      }
    }
    context.photo_views[i].transient_mask = std::move(combined);
  });

  ComputeHomogenizationPlan(context, written);

  for (const uint32_t id : written) {
    auto& view = context.photo_views[id];
    view.lum_lin.release();
    view.rgb_lin.release();
    view.lum_srgb.release();
    view.max_srgb.release();
    view.grad_srgb.release();
    view.p_lin.release();
    view.p_min3.release();
    view.p_max3.release();
    view.depth.release();
    view.depth_smooth.release();
    view.static_mask.release();
    view.person_small.release();
  }
  context.photo_seconds = Seconds(started);
}
// Pass 3, per image: static candidates become static in the mask, unless a candidate that stays
// masked is next to them; false-positive sky components and indoor skies become static too.
ApplyCounts Apply(Context& context, uint32_t id, cv::Size grid) {
  const auto& model = context.model;
  const auto& camera = model.cameras[model.images[id].camera];
  auto& found = context.candidates[id];
  auto labelled = Label(context, id, grid);
  ApplyCounts counts;
  const bool any = std::any_of(found.begin(), found.end(),
                               [](const Candidate& c) { return c.reason == "static"; });
  if (any) {
    cv::Mat stays = labelled.transient_map.clone();
    for (const auto& c : found)
      if (c.reason == "static")
        stays.setTo(0, labelled.labels == c.label);
    const int reach = int(ShadowReach * grid.width);
    cv::dilate(stays, stays,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * reach + 1, 2 * reach + 1}));
    cv::Mat restored = cv::Mat::zeros(grid, CV_8UC1);
    for (auto& c : found) {
      if (c.reason != "static")
        continue;
      const cv::Mat region = labelled.labels == c.label;
      if (cv::countNonZero(stays & region))
        c.reason = "static_near_masked";
      else
        restored.setTo(255, region);
    }
    if (cv::countNonZero(restored)) {
      cv::Mat region;
      cv::resize(restored, region, {camera.width, camera.height}, 0, 0, cv::INTER_NEAREST);
      cv::dilate(region, region, cv::Mat::ones(5, 5, CV_8UC1));
      const cv::Mat change = region & (labelled.mask == 0) & context.domains[model.images[id].camera];
      counts.person_unmasked = uint64_t(cv::countNonZero(change));
      labelled.mask.setTo(255, change);
    }
  }
  const auto& sky_info = context.skies[id];
  if (sky_info.has_raw_sky) {
    const cv::Mat raw_sky = (labelled.mask == 128);
    if (!context.keep_sky) {
      counts.sky_cleared = uint64_t(cv::countNonZero(raw_sky));
      labelled.mask.setTo(255, raw_sky);
    } else if (std::any_of(sky_info.components.begin(), sky_info.components.end(),
                           [](const SkyComponent& c) { return c.surface; })) {
      cv::Mat sky_down;
      cv::resize(raw_sky, sky_down, grid, 0, 0, cv::INTER_AREA);
      cv::Mat sky_labels, sky_stats, sky_centroids;
      cv::connectedComponentsWithStats(sky_down > 0, sky_labels, sky_stats, sky_centroids, 8);
      cv::Mat sky_restored = cv::Mat::zeros(grid, CV_8UC1);
      for (const auto& comp : sky_info.components)
        if (comp.surface)
          sky_restored.setTo(255, sky_labels == comp.label);
      cv::dilate(sky_restored, sky_restored, cv::Mat::ones(3, 3, CV_8UC1));
      cv::Mat full_sky_restored;
      cv::resize(sky_restored, full_sky_restored, {camera.width, camera.height}, 0, 0,
                 cv::INTER_NEAREST);
      const cv::Mat sky_change = full_sky_restored & raw_sky;
      counts.sky_cleared = uint64_t(cv::countNonZero(sky_change));
      labelled.mask.setTo(255, sky_change);
    }
  }
  if (context.keep_sky && cv::countNonZero(labelled.mask == 128) > 0)
    RefineOutdoorSkyBoundary(context, id, labelled.mask, counts);
  const cv::Mat& photo_transient = context.photo_views[id].transient_mask;
  if (!photo_transient.empty()) {
    const uint32_t cam_idx = model.images[id].camera;
    const cv::Mat& full_dom = context.full_domains[cam_idx];
    const cv::Mat dark_small = (photo_transient & 1) != 0;
    const cv::Mat bright_small = (photo_transient & 2) != 0;
    const cv::Mat clip_small = (photo_transient & 4) != 0;
    cv::Mat dark_change = cv::Mat::zeros(camera.height, camera.width, CV_8UC1);
    cv::Mat bright_change = cv::Mat::zeros(camera.height, camera.width, CV_8UC1);
    if (cv::countNonZero(dark_small) > 0) {
      cv::Mat dark_up;
      cv::resize(dark_small, dark_up, {camera.width, camera.height}, 0, 0, cv::INTER_LINEAR);
      dark_change = (dark_up >= 128) & (labelled.mask == 255) & full_dom;
      counts.photo_dark = uint64_t(cv::countNonZero(dark_change));
      labelled.mask.setTo(0, dark_change);
    }
    if (cv::countNonZero(bright_small) > 0) {
      cv::Mat bright_up;
      cv::resize(bright_small, bright_up, {camera.width, camera.height}, 0, 0, cv::INTER_LINEAR);
      bright_change = (bright_up >= 128) & (labelled.mask == 255) & full_dom;
      counts.photo_bright = uint64_t(cv::countNonZero(bright_change));
      labelled.mask.setTo(0, bright_change);
    }
    if (cv::countNonZero(clip_small) > 0) {
      cv::Mat clip_up;
      cv::resize(clip_small, clip_up, {camera.width, camera.height}, 0, 0, cv::INTER_LINEAR);
      const cv::Mat clip_change = (clip_up >= 128) & (labelled.mask == 255) & full_dom;
      counts.photo_clipped = uint64_t(cv::countNonZero(clip_change));
      labelled.mask.setTo(0, clip_change);
    }
  }
  const auto path = context.destination(id);
  fs::create_directories(path.parent_path());
  WriteImage(path, labelled.mask, 1);
  return counts;
}
}  // namespace
Json StaticMaskRules() {
  return {{"flagged_share_minimum", MinRatio},
          {"span_seconds_minimum", MinSpan},
          {"views_seeing_minimum", MinSeen},
          {"max_range_m", MaxRange},
          {"shadow_reach", ShadowReach},
          {"sky_core_erosion_pixels", SkyCoreErosion},
          {"sky_las_share_minimum", SkyLasShare},
          {"sky_image_share_minimum", SkyImageMinShare},
          {"sky_scene_image_share_minimum", SkySceneMinImageShare},
          {"sky_scene_images_minimum", SkySceneMinImages},
          {"sky_edge_guided_filter_radius_4k", SkyEdgeGuidedRadius4k},
          {"sky_edge_alpha_threshold", SkyEdgeAlphaThreshold},
          {"sky_canopy_pore_max_area_4k", SkyCanopyPoreMaxArea4k},
          {"sky_canopy_foliage_border_share", SkyCanopyFoliageBorderShare},
          {"photo_grid_size", PhotoGridSize},
          {"photo_neighbors", PhotoNeighbors},
          {"photo_max_neighbor_dist_m", PhotoMaxNeighborDistM},
          {"photo_min_time_gap_s", PhotoMinTimeGapS},
          {"photo_min_depth_m", PhotoMinDepthM},
          {"photo_max_depth_m", PhotoMaxDepthM},
          {"photo_max_depth_grad", PhotoMaxDepthGrad},
          {"photo_min_gain_samples", PhotoMinGainSamples},
          {"photo_min_seen", PhotoMinSeen},
          {"photo_grow_min_seen", PhotoGrowMinSeen},
          {"photo_min_pair_gain", PhotoMinPairGain},
          {"photo_max_pair_gain", PhotoMaxPairGain},
          {"photo_min_ref_perceptual", PhotoMinRefPerceptual},
          {"photo_min_comp_area", PhotoMinCompArea},
          {"photo_max_image_mask_share", PhotoMaxImageMaskShare},
          {"photo_dark_share", PhotoDarkShare},
          {"photo_dark_ratio", PhotoDarkRatio},
          {"photo_dark_diff", PhotoDarkDiff},
          {"photo_dark_max_grad", PhotoDarkMaxGrad},
          {"photo_dark_grow_share", PhotoDarkGrowShare},
          {"photo_dark_grow_ratio", PhotoDarkGrowRatio},
          {"photo_dark_grow_diff", PhotoDarkGrowDiff},
          {"photo_dark_grow_max_grad", PhotoDarkGrowMaxGrad},
          {"photo_dark_grow_radius", PhotoDarkGrowRadius},
          {"photo_dark_dilate_radius", PhotoDarkDilateRadius},
          {"photo_bright_share", PhotoBrightShare},
          {"photo_bright_ratio", PhotoBrightRatio},
          {"photo_bright_diff", PhotoBrightDiff},
          {"photo_bright_max_grad", PhotoBrightMaxGrad},
          {"photo_bright_grow_share", PhotoBrightGrowShare},
          {"photo_bright_grow_ratio", PhotoBrightGrowRatio},
          {"photo_bright_grow_diff", PhotoBrightGrowDiff},
          {"photo_bright_grow_max_grad", PhotoBrightGrowMaxGrad},
          {"photo_bright_grow_radius", PhotoBrightGrowRadius},
          {"photo_bright_dilate_radius", PhotoBrightDilateRadius},
          {"photo_person_shadow_reach", PhotoPersonShadowReach},
          {"photo_person_min_seen", PhotoPersonMinSeen},
          {"photo_person_min_comp_area", PhotoPersonMinCompArea},
          {"photo_person_dark_share", PhotoPersonDarkShare},
          {"photo_person_dark_ratio", PhotoPersonDarkRatio},
          {"photo_person_dark_diff", PhotoPersonDarkDiff},
          {"photo_person_dark_max_grad", PhotoPersonDarkMaxGrad},
          {"exposure_neighbors", ExposureNeighbors},
          {"exposure_min_samples", ExposureMinSamples},
          {"exposure_iters", ExposureIters},
          {"exposure_min_srgb", ExposureMinSrgb},
          {"exposure_max_srgb", ExposureMaxSrgb},
          {"exposure_max_pair_ev", ExposureMaxPairEv},
          {"exposure_max_pair_mad", ExposureMaxPairMad},
          {"exposure_huber_delta_ev", ExposureHuberDeltaEv},
          {"exposure_min_component_share", ExposureMinComponentShare},
          {"exposure_max_median_residual_ev", ExposureMaxMedianResidualEv},
          {"wb_min_ch_srgb", WbMinChSrgb},
          {"wb_max_ch_srgb", WbMaxChSrgb},
          {"wb_prior_weight", WbPriorWeight},
          {"wb_max_abs_ev", WbMaxAbsEv},
          {"wb_max_pair_ev", WbMaxPairEv},
          {"wb_huber_delta_ev", WbHuberDeltaEv},
          {"wb_iters", WbIters},
          {"homog_kde_bandwidth_ev", HomogKdeBandwidthEv},
          {"homog_max_brighten_ev", HomogMaxBrightenEv},
          {"homog_max_darken_ev", HomogMaxDarkenEv},
          {"homog_min_p50_srgb", HomogMinP50Srgb},
          {"photo_clip_neighbors", PhotoClipNeighbors},
          {"photo_clip_nearest_quota", PhotoClipNearestQuota},
          {"photo_clip_large_ev_diff", PhotoClipLargeEvDiff},
          {"photo_clip_max_neighbor_dist_m", PhotoClipMaxNeighborDistM},
          {"photo_clip_max_depth_m", PhotoClipMaxDepthM},
          {"photo_clip_min_lum", PhotoClipMinLum},
          {"photo_clip_min_max_ch", PhotoClipMinMaxCh},
          {"photo_clip_min_ev_diff", PhotoClipMinEvDiff},
          {"photo_clip_ref_max_lum", PhotoClipRefMaxLum},
          {"photo_clip_ref_max_ch", PhotoClipRefMaxCh},
          {"photo_clip_min_seen", PhotoClipMinSeen},
          {"photo_clip_share", PhotoClipShare},
          {"photo_clip_min_comp_area", PhotoClipMinCompArea},
          {"photo_clip_grow_lum", PhotoClipGrowLum},
          {"photo_clip_grow_max_ch", PhotoClipGrowMaxCh},
          {"photo_clip_high_ev_boost", PhotoClipHighEvBoost},
          {"photo_clip_high_ev_min_lum", PhotoClipHighEvMinLum},
          {"photo_clip_high_ev_min_max_ch", PhotoClipHighEvMinMaxCh},
          {"photo_clip_high_ev_min_depth_m", PhotoClipHighEvMinDepthM},
          {"photo_clip_bloom_ev_boost", PhotoClipBloomEvBoost},
          {"photo_clip_bloom_close_radius", PhotoClipBloomCloseRadius},
          {"photo_clip_high_ev_dilate_radius", PhotoClipHighEvDilateRadius},
          {"photo_clip_dilate_radius", PhotoClipDilateRadius},
          {"photo_max_clip_mask_share", PhotoMaxClipMaskShare}};
}
Json StaticMasks(const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
                 const LasHeader& header, LasSurfelCache& cache,
                 const std::function<fs::path(uint32_t)>& source,
                 const std::function<fs::path(uint32_t)>& destination) {
  const auto started = Clock::now();
  Context context{scene, model, motion, source, destination};
  context.thumbs.resize(model.images.size());
  context.las_interior.resize(model.images.size());
  context.candidates.resize(model.images.size());
  context.skies.resize(model.images.size());
  context.photo_views.resize(model.images.size());
  for (int c = 0; c < 2; ++c) {
    cv::Mat domain = model.cameras[c].Mask();
    context.full_domains[c] = domain.clone();
    const auto grid_cam = MapCamera(model.cameras[c], MapSize);
    cv::Mat grid_domain;
    cv::resize(domain, grid_domain, {grid_cam.width, grid_cam.height}, 0, 0, cv::INTER_NEAREST);
    context.full_domain_grid_pixels[c] = cv::countNonZero(grid_domain);
    cv::erode(domain, domain, cv::getStructuringElement(cv::MORPH_ELLIPSE, {31, 31}));
    context.domains[c] = domain;

    auto& pg = context.photo_grids[c];
    pg.grid_cam = MapCamera(model.cameras[c], PhotoGridSize);
    pg.width = pg.grid_cam.width;
    pg.height = pg.grid_cam.height;
    pg.scale_x = double(pg.width) / double(model.cameras[c].width);
    pg.scale_y = double(pg.height) / double(model.cameras[c].height);
    pg.domain = pg.grid_cam.Mask();
    pg.rays.resize(size_t(pg.width) * size_t(pg.height));
    pg.rho.resize(size_t(pg.width) * size_t(pg.height), 0.f);
    for (int y = 0; y < pg.height; ++y) {
      const uint8_t* dp = pg.domain.ptr<uint8_t>(y);
      for (int x = 0; x < pg.width; ++x) {
        if (!dp[x])
          continue;
        const size_t gidx = size_t(y) * size_t(pg.width) + size_t(x);
        const Eigen::Vector3d ray = model.cameras[c].Ray(
            {(x + 0.5) / pg.scale_x, (y + 0.5) / pg.scale_y});
        pg.rays[gidx] = ray;
        const double theta = std::acos(std::clamp(ray.z(), -1.0, 1.0));
        pg.rho[gidx] = float(theta / (80.0 * Pi / 180.0));
      }
    }
  }
  for (const auto& image : model.images) {
    context.rotations.push_back(image.pose.Rotation().toRotationMatrix());
    context.translations.push_back(image.pose.Translation());
    context.centers.push_back(image.pose.Center());
  }
  // Exposure times from the names: week-second-millisecond_frame-lens.
  for (const auto& image : scene.images) {
    const auto file = fs::u8path(image.name).filename().u8string();
    int week, second, millisecond;
    if (std::sscanf(file.c_str(), "%d-%d-%d_", &week, &second, &millisecond) != 3)
      throw std::runtime_error("Cannot read the exposure time from " + image.name);
    context.seconds.push_back(week * 604800. + second + millisecond / 1000.);
  }
  // Pass 1: every image's candidates and their places, while the LAS is projected.
  std::deque<std::future<void>> jobs;
  size_t images = 0;
  ProjectLasDepths(cache, header, model, motion, MapSize,
                   [&](uint32_t id, cv::Mat&& depth) {
                     if (jobs.size() >= 6) {
                       jobs.front().get();
                       jobs.pop_front();
                     }
                     jobs.push_back(std::async(std::launch::async,
                                               [&context, id, depth = std::move(depth)] {
                                                 Collect(context, id, depth);
                                               }));
                     if (++images % 64 == 0)
                       Log("Static masks: located " + std::to_string(images) + "/" +
                           std::to_string(model.Registered()));
                   });
  for (auto& job : jobs)
    job.get();
  // Pass 2: judge every candidate against all views.
  const int workers = std::clamp(int(std::thread::hardware_concurrency()), 1, 16);
  std::vector<std::pair<uint32_t, size_t>> work;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    for (size_t k = 0; k < context.candidates[i].size(); ++k)
      if (context.candidates[i][k].located)
        work.emplace_back(i, k);
  ParallelFor(work.size(), workers, 4, [&](size_t i) {
    Judge(context, context.candidates[work[i].first][work[i].second]);
  });
  // Scene-level sky decision before Pass 2.5 and Pass 3.
  std::vector<uint32_t> written;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered)
      written.push_back(i);
  size_t raw_sky_images = 0, open_sky_images = 0, surface_sky_components = 0;
  for (const auto id : written) {
    const auto& s = context.skies[id];
    if (!s.has_raw_sky)
      continue;
    ++raw_sky_images;
    for (const auto& comp : s.components)
      surface_sky_components += comp.surface ? 1 : 0;
    if (s.valid_grid_pixels > 0 &&
        double(s.open_sky_grid_pixels) >= SkyImageMinShare * double(s.valid_grid_pixels))
      ++open_sky_images;
  }
  const size_t min_open_images = std::min<size_t>(SkySceneMinImages, (written.size() + 1) / 2);
  context.keep_sky =
      open_sky_images >= min_open_images &&
      double(open_sky_images) >= SkySceneMinImageShare * double(written.size());
  // Pass 2.5: multi-view photometric outlier detection (moving shadows and specular glare).
  DetectPhotometricTransients(context, written, workers);
  // Pass 3: the new masks, on the same grid as pass 1.
  const auto map_camera = [&](uint32_t id) { return MapCamera(model.cameras[model.images[id].camera], MapSize); };
  std::vector<ApplyCounts> applied(model.images.size());
  {
    struct OpenCvThreadsGuard {
      int previous = cv::getNumThreads();
      explicit OpenCvThreadsGuard(int threads) { cv::setNumThreads(threads); }
      ~OpenCvThreadsGuard() { cv::setNumThreads(previous); }
    } opencv_threads_guard(1);
    ParallelFor(written.size(), workers, 1, [&](size_t i) {
      const auto grid = map_camera(written[i]);
      applied[written[i]] = Apply(context, written[i], {grid.width, grid.height});
    });
  }
  Json rows = Json::array();
  std::map<std::string, size_t> reasons;
  uint64_t pixels = 0, sky_pixels = 0, sky_edge_to_sky = 0, sky_edge_to_static = 0;
  uint64_t photo_dark_pixels = 0, photo_bright_pixels = 0, photo_clipped_pixels = 0;
  size_t changed = 0, sky_changed = 0, sky_edge_refined = 0;
  size_t photo_dark_images = 0, photo_bright_images = 0, photo_clipped_images = 0,
         photo_capped_images = 0, photo_clip_capped_images = 0;
  for (const auto id : written) {
    pixels += applied[id].person_unmasked;
    changed += applied[id].person_unmasked > 0;
    sky_pixels += applied[id].sky_cleared;
    sky_changed += applied[id].sky_cleared > 0;
    sky_edge_to_sky += applied[id].sky_edge_to_sky;
    sky_edge_to_static += applied[id].sky_edge_to_static;
    sky_edge_refined += (applied[id].sky_edge_to_sky > 0 || applied[id].sky_edge_to_static > 0);
    photo_dark_pixels += applied[id].photo_dark;
    photo_dark_images += applied[id].photo_dark > 0;
    photo_bright_pixels += applied[id].photo_bright;
    photo_bright_images += applied[id].photo_bright > 0;
    photo_clipped_pixels += applied[id].photo_clipped;
    photo_clipped_images += applied[id].photo_clipped > 0;
    photo_capped_images += context.photo_views[id].capped ? 1 : 0;
    photo_clip_capped_images += context.photo_views[id].clip_capped ? 1 : 0;
    for (const auto& c : context.candidates[id]) {
      ++reasons[c.reason];
      Json row = {{"image_index", id},
                  {"name", scene.images[id].name},
                  {"area", c.area},
                  {"x", c.x},
                  {"y", c.y},
                  {"reason", c.reason}};
      if (c.located)
        row["place"] = {c.place.x(), c.place.y(), c.place.z()};
      if (c.seen) {
        row["seen"] = c.seen;
        row["flagged"] = c.flagged;
        row["ratio"] = double(c.flagged) / double(c.seen);
        row["span_seconds"] = c.span;
      }
      rows.push_back(std::move(row));
    }
  }
  Json exposure_ev = Json::array();
  if (context.exposure_calibrated) {
    for (const auto id : written) {
      if (context.exposure_valid[id]) {
        const auto& wb = context.wb_gains[id];
        exposure_ev.push_back({{"image_index", id},
                               {"ev", context.exposure_ev[id]},
                               {"pixel_ev_shift", context.pixel_ev_shift[id]},
                               {"residual_ev", context.residual_ev[id]},
                               {"wb_gains", {wb[0], wb[1], wb[2]}}});
      }
    }
  }
  Log("Static masks: unmasked " + std::to_string(pixels) + " person pixels (" +
      std::to_string(changed) + " images), sky " +
      std::string(context.keep_sky ? "kept" : "cleared (indoor)") + ", cleared " +
      std::to_string(sky_pixels) + " sky pixels (" + std::to_string(sky_changed) +
      " images), edge refined " + std::to_string(sky_edge_refined) + " images (+" +
      std::to_string(sky_edge_to_sky) + " sky, +" + std::to_string(sky_edge_to_static) +
      " static), exposure " +
      std::string(context.exposure_calibrated ? "calibrated" : "uncalibrated") +
      " (med=" +
      std::to_string(context.exposure_report.value("residual_median_ev", 0.0)) +
      " EV, wb=" + std::string(context.wb_calibrated ? "calibrated" : "off") +
      "), photometric masked dark " + std::to_string(photo_dark_pixels) + " px (" +
      std::to_string(photo_dark_images) + " images), bright " +
      std::to_string(photo_bright_pixels) + " px (" + std::to_string(photo_bright_images) +
      " images, capped " + std::to_string(photo_capped_images) + "), clipped " +
      std::to_string(photo_clipped_pixels) + " px (" + std::to_string(photo_clipped_images) +
      " images, capped " + std::to_string(photo_clip_capped_images) + ") in " +
      std::to_string(context.photo_seconds) + " s");
  return {{"images", written.size()},
          {"images_changed", changed},
          {"components", rows.size()},
          {"unmasked_pixels", pixels},
          {"reasons", reasons},
          {"sky",
           {{"kept", context.keep_sky},
            {"images_with_raw_sky", raw_sky_images},
            {"images_with_open_sky", open_sky_images},
            {"surface_components", surface_sky_components},
            {"images_changed", sky_changed},
            {"cleared_pixels", sky_pixels},
            {"edge_refined_images", sky_edge_refined},
            {"edge_to_sky_pixels", sky_edge_to_sky},
            {"edge_to_static_pixels", sky_edge_to_static}}},
          {"photometric",
           {{"vignetting_coeffs",
             {{context.vig_coeffs[0][0], context.vig_coeffs[0][1], context.vig_coeffs[0][2]},
              {context.vig_coeffs[1][0], context.vig_coeffs[1][1], context.vig_coeffs[1][2]}}},
            {"dark_images", photo_dark_images},
            {"dark_pixels", photo_dark_pixels},
            {"bright_images", photo_bright_images},
            {"bright_pixels", photo_bright_pixels},
            {"clipped_images", photo_clipped_images},
            {"clipped_pixels", photo_clipped_pixels},
            {"capped_images", photo_capped_images},
            {"clip_capped_images", photo_clip_capped_images},
            {"exposure", context.exposure_report},
            {"white_balance", context.wb_report},
            {"seconds", context.photo_seconds}}},
          {"rules", StaticMaskRules()},
          {"seconds", Seconds(started)},
          {"exposure_ev", exposure_ev},
          {"component_rows", rows}};
}
}  // namespace geosun

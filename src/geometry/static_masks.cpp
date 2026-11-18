#include "geometry/static_masks.h"
#include "geometry/supervision.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <thread>

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
  bool keep_sky = true;
};
struct Labelled {
  cv::Mat mask, transient_map, labels, stats, centroids;
  int count = 0;
};
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
};
// Smooth local per-channel mean and variance over seed_mask != 0, blending to global seed
// statistics where local seed support falls below 64 pixels.
void LocalColorStats(const std::array<cv::Mat, 3>& ch, const cv::Mat& seed_mask, int box_size,
                     float min_var, std::array<cv::Mat, 3>& mean_out,
                     std::array<cv::Mat, 3>& var_out) {
  const int rows = seed_mask.rows, cols = seed_mask.cols;
  std::array<double, 3> g_sum{}, g_sum2{};
  size_t g_count = 0;
  for (int y = 0; y < rows; ++y) {
    const uint8_t* m = seed_mask.ptr<uint8_t>(y);
    const float* p0 = ch[0].ptr<float>(y);
    const float* p1 = ch[1].ptr<float>(y);
    const float* p2 = ch[2].ptr<float>(y);
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
  cv::Mat w, sw;
  seed_mask.convertTo(w, CV_32FC1, 1.0 / 255.0);
  cv::boxFilter(w, sw, CV_32F, {box_size, box_size}, {-1, -1}, true, cv::BORDER_REFLECT_101);
  const float area = float(box_size) * float(box_size);
  for (int c = 0; c < 3; ++c) {
    cv::Mat wc = w.mul(ch[c]), wc2 = wc.mul(ch[c]);
    cv::Mat sm, sm2;
    cv::boxFilter(wc, sm, CV_32F, {box_size, box_size}, {-1, -1}, true, cv::BORDER_REFLECT_101);
    cv::boxFilter(wc2, sm2, CV_32F, {box_size, box_size}, {-1, -1}, true, cv::BORDER_REFLECT_101);
    mean_out[c].create(rows, cols, CV_32FC1);
    var_out[c].create(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
      const float* swp = sw.ptr<float>(y);
      const float* smp = sm.ptr<float>(y);
      const float* sm2p = sm2.ptr<float>(y);
      float* mo = mean_out[c].ptr<float>(y);
      float* vo = var_out[c].ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        const float wv = swp[x];
        const float cnt = wv * area;
        if (cnt >= 64.f) {
          const float m = smp[x] / wv;
          mo[x] = m;
          vo[x] = std::max(min_var, sm2p[x] / wv - m * m);
        } else if (cnt > 1.f) {
          const float m_loc = smp[x] / wv;
          const float v_loc = std::max(min_var, sm2p[x] / wv - m_loc * m_loc);
          const float blend = cnt / 64.f;
          mo[x] = blend * m_loc + (1.f - blend) * g_mean[c];
          vo[x] = std::max(min_var, blend * v_loc + (1.f - blend) * g_var[c]);
        } else {
          mo[x] = g_mean[c];
          vo[x] = g_var[c];
        }
      }
    }
  }
}
// 3-channel color guided filter implemented directly with O(1) cv::boxFilter.
cv::Mat GuidedFilterColor(const std::array<cv::Mat, 3>& ch, const cv::Mat& p, int radius,
                          float eps) {
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
  const cv::Mat ma0 = box(a[0]), ma1 = box(a[1]), ma2 = box(a[2]), mb = box(b);
  cv::Mat q(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    const float *p0 = ma0.ptr<float>(y), *p1 = ma1.ptr<float>(y), *p2 = ma2.ptr<float>(y),
                *pb = mb.ptr<float>(y);
    const float *i0 = ch[0].ptr<float>(y), *i1 = ch[1].ptr<float>(y), *i2 = ch[2].ptr<float>(y);
    float* qo = q.ptr<float>(y);
    for (int x = 0; x < cols; ++x)
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
  cv::Mat sky_core;
  cv::erode(sky_roi, sky_core,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * r_in + 1, 2 * r_in + 1}),
            cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
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
  cv::Mat sky_outer, sky_fg_shell, sky_canopy;
  cv::dilate(sky_roi, sky_outer,
             cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * r_out + 1, 2 * r_out + 1}));
  cv::dilate(sky_roi, sky_fg_shell,
             cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * r_fg + 1, 2 * r_fg + 1}));
  cv::dilate(sky_roi, sky_canopy,
             cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * r_canopy + 1, 2 * r_canopy + 1}));
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
  const cv::Mat alpha_sky = GuidedFilterColor(ch, p0, r_gf, 1e-3f);
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
          {"sky_canopy_foliage_border_share", SkyCanopyFoliageBorderShare}};
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
  for (int c = 0; c < 2; ++c) {
    cv::Mat domain = model.cameras[c].Mask();
    context.full_domains[c] = domain.clone();
    const auto grid_cam = MapCamera(model.cameras[c], MapSize);
    cv::Mat grid_domain;
    cv::resize(domain, grid_domain, {grid_cam.width, grid_cam.height}, 0, 0, cv::INTER_NEAREST);
    context.full_domain_grid_pixels[c] = cv::countNonZero(grid_domain);
    cv::erode(domain, domain, cv::getStructuringElement(cv::MORPH_ELLIPSE, {31, 31}));
    context.domains[c] = domain;
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
  // Scene-level sky decision before Pass 3.
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
  // Pass 3: the new masks, on the same grid as pass 1.
  const auto map_camera = [&](uint32_t id) { return MapCamera(model.cameras[model.images[id].camera], MapSize); };
  std::vector<ApplyCounts> applied(model.images.size());
  ParallelFor(written.size(), workers, 1, [&](size_t i) {
    const auto grid = map_camera(written[i]);
    applied[written[i]] = Apply(context, written[i], {grid.width, grid.height});
  });
  Json rows = Json::array();
  std::map<std::string, size_t> reasons;
  uint64_t pixels = 0, sky_pixels = 0, sky_edge_to_sky = 0, sky_edge_to_static = 0;
  size_t changed = 0, sky_changed = 0, sky_edge_refined = 0;
  for (const auto id : written) {
    pixels += applied[id].person_unmasked;
    changed += applied[id].person_unmasked > 0;
    sky_pixels += applied[id].sky_cleared;
    sky_changed += applied[id].sky_cleared > 0;
    sky_edge_to_sky += applied[id].sky_edge_to_sky;
    sky_edge_to_static += applied[id].sky_edge_to_static;
    sky_edge_refined += (applied[id].sky_edge_to_sky > 0 || applied[id].sky_edge_to_static > 0);
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
  Log("Static masks: unmasked " + std::to_string(pixels) + " person pixels (" +
      std::to_string(changed) + " images), sky " +
      std::string(context.keep_sky ? "kept" : "cleared (indoor)") + ", cleared " +
      std::to_string(sky_pixels) + " sky pixels (" + std::to_string(sky_changed) +
      " images), edge refined " + std::to_string(sky_edge_refined) + " images (+" +
      std::to_string(sky_edge_to_sky) + " sky, +" + std::to_string(sky_edge_to_static) +
      " static)");
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
          {"rules", StaticMaskRules()},
          {"seconds", Seconds(started)},
          {"component_rows", rows}};
}
}  // namespace geosun

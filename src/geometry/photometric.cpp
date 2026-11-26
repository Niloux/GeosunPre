#include "geometry/photometric.h"
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>

namespace geosun {
namespace {
constexpr double kMaxFovRad = 80.0 * Pi / 180.0;
constexpr double kVigMinGain = 0.40;
constexpr double kVigMaxGain = 1.05;
constexpr double kMinSrgb = 0.12;
constexpr double kMaxSrgb = 0.76;
constexpr double kMaxSeedGrad = 0.10;
constexpr double kMaxTrackGrad = 0.12;
constexpr double kMinViewCos = 0.35;
constexpr double kLasVoxelSizeM = 0.20;
constexpr double kLasMaxProjectDistM = 15.0;
constexpr double kLasMaxViewAngleCos = 0.4226;  // cos(65 deg)
constexpr size_t kMaxLasSeedsPerBandPerImage = 100;
constexpr size_t kMaxLasSeedsTotal = 25000;
constexpr size_t kMaxLasTrackObs = 20;
constexpr size_t kMaxSfmTracks = 15000;
constexpr double kSfmMaxReprojPx = 2.0;
constexpr double kHoldoutBlockM = 1.0;
constexpr double kMaxProjRho = 0.99;
constexpr double kMinComponentShare = 0.90;
constexpr double kMaxHoldoutMedianEv = 0.18;
constexpr int kMinOuterObsOrder2 = 50;
constexpr int kMinOuterObsOrder3 = 200;
constexpr int kPass1Iters = 10;
constexpr int kPass2Iters = 12;
constexpr double kPass1HuberEv = 0.12;
constexpr double kPass2HuberEv = 0.08;
constexpr double kOutlierTrimEv = 0.40;
constexpr double kHoldoutOrderMarginEv = 0.002;
constexpr double kVigRegOrder2 = 1e-2;
constexpr double kVigRegOrder3 = 5e-2;
constexpr int kWbSmoothIters = 5;
constexpr double kWbSmoothMaxDtS = 2.5;
constexpr double kWbSmoothWeight = 2.0;
constexpr double kWbSmoothHuberEv = 0.06;
constexpr double kWbGlobalPrior = 1e-3;
constexpr double kWbMaxAbsEv = 0.80;
constexpr double kKdeBandwidthEv = 0.35;
constexpr double kSafeMaxBrightenEv = 0.30;
constexpr double kSafeBrightenTargetLin = 0.92;
constexpr double kSafeMaxDarkenEv = 0.80;
constexpr double kSafeMinP50Srgb = 0.30;
constexpr double kSafeMinP10Srgb = 0.06;
constexpr double kSafeSatLoShare = 0.005;
constexpr double kSafeSatHiShare = 0.020;
constexpr double kPostClipLinThresh = 0.985;
constexpr double kPostDarkSatShiftEv = 0.25;
constexpr int kPostClipMinCompArea = 8;

inline float SrgbToLin(float c) {
  const float x = std::clamp(c, 0.0f, 1.0f);
  return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

inline double MedianOf(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

inline double PercentileOf(std::vector<double> v, double frac) {
  if (v.empty())
    return 0.0;
  const size_t idx = std::min(v.size() - 1, size_t(std::floor(frac * double(v.size() - 1))));
  std::nth_element(v.begin(), v.begin() + idx, v.end());
  return v[idx];
}

inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline bool IsHoldoutBlock(const Eigen::Vector3d& world) {
  const int64_t ix = int64_t(std::floor(world.x() / kHoldoutBlockM));
  const int64_t iy = int64_t(std::floor(world.y() / kHoldoutBlockM));
  const int64_t iz = int64_t(std::floor(world.z() / kHoldoutBlockM));
  const uint64_t h = SplitMix64(uint64_t(ix) * 73856093ULL ^ uint64_t(iy) * 19349663ULL ^
                                uint64_t(iz) * 83492791ULL);
  return (h % 10ULL) == 0ULL;
}

inline cv::Vec3f SampleBilinearRgb(const cv::Mat& img, double u, double v) {
  const int x0 = int(std::floor(u)), y0 = int(std::floor(v));
  const float fx = float(u - x0), fy = float(v - y0);
  const cv::Vec3f* r0 = img.ptr<cv::Vec3f>(y0);
  const cv::Vec3f* r1 = img.ptr<cv::Vec3f>(y0 + 1);
  return r0[x0] * ((1.f - fx) * (1.f - fy)) + r0[x0 + 1] * (fx * (1.f - fy)) +
         r1[x0] * ((1.f - fx) * fy) + r1[x0 + 1] * (fx * fy);
}

bool ProjectWorldToGridPhoto(const Reconstruction& model, const ShutterMotion* motion,
                             const std::vector<Eigen::Matrix3d>& rotations,
                             const std::vector<Eigen::Vector3d>& translations,
                             const std::array<PhotometricCamGrid, 2>& grids, uint32_t j,
                             const Eigen::Vector3d& world, double& out_u, double& out_v,
                             double& out_dist, float& out_rho) {
  const uint32_t cid_j = model.images[j].camera;
  const auto& full_cam = model.cameras[cid_j];
  const auto& grid_j = grids[cid_j];
  const Eigen::Vector3d v = rotations[j] * world + translations[j];
  Eigen::Vector3d p = v;
  Eigen::Vector2d px;
  if (motion) {
    double s = 0.0;
    for (int it = 0; it < 2; ++it) {
      p = motion->Apply(j, cid_j, s, v);
      if (p.norm() <= 0.25)
        return false;
      px = full_cam.Project(p);
      s = motion->Fraction(full_cam, px);
    }
    p = motion->Apply(j, cid_j, s, v);
  }
  const double dist = p.norm();
  if (dist <= 0.25)
    return false;
  const double theta = std::acos(std::clamp(p.z() / dist, -1.0, 1.0));
  if (theta >= kMaxProjRho * kMaxFovRad)
    return false;
  px = full_cam.Project(p);
  out_u = px.x() * grid_j.scale_x - 0.5;
  out_v = px.y() * grid_j.scale_y - 0.5;
  if (out_u < 1.0 || out_u >= double(grid_j.width) - 2.0 || out_v < 1.0 ||
      out_v >= double(grid_j.height) - 2.0)
    return false;
  out_dist = dist;
  out_rho = float(theta / kMaxFovRad);
  return true;
}

struct TrackObs {
  uint32_t slot = 0;  // index into written (0..N-1)
  uint8_t cam = 0;    // 0 or 1
  float rho = 0.f;
  std::array<float, 3> log2_rgb{};
  bool active = true;
};

struct PhotometricTrack {
  bool holdout = false;
  bool from_las = true;
  std::vector<TrackObs> obs;
};

struct LasSeed {
  Eigen::Vector3d world = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
};

inline uint64_t VoxelKey(const Eigen::Vector3d& p) {
  const int64_t ix = int64_t(std::floor(p.x() / kLasVoxelSizeM)) + (1 << 20);
  const int64_t iy = int64_t(std::floor(p.y() / kLasVoxelSizeM)) + (1 << 20);
  const int64_t iz = int64_t(std::floor(p.z() / kLasVoxelSizeM)) + (1 << 20);
  return (uint64_t(ix & 0x1fffffLL) << 42) | (uint64_t(iy & 0x1fffffLL) << 21) |
         uint64_t(iz & 0x1fffffLL);
}

Eigen::Vector3d CamPointForPixel(const Camera& cam, const PhotometricCamGrid& grid,
                                 const ShutterMotion* motion, uint32_t img_id, uint32_t cid,
                                 int x, int y, double depth) {
  const size_t gidx = size_t(y) * size_t(grid.width) + size_t(x);
  const Eigen::Vector3d moved = grid.rays[gidx] * depth;
  if (!motion)
    return moved;
  const double s = motion->Fraction(cam, {(x + 0.5) / grid.scale_x, (y + 0.5) / grid.scale_y});
  return motion->Reference(img_id, cid, s, moved);
}

std::vector<PhotometricTrack> BuildPhotometricTracks(
    const Reconstruction& model, const ShutterMotion* motion, const std::vector<uint32_t>& written,
    const std::vector<int>& slot_of_image, const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& translations, const std::vector<Eigen::Vector3d>& centers,
    const std::array<PhotometricCamGrid, 2>& grids, const std::vector<PhotometricViewData>& views,
    int workers, size_t& out_las_tracks, size_t& out_sfm_tracks) {
  const float lin_min = SrgbToLin(float(kMinSrgb));
  const float lin_max = SrgbToLin(float(kMaxSrgb));
  const size_t n = written.size();

  // 1. Collect spatially-stratified LAS surface seeds across all registered images and radial bands.
  std::vector<std::vector<LasSeed>> per_image_seeds(n);
  ParallelFor(n, workers, 1, [&](size_t slot_i) {
    const uint32_t i = written[slot_i];
    const uint32_t cid = model.images[i].camera;
    const auto& cam = model.cameras[cid];
    const auto& grid = grids[cid];
    const auto& view = views[i];
    const Eigen::Matrix3d rt = rotations[i].transpose();

    std::array<std::vector<LasSeed>, 3> bands;
    for (int y = 1; y < grid.height - 1; ++y) {
      const uint8_t* mp = view.static_mask.ptr<uint8_t>(y);
      const uint8_t* sp = view.depth_smooth.ptr<uint8_t>(y);
      const float* gp = view.grad_srgb.ptr<float>(y);
      const cv::Vec3f* rp = view.rgb_lin.ptr<cv::Vec3f>(y);
      const float* dp = view.depth.ptr<float>(y);
      const float* d_prev = view.depth.ptr<float>(y - 1);
      const float* d_next = view.depth.ptr<float>(y + 1);
      for (int x = 1; x < grid.width - 1; ++x) {
        if (!mp[x] || !sp[x] || gp[x] >= float(kMaxSeedGrad))
          continue;
        const size_t gidx = size_t(y) * size_t(grid.width) + size_t(x);
        const float rho = grid.rho[gidx];
        if (rho <= 0.f || rho >= float(kMaxProjRho))
          continue;
        const cv::Vec3f c = rp[x];
        if (std::min({c[0], c[1], c[2]}) <= lin_min || std::max({c[0], c[1], c[2]}) >= lin_max)
          continue;
        const float d0 = dp[x], dl = dp[x - 1], dr = dp[x + 1], du = d_prev[x], dd = d_next[x];
        if (d0 <= 0.35f || dl <= 0.35f || dr <= 0.35f || du <= 0.35f || dd <= 0.35f)
          continue;
        const Eigen::Vector3d p_c = CamPointForPixel(cam, grid, motion, i, cid, x, y, double(d0));
        const Eigen::Vector3d p_l = CamPointForPixel(cam, grid, motion, i, cid, x - 1, y, double(dl));
        const Eigen::Vector3d p_r = CamPointForPixel(cam, grid, motion, i, cid, x + 1, y, double(dr));
        const Eigen::Vector3d p_u = CamPointForPixel(cam, grid, motion, i, cid, x, y - 1, double(du));
        const Eigen::Vector3d p_d = CamPointForPixel(cam, grid, motion, i, cid, x, y + 1, double(dd));
        Eigen::Vector3d n_cam = (p_r - p_l).cross(p_d - p_u);
        const double nn = n_cam.norm();
        if (nn <= 1e-8)
          continue;
        n_cam /= nn;
        if (n_cam.dot(-p_c.normalized()) < 0.0)
          n_cam = -n_cam;
        if (n_cam.dot(-p_c.normalized()) < kMinViewCos)
          continue;
        const Eigen::Vector3d world = rt * (p_c - translations[i]);
        const Eigen::Vector3d n_world = (rt * n_cam).normalized();
        const int band = rho < 0.5f ? 0 : (rho < 0.8f ? 1 : 2);
        bands[band].push_back({world, n_world});
      }
    }
    auto& out = per_image_seeds[slot_i];
    for (int b = 0; b < 3; ++b) {
      const auto& src = bands[b];
      if (src.empty())
        continue;
      const size_t step = std::max<size_t>(1, src.size() / kMaxLasSeedsPerBandPerImage);
      for (size_t k = 0; k < src.size() && out.size() < 3 * kMaxLasSeedsPerBandPerImage; k += step)
        out.push_back(src[k]);
    }
  });

  // Deduplicate LAS seeds in 0.20 m world voxels (keeping up to 2 distinct surface orientations).
  std::unordered_map<uint64_t, std::array<LasSeed, 2>> voxel_map;
  std::unordered_map<uint64_t, uint8_t> voxel_cnt;
  voxel_map.reserve(32768);
  std::vector<LasSeed> all_seeds;
  all_seeds.reserve(32768);
  for (const auto& vec : per_image_seeds) {
    for (const auto& s : vec) {
      const uint64_t key = VoxelKey(s.world);
      uint8_t& cnt = voxel_cnt[key];
      if (cnt == 0) {
        voxel_map[key][0] = s;
        cnt = 1;
        all_seeds.push_back(s);
      } else if (cnt == 1 && voxel_map[key][0].normal.dot(s.normal) < 0.7) {
        voxel_map[key][1] = s;
        cnt = 2;
        all_seeds.push_back(s);
      }
    }
  }
  if (all_seeds.size() > kMaxLasSeedsTotal) {
    std::vector<LasSeed> thinned;
    thinned.reserve(kMaxLasSeedsTotal);
    const double step = double(all_seeds.size()) / double(kMaxLasSeedsTotal);
    for (size_t k = 0; k < kMaxLasSeedsTotal; ++k)
      thinned.push_back(all_seeds[size_t(double(k) * step)]);
    all_seeds = std::move(thinned);
  }

  // Project LAS seeds into nearby covisible frames to form multi-view photometric tracks.
  std::vector<PhotometricTrack> las_tracks(all_seeds.size());
  ParallelFor(all_seeds.size(), workers, 64, [&](size_t si) {
    const auto& seed = all_seeds[si];
    struct CandFrame {
      double dist = 0.0;
      uint32_t slot = 0;
    };
    std::vector<CandFrame> cands;
    for (size_t slot_j = 0; slot_j < n; ++slot_j) {
      const uint32_t j = written[slot_j];
      const Eigen::Vector3d to_cam = centers[j] - seed.world;
      const double dist = to_cam.norm();
      if (dist <= 0.35 || dist >= kLasMaxProjectDistM)
        continue;
      if ((to_cam / dist).dot(seed.normal) < kLasMaxViewAngleCos)
        continue;
      cands.push_back({dist, uint32_t(slot_j)});
    }
    if (cands.size() < 2)
      return;
    std::sort(cands.begin(), cands.end(),
              [](const CandFrame& a, const CandFrame& b) { return a.dist < b.dist; });
    std::vector<TrackObs> valid_obs;
    valid_obs.reserve(std::min<size_t>(cands.size(), kMaxLasTrackObs * 2));
    for (const auto& cf : cands) {
      const uint32_t j = written[cf.slot];
      const uint8_t cam_j = uint8_t(model.images[j].camera);
      const auto& view_j = views[j];
      double u = 0.0, v = 0.0, d_proj = 0.0;
      float rho_j = 0.f;
      if (!ProjectWorldToGridPhoto(model, motion, rotations, translations, grids, j, seed.world,
                                   u, v, d_proj, rho_j))
        continue;
      if (rho_j >= float(kMaxProjRho))
        continue;
      const int ui = int(std::lround(u)), vi = int(std::lround(v));
      if (!view_j.static_mask.at<uint8_t>(vi, ui))
        continue;
      const float d_j = view_j.depth.at<float>(vi, ui);
      if (d_j <= 0.30f || std::abs(d_proj - double(d_j)) > std::max(0.08, 0.05 * d_proj))
        continue;
      if (view_j.grad_srgb.at<float>(vi, ui) >= float(kMaxTrackGrad))
        continue;
      const cv::Vec3f c_lin = SampleBilinearRgb(view_j.rgb_lin, u, v);
      if (std::min({c_lin[0], c_lin[1], c_lin[2]}) <= lin_min ||
          std::max({c_lin[0], c_lin[1], c_lin[2]}) >= lin_max)
        continue;
      valid_obs.push_back(
          {cf.slot, cam_j, rho_j,
           {std::log2(c_lin[0]), std::log2(c_lin[1]), std::log2(c_lin[2])}, true});
    }
    if (valid_obs.size() < 2)
      return;
    // Keep up to kMaxLasTrackObs balanced across both lenses and radial bands.
    PhotometricTrack tr;
    tr.holdout = IsHoldoutBlock(seed.world);
    tr.from_las = true;
    if (valid_obs.size() <= kMaxLasTrackObs) {
      tr.obs = std::move(valid_obs);
    } else {
      std::array<std::vector<size_t>, 2> by_cam;
      for (size_t k = 0; k < valid_obs.size(); ++k)
        by_cam[valid_obs[k].cam].push_back(k);
      std::array<size_t, 2> quota{kMaxLasTrackObs / 2, kMaxLasTrackObs - kMaxLasTrackObs / 2};
      for (int c = 0; c < 2; ++c) {
        if (by_cam[c].size() < quota[c]) {
          const size_t deficit = quota[c] - by_cam[c].size();
          quota[c] = by_cam[c].size();
          quota[1 - c] = std::min(by_cam[1 - c].size(), quota[1 - c] + deficit);
        }
      }
      for (int c = 0; c < 2; ++c) {
        const auto& idx_list = by_cam[c];
        if (quota[c] == 0)
          continue;
        const double step = double(idx_list.size()) / double(quota[c]);
        for (size_t k = 0; k < quota[c]; ++k)
          tr.obs.push_back(valid_obs[idx_list[size_t(double(k) * step)]]);
      }
    }
    if (tr.obs.size() >= 2)
      las_tracks[si] = std::move(tr);
  });

  std::vector<PhotometricTrack> tracks;
  tracks.reserve(all_seeds.size() + kMaxSfmTracks);
  out_las_tracks = 0;
  for (auto& tr : las_tracks) {
    if (tr.obs.size() >= 2) {
      tracks.push_back(std::move(tr));
      ++out_las_tracks;
    }
  }

  // 2. Supplement with clean static SfM 3D point tracks (strong cross-lens & corridor/doorway ties).
  std::vector<const ModelPoint*> sfm_cands;
  sfm_cands.reserve(model.points.size());
  for (const auto& pt : model.points) {
    if (pt.track.size() >= 3 && std::isfinite(pt.error) && pt.error < kSfmMaxReprojPx)
      sfm_cands.push_back(&pt);
  }
  out_sfm_tracks = 0;
  if (!sfm_cands.empty()) {
    const size_t step = std::max<size_t>(1, sfm_cands.size() / kMaxSfmTracks);
    for (size_t pi = 0; pi < sfm_cands.size() && out_sfm_tracks < kMaxSfmTracks; pi += step) {
      const auto& pt = *sfm_cands[pi];
      PhotometricTrack tr;
      tr.holdout = IsHoldoutBlock(pt.xyz);
      tr.from_las = false;
      for (const auto& o : pt.track) {
        const int slot = slot_of_image[o.image];
        if (slot < 0)
          continue;
        const auto& img = model.images[o.image];
        if (o.feature >= img.pixels.size())
          continue;
        const uint8_t cid = uint8_t(img.camera);
        const auto& grid = grids[cid];
        const auto& view = views[o.image];
        const Eigen::Vector2d& px = img.pixels[o.feature];
        const double u = px.x() * grid.scale_x - 0.5;
        const double v = px.y() * grid.scale_y - 0.5;
        if (u < 1.0 || u >= double(grid.width) - 2.0 || v < 1.0 ||
            v >= double(grid.height) - 2.0)
          continue;
        const int ui = int(std::lround(u)), vi = int(std::lround(v));
        if (!view.static_mask.at<uint8_t>(vi, ui) ||
            view.grad_srgb.at<float>(vi, ui) >= float(kMaxTrackGrad))
          continue;
        const float rho = grid.rho[size_t(vi) * size_t(grid.width) + size_t(ui)];
        if (rho <= 0.f || rho >= float(kMaxProjRho))
          continue;
        const cv::Vec3f c_lin = SampleBilinearRgb(view.rgb_lin, u, v);
        if (std::min({c_lin[0], c_lin[1], c_lin[2]}) <= lin_min ||
            std::max({c_lin[0], c_lin[1], c_lin[2]}) >= lin_max)
          continue;
        tr.obs.push_back(
            {uint32_t(slot), cid, rho,
             {std::log2(c_lin[0]), std::log2(c_lin[1]), std::log2(c_lin[2])}, true});
      }
      if (tr.obs.size() >= 2) {
        tracks.push_back(std::move(tr));
        ++out_sfm_tracks;
      }
    }
  }
  return tracks;
}

struct ChannelSolution {
  std::vector<double> e;
  std::array<std::array<double, 3>, 2> vig{};
};

// Solves log2(I_ch) = L_t + e_i + vig_c(rho) via exact per-track Schur complement Huber IRLS.
bool SolveChannelSchur(int ch, size_t n_imgs, const std::vector<PhotometricTrack>& tracks,
                       const std::vector<int>& comp_anchors, const std::vector<int>& largest_comp,
                       const std::array<int, 2>& vig_order, int iters, double huber_delta,
                       ChannelSolution& sol) {
  const std::array<int, 2> v_base{int(n_imgs), int(n_imgs) + vig_order[0]};
  const int p_dim = int(n_imgs) + vig_order[0] + vig_order[1];
  if (sol.e.size() != n_imgs)
    sol.e.assign(n_imgs, 0.0);

  const bool use_dense = (p_dim <= 1200);
  Eigen::MatrixXd H_dense;
  Eigen::VectorXd rhs_dense;
  std::vector<Eigen::Triplet<double>> triplets;
  std::vector<std::vector<double>> obs_weights(tracks.size());
  for (size_t ti = 0; ti < tracks.size(); ++ti)
    obs_weights[ti].assign(tracks[ti].obs.size(), 1.0);

  auto vig_eval = [&](uint8_t cam, float rho) -> double {
    const int ord = vig_order[cam];
    if (ord == 0)
      return 0.0;
    const double r2 = double(rho) * double(rho);
    double v = sol.vig[cam][0] * r2 + sol.vig[cam][1] * r2 * r2;
    if (ord >= 3)
      v += sol.vig[cam][2] * r2 * r2 * r2;
    return v;
  };

  auto init_weights_from_state = [&]() {
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      const auto& tr = tracks[ti];
      if (tr.holdout)
        continue;
      int active_cnt = 0;
      for (const auto& o : tr.obs)
        if (o.active)
          ++active_cnt;
      if (active_cnt < 2)
        continue;
      double sum_wy = 0.0, sum_w = 0.0;
      for (size_t oi = 0; oi < tr.obs.size(); ++oi) {
        const auto& o = tr.obs[oi];
        if (!o.active)
          continue;
        const double w = obs_weights[ti][oi];
        sum_w += w;
        sum_wy += w * (double(o.log2_rgb[ch]) - sol.e[o.slot] - vig_eval(o.cam, o.rho));
      }
      if (sum_w <= 1e-12)
        continue;
      const double L_t = sum_wy / sum_w;
      for (size_t oi = 0; oi < tr.obs.size(); ++oi) {
        const auto& o = tr.obs[oi];
        if (!o.active)
          continue;
        const double res =
            std::abs(double(o.log2_rgb[ch]) - sol.e[o.slot] - vig_eval(o.cam, o.rho) - L_t);
        obs_weights[ti][oi] = res < huber_delta ? 1.0 : huber_delta / std::max(res, 1e-6);
      }
    }
  };
  // If we already have a warm start (e.g., Pass 2), initialize Huber weights from it.
  bool warm = false;
  for (double ev : sol.e)
    if (ev != 0.0) {
      warm = true;
      break;
    }
  if (warm)
    init_weights_from_state();

  struct Entry {
    int col = 0;
    double val = 0.0;
  };
  std::vector<Entry> g_vec;
  g_vec.reserve(64);
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(p_dim);

  for (int iter = 0; iter < iters; ++iter) {
    if (use_dense) {
      H_dense.setZero(p_dim, p_dim);
      rhs_dense.setZero(p_dim);
    } else {
      triplets.clear();
      triplets.reserve(tracks.size() * 64);
      rhs_dense.setZero(p_dim);
    }

    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      const auto& tr = tracks[ti];
      if (tr.holdout)
        continue;
      double W_t = 0.0, Y_t = 0.0;
      int active_cnt = 0;
      for (size_t oi = 0; oi < tr.obs.size(); ++oi) {
        const auto& o = tr.obs[oi];
        if (!o.active)
          continue;
        ++active_cnt;
        const double w = obs_weights[ti][oi];
        W_t += w;
        Y_t += w * double(o.log2_rgb[ch]);
      }
      if (active_cnt < 2 || W_t <= 1e-12)
        continue;

      g_vec.clear();
      auto add_g = [&](int col, double v) {
        for (auto& e : g_vec) {
          if (e.col == col) {
            e.val += v;
            return;
          }
        }
        g_vec.push_back({col, v});
      };

      for (size_t oi = 0; oi < tr.obs.size(); ++oi) {
        const auto& o = tr.obs[oi];
        if (!o.active)
          continue;
        const double w = obs_weights[ti][oi];
        const double y = double(o.log2_rgb[ch]);
        const int ord = vig_order[o.cam];
        const double r2 = double(o.rho) * double(o.rho);
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        int cols[4] = {int(o.slot), 0, 0, 0};
        double vals[4] = {1.0, r2, r4, r6};
        const int k_dim = 1 + ord;
        for (int a = 0; a < ord; ++a)
          cols[1 + a] = v_base[o.cam] + a;

        for (int a = 0; a < k_dim; ++a) {
          add_g(cols[a], w * vals[a]);
          rhs_dense[cols[a]] += w * vals[a] * y;
          for (int b = 0; b < k_dim; ++b) {
            const double vv = w * vals[a] * vals[b];
            if (use_dense)
              H_dense(cols[a], cols[b]) += vv;
            else
              triplets.emplace_back(cols[a], cols[b], vv);
          }
        }
      }

      const double inv_W = 1.0 / W_t;
      const double y_over_w = Y_t * inv_W;
      for (const auto& ga : g_vec) {
        rhs_dense[ga.col] -= ga.val * y_over_w;
        for (const auto& gb : g_vec) {
          const double vv = -inv_W * ga.val * gb.val;
          if (use_dense)
            H_dense(ga.col, gb.col) += vv;
          else
            triplets.emplace_back(ga.col, gb.col, vv);
        }
      }
    }

    auto add_diag = [&](int col, double v) {
      if (use_dense)
        H_dense(col, col) += v;
      else
        triplets.emplace_back(col, col, v);
    };
    for (const int anchor : comp_anchors)
      add_diag(anchor, 1.0);
    for (int d = 0; d < p_dim; ++d)
      add_diag(d, 1e-6);
    for (int c = 0; c < 2; ++c) {
      if (vig_order[c] >= 2)
        add_diag(v_base[c] + 1, kVigRegOrder2);
      if (vig_order[c] >= 3)
        add_diag(v_base[c] + 2, kVigRegOrder3);
    }

    if (use_dense) {
      Eigen::LDLT<Eigen::MatrixXd> ldlt(H_dense);
      if (ldlt.info() != Eigen::Success)
        return false;
      theta = ldlt.solve(rhs_dense);
      if (!theta.allFinite())
        return false;
    } else {
      Eigen::SparseMatrix<double> mat(p_dim, p_dim);
      mat.setFromTriplets(triplets.begin(), triplets.end());
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(mat);
      if (solver.info() != Eigen::Success)
        return false;
      theta = solver.solve(rhs_dense);
      if (solver.info() != Eigen::Success || !theta.allFinite())
        return false;
    }

    std::vector<double> lc_vals;
    lc_vals.reserve(largest_comp.size());
    for (const int s : largest_comp)
      lc_vals.push_back(theta[s]);
    const double med = MedianOf(std::move(lc_vals));
    for (size_t s = 0; s < n_imgs; ++s)
      sol.e[s] = std::clamp(theta[int(s)] - med, -12.0, 12.0);
    for (int c = 0; c < 2; ++c) {
      sol.vig[c] = {0.0, 0.0, 0.0};
      for (int a = 0; a < vig_order[c]; ++a)
        sol.vig[c][size_t(a)] = theta[v_base[c] + a];
    }

    if (iter + 1 < iters)
      init_weights_from_state();
  }
  return true;
}

bool VignettingInBounds(const std::array<double, 3>& coeffs, int order) {
  if (order == 0)
    return true;
  for (int i = 0; i <= 40; ++i) {
    const double rho = double(i) / 40.0;
    const double r2 = rho * rho;
    double lg = coeffs[0] * r2 + coeffs[1] * r2 * r2;
    if (order >= 3)
      lg += coeffs[2] * r2 * r2 * r2;
    const double v = std::exp2(lg);
    if (!std::isfinite(v) || v < kVigMinGain || v > kVigMaxGain)
      return false;
  }
  return true;
}

struct HoldoutMetrics {
  size_t lum_samples = 0;
  double lum_before_p50 = 0.0, lum_before_p90 = 0.0;
  double lum_after_p50 = 0.0, lum_after_p90 = 0.0;
  double edge_after_p50 = 0.0;
  double cross_rg_after_p50 = 0.0, cross_bg_after_p50 = 0.0;
};

HoldoutMetrics EvaluateHoldout(const std::vector<PhotometricTrack>& tracks,
                               const std::array<ChannelSolution, 3>& ch_sol,
                               const std::vector<double>& ev_final,
                               const std::vector<std::array<double, 3>>& wb_gains,
                               const std::array<int, 2>& vig_order) {
  std::vector<double> lum_b, lum_a, edge_a, cross_rg, cross_bg;
  auto vig_log2 = [&](uint8_t cam, int ch, float rho) {
    const int ord = vig_order[cam];
    if (ord == 0)
      return 0.0;
    const double r2 = double(rho) * double(rho);
    double v = ch_sol[ch].vig[cam][0] * r2 + ch_sol[ch].vig[cam][1] * r2 * r2;
    if (ord >= 3)
      v += ch_sol[ch].vig[cam][2] * r2 * r2 * r2;
    return v;
  };
  for (const auto& tr : tracks) {
    if (!tr.holdout || tr.obs.size() < 2)
      continue;
    std::vector<double> raw_y, cor_y;
    std::array<std::vector<double>, 2> rg_cam, bg_cam;
    std::vector<float> rhos;
    for (const auto& o : tr.obs) {
      const double lr = std::exp2(double(o.log2_rgb[0]));
      const double lg = std::exp2(double(o.log2_rgb[1]));
      const double lb = std::exp2(double(o.log2_rgb[2]));
      const double raw_lum = std::log2(std::max(0.2126 * lr + 0.7152 * lg + 0.0722 * lb, 1e-6));
      raw_y.push_back(raw_lum);

      const double cr = lr / EvaluateVignettingGain(ch_sol[0].vig[o.cam], vig_order[o.cam], o.rho) *
                        wb_gains[o.slot][0];
      const double cg = lg / EvaluateVignettingGain(ch_sol[1].vig[o.cam], vig_order[o.cam], o.rho) *
                        wb_gains[o.slot][1];
      const double cb = lb / EvaluateVignettingGain(ch_sol[2].vig[o.cam], vig_order[o.cam], o.rho) *
                        wb_gains[o.slot][2];
      const double cor_lum =
          std::log2(std::max(0.2126 * cr + 0.7152 * cg + 0.0722 * cb, 1e-6)) - ev_final[o.slot];
      cor_y.push_back(cor_lum);
      rhos.push_back(o.rho);

      const double log_rg =
          (double(o.log2_rgb[0]) - vig_log2(o.cam, 0, o.rho)) -
          (double(o.log2_rgb[1]) - vig_log2(o.cam, 1, o.rho)) +
          std::log2(std::max(wb_gains[o.slot][0] / wb_gains[o.slot][1], 1e-6));
      const double log_bg =
          (double(o.log2_rgb[2]) - vig_log2(o.cam, 2, o.rho)) -
          (double(o.log2_rgb[1]) - vig_log2(o.cam, 1, o.rho)) +
          std::log2(std::max(wb_gains[o.slot][2] / wb_gains[o.slot][1], 1e-6));
      rg_cam[o.cam].push_back(log_rg);
      bg_cam[o.cam].push_back(log_bg);
    }
    const double med_b = MedianOf(raw_y);
    const double med_a = MedianOf(cor_y);
    for (size_t k = 0; k < cor_y.size(); ++k) {
      lum_b.push_back(std::abs(raw_y[k] - med_b));
      const double err = std::abs(cor_y[k] - med_a);
      lum_a.push_back(err);
      if (rhos[k] >= 0.75f)
        edge_a.push_back(err);
    }
    if (!rg_cam[0].empty() && !rg_cam[1].empty()) {
      cross_rg.push_back(std::abs(MedianOf(rg_cam[0]) - MedianOf(rg_cam[1])));
      cross_bg.push_back(std::abs(MedianOf(bg_cam[0]) - MedianOf(bg_cam[1])));
    }
  }
  HoldoutMetrics m;
  m.lum_samples = lum_a.size();
  m.lum_before_p50 = PercentileOf(lum_b, 0.50);
  m.lum_before_p90 = PercentileOf(lum_b, 0.90);
  m.lum_after_p50 = PercentileOf(lum_a, 0.50);
  m.lum_after_p90 = PercentileOf(lum_a, 0.90);
  m.edge_after_p50 = PercentileOf(edge_a, 0.50);
  m.cross_rg_after_p50 = PercentileOf(cross_rg, 0.50);
  m.cross_bg_after_p50 = PercentileOf(cross_bg, 0.50);
  return m;
}

// Per-lens 1D temporal Huber smoothing of per-frame log2(R/G) and log2(B/G) chromaticity offsets.
std::vector<double> SmoothWbChannel(const Reconstruction& model, const std::vector<uint32_t>& written,
                                    const std::vector<double>& seconds,
                                    const std::vector<double>& raw_offset,
                                    const std::vector<int>& obs_counts) {
  const size_t n = written.size();
  std::vector<double> out = raw_offset;
  for (uint32_t cid = 0; cid < 2; ++cid) {
    std::vector<size_t> seq;
    seq.reserve(n);
    for (size_t s = 0; s < n; ++s)
      if (model.images[written[s]].camera == cid)
        seq.push_back(s);
    if (seq.empty())
      continue;
    std::sort(seq.begin(), seq.end(),
              [&](size_t a, size_t b) { return seconds[written[a]] < seconds[written[b]]; });
    const int k = int(seq.size());
    std::vector<double> hw(size_t(std::max(0, k - 1)), 1.0);
    std::vector<double> cur(k, 0.0);
    for (int i = 0; i < k; ++i)
      cur[i] = std::clamp(raw_offset[seq[i]], -kWbMaxAbsEv, kWbMaxAbsEv);

    for (int iter = 0; iter < kWbSmoothIters; ++iter) {
      Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(k, k);
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k);
      for (int i = 0; i < k; ++i) {
        const size_t s = seq[i];
        const double dw = std::min(double(obs_counts[s]), 100.0);
        mat(i, i) += dw + kWbGlobalPrior;
        rhs[i] += dw * raw_offset[s];
        if (i + 1 < k) {
          const double dt = std::abs(seconds[written[seq[i + 1]]] - seconds[written[s]]);
          if (dt <= kWbSmoothMaxDtS) {
            const double sw = kWbSmoothWeight * hw[size_t(i)];
            mat(i, i) += sw;
            mat(i + 1, i + 1) += sw;
            mat(i, i + 1) -= sw;
            mat(i + 1, i) -= sw;
          }
        }
      }
      const Eigen::VectorXd sol = mat.ldlt().solve(rhs);
      if (!sol.allFinite())
        break;
      for (int i = 0; i < k; ++i)
        cur[i] = std::clamp(sol[i], -kWbMaxAbsEv, kWbMaxAbsEv);
      for (int i = 0; i + 1 < k; ++i) {
        const double diff = std::abs(cur[i] - cur[i + 1]);
        hw[size_t(i)] =
            diff < kWbSmoothHuberEv ? 1.0 : kWbSmoothHuberEv / std::max(diff, 1e-6);
      }
    }
    for (int i = 0; i < k; ++i)
      out[seq[i]] = cur[i];
  }
  return out;
}

double ComputeKdeMode(const std::vector<double>& vals) {
  if (vals.empty())
    return 0.0;
  auto [it_min, it_max] = std::minmax_element(vals.begin(), vals.end());
  const double vmin = *it_min, vmax = *it_max;
  if (vmax - vmin <= 1e-6 || vals.size() < 2)
    return vmin;
  constexpr int kBins = 400;
  const double inv_2bw2 = 0.5 / (kKdeBandwidthEv * kKdeBandwidthEv);
  double best_score = -1.0, best_val = vmin;
  for (int b = 0; b < kBins; ++b) {
    const double g = vmin + (vmax - vmin) * (double(b) / double(kBins - 1));
    double score = 0.0;
    for (const double v : vals) {
      const double d = g - v;
      score += std::exp(-d * d * inv_2bw2);
    }
    if (score > best_score) {
      best_score = score;
      best_val = g;
    }
  }
  return best_val;
}
}  // namespace

double EvaluateVignettingGain(const std::array<double, 3>& coeffs, int order, double rho) {
  if (order <= 0)
    return 1.0;
  const double r2 = rho * rho;
  double lg = coeffs[0] * r2 + coeffs[1] * r2 * r2;
  if (order >= 3)
    lg += coeffs[2] * r2 * r2 * r2;
  return std::clamp(std::exp2(lg), kVigMinGain, kVigMaxGain);
}

PhotometricCalibration SolvePhotometricCalibration(
    const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
    const std::vector<uint32_t>& written, const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& translations, const std::vector<Eigen::Vector3d>& centers,
    const std::vector<double>& seconds, const std::array<PhotometricCamGrid, 2>& grids,
    const std::vector<PhotometricViewData>& views, int workers) {
  (void)scene;
  PhotometricCalibration cal;
  const size_t n_total = model.images.size();
  const size_t n = written.size();
  cal.exposure_ev.assign(n_total, 0.0);
  cal.pixel_ev_shift.assign(n_total, 0.0);
  cal.residual_ev.assign(n_total, 0.0);
  cal.exposure_valid.assign(n_total, 0);
  cal.wb_gains.assign(n_total, {1.0, 1.0, 1.0});
  cal.report = {{"calibrated", false},
                {"vignetting_order", {0, 0}},
                {"vignetting_coeffs_rgb", Json::array()},
                {"vignetting_edge_gain", Json::array()},
                {"exposure", {{"calibrated", false}}},
                {"white_balance", {{"calibrated", false}}},
                {"holdout", Json::object()},
                {"connectivity", Json::object()}};
  if (n < 2)
    return cal;

  std::vector<int> slot_of_image(n_total, -1);
  for (size_t s = 0; s < n; ++s)
    slot_of_image[written[s]] = int(s);

  size_t las_track_count = 0, sfm_track_count = 0;
  std::vector<PhotometricTrack> tracks =
      BuildPhotometricTracks(model, motion, written, slot_of_image, rotations, translations,
                             centers, grids, views, workers, las_track_count, sfm_track_count);

  // Count train/holdout, per-camera radial coverage, and build connectivity graph on train tracks.
  std::vector<std::vector<size_t>> adj(n);
  std::vector<int> train_obs_per_img(n, 0);
  std::array<std::array<int, 3>, 2> radial_counts{};
  size_t train_track_count = 0, holdout_track_count = 0;
  for (const auto& tr : tracks) {
    if (tr.holdout) {
      ++holdout_track_count;
      continue;
    }
    ++train_track_count;
    for (size_t a = 0; a < tr.obs.size(); ++a) {
      const auto& oa = tr.obs[a];
      ++train_obs_per_img[oa.slot];
      const int band = oa.rho < 0.5f ? 0 : (oa.rho < 0.8f ? 1 : 2);
      ++radial_counts[oa.cam][band];
      for (size_t b = a + 1; b < tr.obs.size(); ++b) {
        const auto& ob = tr.obs[b];
        if (oa.slot != ob.slot) {
          adj[oa.slot].push_back(ob.slot);
          adj[ob.slot].push_back(oa.slot);
        }
      }
    }
  }

  std::vector<int> comp_id(n, -1);
  std::vector<std::vector<int>> components;
  std::vector<int> queue;
  for (size_t s = 0; s < n; ++s) {
    if (comp_id[s] >= 0)
      continue;
    const int cid = int(components.size());
    comp_id[s] = cid;
    queue.clear();
    queue.push_back(int(s));
    for (size_t head = 0; head < queue.size(); ++head) {
      const int u = queue[head];
      for (const size_t v : adj[size_t(u)]) {
        if (comp_id[v] < 0) {
          comp_id[v] = cid;
          queue.push_back(int(v));
        }
      }
    }
    components.push_back(queue);
  }
  size_t largest_idx = 0;
  for (size_t c = 1; c < components.size(); ++c)
    if (components[c].size() > components[largest_idx].size())
      largest_idx = c;
  std::vector<int> comp_anchors;
  comp_anchors.reserve(components.size());
  for (const auto& comp : components) {
    int best = comp[0];
    for (const int s : comp)
      if (train_obs_per_img[size_t(s)] > train_obs_per_img[size_t(best)])
        best = s;
    comp_anchors.push_back(best);
  }

  cal.report["connectivity"] = {{"total_images", n},
                                {"components", components.size()},
                                {"largest_component_images", components[largest_idx].size()},
                                {"las_tracks", las_track_count},
                                {"sfm_tracks", sfm_track_count},
                                {"train_tracks", train_track_count},
                                {"holdout_tracks", holdout_track_count},
                                {"radial_counts",
                                 {{radial_counts[0][0], radial_counts[0][1], radial_counts[0][2]},
                                  {radial_counts[1][0], radial_counts[1][1], radial_counts[1][2]}}}};

  const size_t min_comp_images = size_t(std::ceil(kMinComponentShare * double(n)));
  if (train_track_count < 16 || components[largest_idx].size() < min_comp_images) {
    Log("Photometric calibration skipped: insufficient connected tracks (" +
        std::to_string(components[largest_idx].size()) + "/" + std::to_string(n) + " images)");
    return cal;
  }

  std::array<int, 2> vig_order{3, 3};
  for (int c = 0; c < 2; ++c) {
    if (radial_counts[c][2] < kMinOuterObsOrder2)
      vig_order[c] = 0;
    else if (radial_counts[c][2] < kMinOuterObsOrder3)
      vig_order[c] = 2;
    else
      vig_order[c] = 3;
  }

  std::array<ChannelSolution, 3> ch_sol;
  for (int ch = 0; ch < 3; ++ch) {
    if (!SolveChannelSchur(ch, n, tracks, comp_anchors, components[largest_idx], vig_order,
                           kPass1Iters, kPass1HuberEv, ch_sol[ch])) {
      Log("Photometric calibration Pass 1 failed on channel " + std::to_string(ch));
      return cal;
    }
  }

  // Pass 1 outlier trimming across all 3 channels jointly.
  size_t trimmed_obs = 0;
  for (auto& tr : tracks) {
    if (tr.holdout)
      continue;
    for (int ch = 0; ch < 3; ++ch) {
      double sum_y = 0.0;
      int cnt = 0;
      for (const auto& o : tr.obs) {
        if (!o.active)
          continue;
        double v_lg = 0.0;
        if (vig_order[o.cam] > 0) {
          const double r2 = double(o.rho) * double(o.rho);
          v_lg = ch_sol[ch].vig[o.cam][0] * r2 + ch_sol[ch].vig[o.cam][1] * r2 * r2 +
                 (vig_order[o.cam] >= 3 ? ch_sol[ch].vig[o.cam][2] * r2 * r2 * r2 : 0.0);
        }
        sum_y += double(o.log2_rgb[ch]) - ch_sol[ch].e[o.slot] - v_lg;
        ++cnt;
      }
      if (cnt < 2)
        continue;
      const double L_t = sum_y / double(cnt);
      for (auto& o : tr.obs) {
        if (!o.active)
          continue;
        double v_lg = 0.0;
        if (vig_order[o.cam] > 0) {
          const double r2 = double(o.rho) * double(o.rho);
          v_lg = ch_sol[ch].vig[o.cam][0] * r2 + ch_sol[ch].vig[o.cam][1] * r2 * r2 +
                 (vig_order[o.cam] >= 3 ? ch_sol[ch].vig[o.cam][2] * r2 * r2 * r2 : 0.0);
        }
        const double res = std::abs(double(o.log2_rgb[ch]) - ch_sol[ch].e[o.slot] - v_lg - L_t);
        if (res > kOutlierTrimEv) {
          o.active = false;
          ++trimmed_obs;
        }
      }
    }
  }

  auto run_pass2 = [&](const std::array<int, 2>& orders, std::array<ChannelSolution, 3>& out_sol) {
    for (int ch = 0; ch < 3; ++ch) {
      if (!SolveChannelSchur(ch, n, tracks, comp_anchors, components[largest_idx], orders,
                             kPass2Iters, kPass2HuberEv, out_sol[ch]))
        return false;
    }
    return true;
  };

  if (!run_pass2(vig_order, ch_sol)) {
    Log("Photometric calibration Pass 2 failed");
    return cal;
  }

  auto build_ev_wb = [&](const std::array<ChannelSolution, 3>& sol,
                         std::vector<double>& out_ev_slot,
                         std::vector<std::array<double, 3>>& out_wb_total,
                         std::vector<double>& out_rg_slot,
                         std::vector<double>& out_bg_slot) {
    out_ev_slot.assign(n, 0.0);
    std::vector<double> raw_rg(n, 0.0), raw_bg(n, 0.0);
    for (size_t s = 0; s < n; ++s) {
      out_ev_slot[s] =
          0.2126 * sol[0].e[s] + 0.7152 * sol[1].e[s] + 0.0722 * sol[2].e[s];
      raw_rg[s] = sol[0].e[s] - sol[1].e[s];
      raw_bg[s] = sol[2].e[s] - sol[1].e[s];
    }
    const double med_ev = MedianOf(out_ev_slot);
    const double med_rg = MedianOf(raw_rg);
    const double med_bg = MedianOf(raw_bg);
    for (size_t s = 0; s < n; ++s) {
      out_ev_slot[s] -= med_ev;
      raw_rg[s] -= med_rg;
      raw_bg[s] -= med_bg;
    }
    out_rg_slot = SmoothWbChannel(model, written, seconds, raw_rg, train_obs_per_img);
    out_bg_slot = SmoothWbChannel(model, written, seconds, raw_bg, train_obs_per_img);
    out_wb_total.assign(n_total, {1.0, 1.0, 1.0});
    for (size_t s = 0; s < n; ++s) {
      const double gr = std::exp2(-out_rg_slot[s]);
      const double gg = 1.0;
      const double gb = std::exp2(-out_bg_slot[s]);
      const double norm = 0.2126 * gr + 0.7152 * gg + 0.0722 * gb;
      out_wb_total[written[s]] = {gr / norm, gg / norm, gb / norm};
    }
  };

  std::vector<double> ev_slot, rg_slot, bg_slot;
  std::vector<std::array<double, 3>> wb_total;
  build_ev_wb(ch_sol, ev_slot, wb_total, rg_slot, bg_slot);
  std::vector<double> ev_total(n_total, 0.0);
  for (size_t s = 0; s < n; ++s)
    ev_total[written[s]] = ev_slot[s];
  HoldoutMetrics holdout = EvaluateHoldout(tracks, ch_sol, ev_total, wb_total, vig_order);

  // Check physical vignetting bounds and holdout-guided downgrade from order 3 -> 2 if needed.
  std::array<bool, 2> bad_vig{false, false};
  for (int c = 0; c < 2; ++c) {
    for (int ch = 0; ch < 3; ++ch) {
      if (!VignettingInBounds(ch_sol[ch].vig[c], vig_order[c]))
        bad_vig[c] = true;
    }
  }
  if (vig_order[0] == 3 || vig_order[1] == 3) {
    std::array<int, 2> test_order = vig_order;
    bool try_downgrade = false;
    for (int c = 0; c < 2; ++c) {
      if (vig_order[c] == 3 && bad_vig[c]) {
        test_order[c] = 2;
        try_downgrade = true;
      }
    }
    if (!try_downgrade) {
      // Also test order-2 globally on holdout to ensure 6th-order terms actually help or don't hurt.
      test_order = {vig_order[0] == 3 ? 2 : vig_order[0], vig_order[1] == 3 ? 2 : vig_order[1]};
      try_downgrade = (test_order != vig_order);
    }
    if (try_downgrade) {
      std::array<ChannelSolution, 3> sol_o2 = ch_sol;
      if (run_pass2(test_order, sol_o2)) {
        std::vector<double> ev2, rg2, bg2;
        std::vector<std::array<double, 3>> wb2;
        build_ev_wb(sol_o2, ev2, wb2, rg2, bg2);
        std::vector<double> ev2_total(n_total, 0.0);
        for (size_t s = 0; s < n; ++s)
          ev2_total[written[s]] = ev2[s];
        const HoldoutMetrics h2 = EvaluateHoldout(tracks, sol_o2, ev2_total, wb2, test_order);
        const bool any_bad_now = bad_vig[0] || bad_vig[1];
        if (any_bad_now || (h2.lum_samples > 0 &&
                            h2.lum_after_p50 <= holdout.lum_after_p50 + kHoldoutOrderMarginEv)) {
          // Only keep order 3 if it improved holdout by more than margin and stayed in bounds.
          if (any_bad_now || holdout.lum_after_p50 + kHoldoutOrderMarginEv > h2.lum_after_p50) {
            vig_order = test_order;
            ch_sol = std::move(sol_o2);
            ev_slot = std::move(ev2);
            wb_total = std::move(wb2);
            rg_slot = std::move(rg2);
            bg_slot = std::move(bg2);
            ev_total = std::move(ev2_total);
            holdout = h2;
          }
        }
      }
    }
  }

  for (int c = 0; c < 2; ++c) {
    for (int ch = 0; ch < 3; ++ch) {
      if (!VignettingInBounds(ch_sol[ch].vig[c], vig_order[c])) {
        vig_order[c] = 0;
        for (int k = 0; k < 3; ++k)
          ch_sol[k].vig[c] = {0.0, 0.0, 0.0};
        break;
      }
    }
  }

  for (int c = 0; c < 2; ++c)
    for (int ch = 0; ch < 3; ++ch)
      for (int a = 0; a < vig_order[c]; ++a)
        cal.vig_coeffs_rgb[c][ch][a] = ch_sol[ch].vig[c][a];
  cal.vig_order = vig_order;

  const int lc_id = int(largest_idx);
  std::vector<double> valid_evs;
  double ev_min = 0.0, ev_max = 0.0;
  double max_abs_rg = 0.0, max_abs_bg = 0.0;
  std::array<std::vector<double>, 2> rg_by_cam, bg_by_cam;
  for (size_t s = 0; s < n; ++s) {
    const uint32_t img_id = written[s];
    if (comp_id[s] != lc_id && train_obs_per_img[s] == 0)
      continue;
    cal.exposure_ev[img_id] = ev_slot[s];
    cal.exposure_valid[img_id] = 1;
    cal.wb_gains[img_id] = wb_total[img_id];
    valid_evs.push_back(ev_slot[s]);
    if (valid_evs.size() == 1)
      ev_min = ev_max = ev_slot[s];
    else {
      ev_min = std::min(ev_min, ev_slot[s]);
      ev_max = std::max(ev_max, ev_slot[s]);
    }
    max_abs_rg = std::max(max_abs_rg, std::abs(rg_slot[s]));
    max_abs_bg = std::max(max_abs_bg, std::abs(bg_slot[s]));
    const int cam = model.images[img_id].camera;
    rg_by_cam[cam].push_back(rg_slot[s]);
    bg_by_cam[cam].push_back(bg_slot[s]);
  }
  cal.reference_ev = ComputeKdeMode(valid_evs);

  const bool holdout_ok =
      holdout.lum_samples == 0 ||
      (holdout.lum_after_p50 <= kMaxHoldoutMedianEv &&
       holdout.lum_after_p50 < holdout.lum_before_p50);
  cal.calibrated = !valid_evs.empty() && holdout_ok;
  cal.wb_calibrated = cal.calibrated;

  Json vig_json = Json::array();
  Json vig_edge_json = Json::array();
  for (int c = 0; c < 2; ++c) {
    Json cam_ch = Json::array();
    Json cam_edge = Json::array();
    for (int ch = 0; ch < 3; ++ch) {
      cam_ch.push_back(
          {cal.vig_coeffs_rgb[c][ch][0], cal.vig_coeffs_rgb[c][ch][1], cal.vig_coeffs_rgb[c][ch][2]});
      cam_edge.push_back(EvaluateVignettingGain(cal.vig_coeffs_rgb[c][ch], vig_order[c], 1.0));
    }
    vig_json.push_back(std::move(cam_ch));
    vig_edge_json.push_back(std::move(cam_edge));
  }
  cal.report["calibrated"] = cal.calibrated;
  cal.report["trimmed_obs"] = trimmed_obs;
  cal.report["vignetting_order"] = {vig_order[0], vig_order[1]};
  cal.report["vignetting_coeffs_rgb"] = std::move(vig_json);
  cal.report["vignetting_edge_gain"] = std::move(vig_edge_json);
  cal.report["exposure"] = {{"calibrated", cal.calibrated},
                            {"valid_images", valid_evs.size()},
                            {"ev_min", ev_min},
                            {"ev_max", ev_max},
                            {"reference_ev", cal.reference_ev}};
  cal.report["white_balance"] = {{"calibrated", cal.wb_calibrated},
                                 {"max_abs_rg_ev", max_abs_rg},
                                 {"max_abs_bg_ev", max_abs_bg},
                                 {"cam_median_rg_ev", {MedianOf(rg_by_cam[0]), MedianOf(rg_by_cam[1])}},
                                 {"cam_median_bg_ev", {MedianOf(bg_by_cam[0]), MedianOf(bg_by_cam[1])}}};
  cal.report["holdout"] = {{"samples", holdout.lum_samples},
                           {"lum_before_p50_ev", holdout.lum_before_p50},
                           {"lum_before_p90_ev", holdout.lum_before_p90},
                           {"lum_after_p50_ev", holdout.lum_after_p50},
                           {"lum_after_p90_ev", holdout.lum_after_p90},
                           {"edge_after_p50_ev", holdout.edge_after_p50},
                           {"cross_lens_rg_p50_ev", holdout.cross_rg_after_p50},
                           {"cross_lens_bg_p50_ev", holdout.cross_bg_after_p50}};
  return cal;
}

void PlanSafeHomogenization(const Reconstruction& model, const std::vector<uint32_t>& written,
                            const std::array<PhotometricCamGrid, 2>& grids,
                            const std::vector<PhotometricViewData>& views,
                            const std::vector<cv::Mat>& clip_masks,
                            PhotometricCalibration& cal, std::vector<cv::Mat>& post_clip_masks) {
  post_clip_masks.assign(model.images.size(), {});
  if (!cal.calibrated)
    return;

  // Precompute per-camera RGB inverse vignetting maps on the 256 grid.
  std::array<std::array<cv::Mat, 3>, 2> inv_vig_maps;
  for (int c = 0; c < 2; ++c) {
    const auto& g = grids[c];
    for (int ch = 0; ch < 3; ++ch) {
      inv_vig_maps[c][ch].create(g.height, g.width, CV_32FC1);
      for (int y = 0; y < g.height; ++y) {
        float* vp = inv_vig_maps[c][ch].ptr<float>(y);
        for (int x = 0; x < g.width; ++x) {
          const float rho = g.rho[size_t(y) * size_t(g.width) + size_t(x)];
          vp[x] = float(
              1.0 / EvaluateVignettingGain(cal.vig_coeffs_rgb[c][ch], cal.vig_order[c], rho));
        }
      }
    }
  }

  size_t homog_cnt = 0, exact_zero_cnt = 0;
  double res_min = 0.0, res_max = 0.0, shift_min = 0.0, shift_max = 0.0;
  bool first = true;
  uint64_t total_static_px = 0, total_new_clip_px = 0, total_dark_sat_px = 0, total_post_mask_px = 0;
  double worst_new_clip_rate = 0.0;
  std::string worst_new_clip_name;

  std::vector<double> lum_vals, max_corr_vals;
  for (const uint32_t id : written) {
    if (!cal.exposure_valid[id])
      continue;
    const uint32_t cid = model.images[id].camera;
    const auto& g = grids[cid];
    const auto& view = views[id];
    const cv::Mat& clip = clip_masks[id];
    const bool has_clip = !clip.empty();
    const auto& wb = cal.wb_gains[id];

    lum_vals.clear();
    max_corr_vals.clear();
    size_t static_unmasked = 0, sat_unmasked = 0;
    for (int y = 0; y < g.height; ++y) {
      const uint8_t* mp = view.static_mask.ptr<uint8_t>(y);
      const uint8_t* cp = has_clip ? clip.ptr<uint8_t>(y) : nullptr;
      const float* sr = view.lum_srgb.ptr<float>(y);
      const float* mr = view.max_srgb.ptr<float>(y);
      const cv::Vec3f* rp = view.rgb_lin.ptr<cv::Vec3f>(y);
      const float* iv0 = inv_vig_maps[cid][0].ptr<float>(y);
      const float* iv1 = inv_vig_maps[cid][1].ptr<float>(y);
      const float* iv2 = inv_vig_maps[cid][2].ptr<float>(y);
      for (int x = 0; x < g.width; ++x) {
        if (!mp[x] || (cp && cp[x]))
          continue;
        ++static_unmasked;
        lum_vals.push_back(double(sr[x]));
        if (sr[x] >= 0.94f && mr[x] >= 0.975f) {
          ++sat_unmasked;
        } else {
          const double mx_c = std::max({double(rp[x][0]) * iv0[x] * wb[0],
                                        double(rp[x][1]) * iv1[x] * wb[1],
                                        double(rp[x][2]) * iv2[x] * wb[2]});
          max_corr_vals.push_back(mx_c);
        }
      }
    }

    const double want = cal.exposure_ev[id] - cal.reference_ev;
    double shift = 0.0;
    if (static_unmasked > 0) {
      const double p50 = PercentileOf(lum_vals, 0.50);
      const double p10 = PercentileOf(lum_vals, 0.10);
      if (want < 0.0) {
        const double m98 = max_corr_vals.empty() ? 1.0 : PercentileOf(max_corr_vals, 0.98);
        const double max_brighten =
            m98 < kSafeBrightenTargetLin
                ? std::min(kSafeMaxBrightenEv, std::log2(kSafeBrightenTargetLin / std::max(m98, 1e-4)))
                : 0.0;
        shift = std::max(want, -max_brighten);
      } else if (want > 0.0) {
        const double md50 =
            p50 > kSafeMinP50Srgb
                ? std::log2(std::max<double>(SrgbToLin(float(p50)), 1e-6) /
                            std::max<double>(SrgbToLin(float(kSafeMinP50Srgb)), 1e-6))
                : 0.0;
        const double md10 =
            p10 > kSafeMinP10Srgb
                ? std::log2(std::max<double>(SrgbToLin(float(p10)), 1e-6) /
                            std::max<double>(SrgbToLin(float(kSafeMinP10Srgb)), 1e-6))
                : 0.0;
        const double sat_share = double(sat_unmasked) / double(static_unmasked);
        double sat_cap = kSafeMaxDarkenEv;
        if (sat_share >= kSafeSatHiShare)
          sat_cap = 0.0;
        else if (sat_share > kSafeSatLoShare)
          sat_cap = kSafeMaxDarkenEv * (kSafeSatHiShare - sat_share) /
                    (kSafeSatHiShare - kSafeSatLoShare);
        shift = std::min({want, kSafeMaxDarkenEv, std::max(0.0, md50), std::max(0.0, md10), sat_cap});
      }
    }

    const double residual = want - shift;
    cal.pixel_ev_shift[id] = shift;
    cal.residual_ev[id] = residual;
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

    // Closed-loop post-correction clipping & darkened-saturation audit on the 256 grid.
    const double exp_gain = std::exp2(-shift);
    const float g0 = float(exp_gain * wb[0]);
    const float g1 = float(exp_gain * wb[1]);
    const float g2 = float(exp_gain * wb[2]);
    cv::Mat new_clip_raw = cv::Mat::zeros(g.height, g.width, CV_8UC1);
    cv::Mat dark_sat_raw = cv::Mat::zeros(g.height, g.width, CV_8UC1);
    uint64_t img_static = 0, img_new_clip = 0;
    for (int y = 0; y < g.height; ++y) {
      const uint8_t* mp = view.static_mask.ptr<uint8_t>(y);
      const uint8_t* cp = has_clip ? clip.ptr<uint8_t>(y) : nullptr;
      const float* sr = view.lum_srgb.ptr<float>(y);
      const float* mr = view.max_srgb.ptr<float>(y);
      const cv::Vec3f* rp = view.rgb_lin.ptr<cv::Vec3f>(y);
      const float* iv0 = inv_vig_maps[cid][0].ptr<float>(y);
      const float* iv1 = inv_vig_maps[cid][1].ptr<float>(y);
      const float* iv2 = inv_vig_maps[cid][2].ptr<float>(y);
      uint8_t* ncp = new_clip_raw.ptr<uint8_t>(y);
      uint8_t* dsp = dark_sat_raw.ptr<uint8_t>(y);
      for (int x = 0; x < g.width; ++x) {
        if (!mp[x])
          continue;
        ++img_static;
        const bool already_clipped = (cp && cp[x] != 0);
        const bool orig_sat = (sr[x] >= 0.94f && mr[x] >= 0.975f);
        if (!orig_sat) {
          const float r_out = rp[x][0] * iv0[x] * g0;
          const float g_out = rp[x][1] * iv1[x] * g1;
          const float b_out = rp[x][2] * iv2[x] * g2;
          const float lum_clamped = 0.2126f * std::min(1.0f, r_out) +
                                    0.7152f * std::min(1.0f, g_out) +
                                    0.0722f * std::min(1.0f, b_out);
          if (lum_clamped >= float(kPostClipLinThresh)) {
            ++img_new_clip;
            if (!already_clipped)
              ncp[x] = 255;
          }
        } else if (shift > kPostDarkSatShiftEv && !already_clipped) {
          ++total_dark_sat_px;
          dsp[x] = 255;
        }
      }
    }
    total_static_px += img_static;
    total_new_clip_px += img_new_clip;
    if (img_static > 0) {
      const double rate = double(img_new_clip) / double(img_static);
      if (rate > worst_new_clip_rate) {
        worst_new_clip_rate = rate;
        worst_new_clip_name = std::to_string(id);
      }
    }
    cv::Mat post_mask = dark_sat_raw;
    if (cv::countNonZero(new_clip_raw) >= kPostClipMinCompArea) {
      cv::Mat labels, stats, centroids;
      const int cnt =
          cv::connectedComponentsWithStats(new_clip_raw, labels, stats, centroids, 8);
      for (int k = 1; k < cnt; ++k) {
        if (stats.at<int>(k, cv::CC_STAT_AREA) >= kPostClipMinCompArea)
          post_mask.setTo(255, labels == k);
      }
    }
    const int post_cnt = cv::countNonZero(post_mask);
    if (post_cnt > 0) {
      total_post_mask_px += uint64_t(post_cnt);
      post_clip_masks[id] = std::move(post_mask);
    }
  }

  cal.report["exposure"]["homogenized_images"] = homog_cnt;
  cal.report["exposure"]["exact_zero_residual_images"] = exact_zero_cnt;
  cal.report["exposure"]["residual_ev_min"] = res_min;
  cal.report["exposure"]["residual_ev_max"] = res_max;
  cal.report["exposure"]["pixel_shift_min"] = shift_min;
  cal.report["exposure"]["pixel_shift_max"] = shift_max;
  cal.report["homogenization"] = {
      {"mode", "safe_bounded"},
      {"new_clip_rate", total_static_px ? double(total_new_clip_px) / double(total_static_px) : 0.0},
      {"worst_frame_new_clip_rate", worst_new_clip_rate},
      {"worst_frame_id", worst_new_clip_name},
      {"darkened_sat_rate", total_static_px ? double(total_dark_sat_px) / double(total_static_px) : 0.0},
      {"post_clip_grid_pixels", total_post_mask_px}};
}

Json PhotometricRules() {
  return {{"photometric_version", 3},
          {"min_srgb", kMinSrgb},
          {"max_srgb", kMaxSrgb},
          {"max_seed_grad", kMaxSeedGrad},
          {"max_track_grad", kMaxTrackGrad},
          {"min_view_cos", kMinViewCos},
          {"las_voxel_size_m", kLasVoxelSizeM},
          {"las_max_project_dist_m", kLasMaxProjectDistM},
          {"las_max_view_angle_cos", kLasMaxViewAngleCos},
          {"max_las_seeds_total", kMaxLasSeedsTotal},
          {"max_sfm_tracks", kMaxSfmTracks},
          {"sfm_max_reproj_px", kSfmMaxReprojPx},
          {"holdout_block_m", kHoldoutBlockM},
          {"vig_min_gain", kVigMinGain},
          {"vig_max_gain", kVigMaxGain},
          {"pass1_huber_ev", kPass1HuberEv},
          {"pass2_huber_ev", kPass2HuberEv},
          {"outlier_trim_ev", kOutlierTrimEv},
          {"wb_smooth_max_dt_s", kWbSmoothMaxDtS},
          {"wb_smooth_weight", kWbSmoothWeight},
          {"wb_max_abs_ev", kWbMaxAbsEv},
          {"kde_bandwidth_ev", kKdeBandwidthEv},
          {"safe_max_brighten_ev", kSafeMaxBrightenEv},
          {"safe_brighten_target_lin", kSafeBrightenTargetLin},
          {"safe_max_darken_ev", kSafeMaxDarkenEv},
          {"safe_min_p50_srgb", kSafeMinP50Srgb},
          {"safe_min_p10_srgb", kSafeMinP10Srgb},
          {"safe_sat_lo_share", kSafeSatLoShare},
          {"safe_sat_hi_share", kSafeSatHiShare},
          {"post_clip_lin_thresh", kPostClipLinThresh},
          {"post_dark_sat_shift_ev", kPostDarkSatShiftEv}};
}
}  // namespace geosun

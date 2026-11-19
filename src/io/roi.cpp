#include "io/roi.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace geosun {
namespace {

double Cross2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
  return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

bool OnSegment2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& p,
                 double eps) {
  return p.x() >= std::min(a.x(), b.x()) - eps && p.x() <= std::max(a.x(), b.x()) + eps &&
         p.y() >= std::min(a.y(), b.y()) - eps && p.y() <= std::max(a.y(), b.y()) + eps &&
         std::abs(Cross2D(a, b, p)) <= eps * std::max(1.0, (b - a).norm());
}

bool SegmentsIntersect2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                         const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
  constexpr double kEps = 1e-11;
  if (std::max(a.x(), b.x()) + kEps < std::min(c.x(), d.x()) ||
      std::max(c.x(), d.x()) + kEps < std::min(a.x(), b.x()) ||
      std::max(a.y(), b.y()) + kEps < std::min(c.y(), d.y()) ||
      std::max(c.y(), d.y()) + kEps < std::min(a.y(), b.y()))
    return false;
  const double len_ab = std::max(1.0, (b - a).norm());
  const double len_cd = std::max(1.0, (d - c).norm());
  const double d1 = Cross2D(a, b, c) / len_ab;
  const double d2 = Cross2D(a, b, d) / len_ab;
  const double d3 = Cross2D(c, d, a) / len_cd;
  const double d4 = Cross2D(c, d, b) / len_cd;
  if (((d1 > kEps && d2 < -kEps) || (d1 < -kEps && d2 > kEps)) &&
      ((d3 > kEps && d4 < -kEps) || (d3 < -kEps && d4 > kEps)))
    return true;
  if (std::abs(d1) <= kEps && OnSegment2D(a, b, c, kEps))
    return true;
  if (std::abs(d2) <= kEps && OnSegment2D(a, b, d, kEps))
    return true;
  if (std::abs(d3) <= kEps && OnSegment2D(c, d, a, kEps))
    return true;
  if (std::abs(d4) <= kEps && OnSegment2D(c, d, b, kEps))
    return true;
  return false;
}

std::vector<Eigen::Vector2d> ConvexHull2D(std::vector<Eigen::Vector2d> pts) {
  std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
  });
  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                          return (a - b).squaredNorm() <= 1e-12;
                        }),
            pts.end());
  if (pts.size() <= 2)
    return pts;
  std::vector<Eigen::Vector2d> lower, upper;
  for (const auto& p : pts) {
    while (lower.size() >= 2 && Cross2D(lower[lower.size() - 2], lower.back(), p) <= 0.0)
      lower.pop_back();
    lower.push_back(p);
  }
  for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
    while (upper.size() >= 2 && Cross2D(upper[upper.size() - 2], upper.back(), *it) <= 0.0)
      upper.pop_back();
    upper.push_back(*it);
  }
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

bool IntersectForwardRays(const Eigen::Vector2d& a0, const Eigen::Vector2d& a1,
                          const Eigen::Vector2d& b0, const Eigen::Vector2d& b1,
                          Eigen::Vector2d& out_q) {
  const Eigen::Vector2d da = a1 - a0;
  const Eigen::Vector2d db = b1 - b0;
  const double denom = da.x() * db.y() - da.y() * db.x();
  if (std::abs(denom) <= 1e-12 * std::max(1.0, da.norm() * db.norm()))
    return false;
  const Eigen::Vector2d diff = b0 - a0;
  const double ta = (diff.x() * db.y() - diff.y() * db.x()) / denom;
  const double tb = (diff.x() * da.y() - diff.y() * da.x()) / denom;
  if (!(ta > 1.0 && tb > 1.0) || !std::isfinite(ta) || !std::isfinite(tb))
    return false;
  out_q = a0 + ta * da;
  return out_q.allFinite();
}

std::vector<Eigen::Vector2d> CircumscribedSimplifyConvex(std::vector<Eigen::Vector2d> hull,
                                                         int max_vertices) {
  while (static_cast<int>(hull.size()) > max_vertices && hull.size() >= 4) {
    const size_t n = hull.size();
    size_t best_edge = n;
    double best_added_area = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_q = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < n; ++i) {
      const size_t im1 = (i + n - 1) % n;
      const size_t ip1 = (i + 1) % n;
      const size_t ip2 = (i + 2) % n;
      Eigen::Vector2d q;
      if (!IntersectForwardRays(hull[im1], hull[i], hull[ip2], hull[ip1], q))
        continue;
      const double added_area = 0.5 * std::abs(Cross2D(hull[i], hull[ip1], q));
      if (added_area < best_added_area) {
        best_added_area = added_area;
        best_edge = i;
        best_q = q;
      }
    }
    if (best_edge == n)
      break;
    const size_t ip1 = (best_edge + 1) % n;
    hull[best_edge] = best_q;
    hull.erase(hull.begin() + static_cast<ptrdiff_t>(ip1));
  }
  return hull;
}

void RasterizeBresenham(int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                        std::unordered_set<uint64_t>& out_cells) {
  int32_t dx = std::abs(x1 - x0);
  int32_t dy = -std::abs(y1 - y0);
  const int32_t sx = x0 < x1 ? 1 : -1;
  const int32_t sy = y0 < y1 ? 1 : -1;
  int32_t err = dx + dy;
  for (;;) {
    out_cells.insert(PackGridCell2D(x0, y0));
    if (x0 == x1 && y0 == y1)
      break;
    const int32_t e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

std::vector<Eigen::Vector2d> BoundingRectPolygon(const std::vector<Eigen::Vector2d>& pts,
                                                 double margin) {
  Eigen::Vector2d lo(std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity());
  Eigen::Vector2d hi(-std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity());
  for (const auto& p : pts) {
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  const double pad = std::max(0.5, margin);
  lo.array() -= pad;
  hi.array() += pad;
  return {{lo.x(), lo.y()}, {hi.x(), lo.y()}, {hi.x(), hi.y()}, {lo.x(), hi.y()}};
}

}  // namespace

bool PointInPolygon2D(const std::vector<Eigen::Vector2d>& poly, double x, double y) {
  bool inside = false;
  const size_t n = poly.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const auto& a = poly[i];
    const auto& b = poly[j];
    if ((a.y() > y) != (b.y() > y) &&
        x < (b.x() - a.x()) * (y - a.y()) / (b.y() - a.y()) + a.x())
      inside = !inside;
  }
  return inside;
}

double PolygonArea2D(const std::vector<Eigen::Vector2d>& poly) {
  if (poly.size() < 3)
    return 0.0;
  double twice = 0.0;
  for (size_t i = 0; i < poly.size(); ++i) {
    const auto& a = poly[i];
    const auto& b = poly[(i + 1) % poly.size()];
    twice += a.x() * b.y() - b.x() * a.y();
  }
  return 0.5 * std::abs(twice);
}

bool IsSimpleNonSelfIntersecting(const std::vector<Eigen::Vector2d>& poly) {
  const size_t n = poly.size();
  if (n < 3)
    return false;
  for (size_t i = 0; i < n; ++i) {
    if (!poly[i].allFinite())
      return false;
    const auto& next = poly[(i + 1) % n];
    if ((next - poly[i]).squaredNorm() <= 1e-12)
      return false;
    for (size_t j = i + 1; j < n; ++j)
      if ((poly[i] - poly[j]).squaredNorm() <= 1e-12)
        return false;
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t in = (i + 1) % n;
    const size_t ip = (i + n - 1) % n;
    const Eigen::Vector2d e0 = poly[i] - poly[ip];
    const Eigen::Vector2d e1 = poly[in] - poly[i];
    if (std::abs(e0.x() * e1.y() - e0.y() * e1.x()) <=
            1e-11 * std::max(1.0, e0.norm() * e1.norm()) &&
        e0.dot(e1) < 0.0)
      return false;
    for (size_t j = i + 1; j < n; ++j) {
      const size_t jn = (j + 1) % n;
      if (j == in || jn == i)
        continue;
      if (SegmentsIntersect2D(poly[i], poly[in], poly[j], poly[jn]))
        return false;
    }
  }
  return true;
}

RoiPolygon BuildRoiPolygon2D(const std::unordered_map<uint64_t, uint32_t>& raw_cell_counts,
                             const Reconstruction& model, const RoiParams& params) {
  if (!(params.cell_size_m > 0.0) || params.max_vertices < 3)
    throw std::runtime_error("Invalid ROI construction parameters");

  std::vector<Eigen::Vector3d> cam_centers;
  cam_centers.reserve(model.images.size());
  std::unordered_set<uint64_t> cam_cells;
  std::array<bool, 2> have_prev{false, false};
  std::array<std::pair<int32_t, int32_t>, 2> prev_cell{};
  for (const auto& img : model.images) {
    if (!img.registered)
      continue;
    const Eigen::Vector3d c = img.pose.Center();
    if (!c.allFinite())
      throw std::runtime_error("Nonfinite registered camera center during ROI construction");
    cam_centers.push_back(c);
    int32_t ix = 0, iy = 0;
    if (!GridCellCoord2D(c.x(), c.y(), params.cell_size_m, ix, iy))
      throw std::runtime_error("Camera coordinates exceed ROI grid extent");
    cam_cells.insert(PackGridCell2D(ix, iy));
    const uint32_t cam = std::min<uint32_t>(img.camera, 1);
    if (have_prev[cam]) {
      if (std::max(std::abs(ix - prev_cell[cam].first),
                   std::abs(iy - prev_cell[cam].second)) <= 20) {
        RasterizeBresenham(prev_cell[cam].first, prev_cell[cam].second, ix, iy, cam_cells);
      }
    }
    prev_cell[cam] = {ix, iy};
    have_prev[cam] = true;
  }
  if (cam_centers.empty())
    throw std::runtime_error("Cannot build ROI without registered cameras");

  std::unordered_set<uint64_t> occ_cells = cam_cells;
  uint64_t total_raw_points = 0;
  for (const auto& [key, count] : raw_cell_counts) {
    total_raw_points += count;
    if (count >= params.min_cell_points)
      occ_cells.insert(key);
  }

  int32_t min_ix = std::numeric_limits<int32_t>::max();
  int32_t max_ix = std::numeric_limits<int32_t>::lowest();
  int32_t min_iy = std::numeric_limits<int32_t>::max();
  int32_t max_iy = std::numeric_limits<int32_t>::lowest();
  for (uint64_t key : occ_cells) {
    const auto [ix, iy] = UnpackGridCell2D(key);
    min_ix = std::min(min_ix, ix);
    max_ix = std::max(max_ix, ix);
    min_iy = std::min(min_iy, iy);
    max_iy = std::max(max_iy, iy);
  }
  const int dilate_radius =
      std::max(1, static_cast<int>(std::lround(params.margin_xy_m / params.cell_size_m)));
  const int pad = static_cast<int>(std::ceil(params.margin_xy_m / params.cell_size_m)) + 3;
  const int64_t ox = static_cast<int64_t>(min_ix) - pad;
  const int64_t oy = static_cast<int64_t>(min_iy) - pad;
  const int64_t width64 = static_cast<int64_t>(max_ix) - min_ix + 1 + 2LL * pad;
  const int64_t height64 = static_cast<int64_t>(max_iy) - min_iy + 1 + 2LL * pad;
  if (width64 <= 0 || height64 <= 0 || width64 > 16384 || height64 > 16384 ||
      width64 * height64 > 64LL * 1024 * 1024)
    throw std::runtime_error("ROI occupancy grid exceeds supported dimensions");
  const int width = static_cast<int>(width64);
  const int height = static_cast<int>(height64);

  cv::Mat occ_img(height, width, CV_8UC1, cv::Scalar(0));
  for (uint64_t key : occ_cells) {
    const auto [ix, iy] = UnpackGridCell2D(key);
    occ_img.at<uint8_t>(static_cast<int>(iy - oy), static_cast<int>(ix - ox)) = 255;
  }
  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * dilate_radius + 1, 2 * dilate_radius + 1));
  cv::Mat dil_img;
  cv::dilate(occ_img, dil_img, kernel);

  cv::Mat main_img(height, width, CV_8UC1, cv::Scalar(0));
  std::deque<cv::Point> bfs;
  for (uint64_t key : cam_cells) {
    const auto [ix, iy] = UnpackGridCell2D(key);
    const int gx = static_cast<int>(ix - ox);
    const int gy = static_cast<int>(iy - oy);
    if (dil_img.at<uint8_t>(gy, gx) && !main_img.at<uint8_t>(gy, gx)) {
      main_img.at<uint8_t>(gy, gx) = 255;
      bfs.emplace_back(gx, gy);
    }
  }
  static constexpr int kDx[4] = {1, -1, 0, 0};
  static constexpr int kDy[4] = {0, 0, 1, -1};
  while (!bfs.empty()) {
    const cv::Point p = bfs.front();
    bfs.pop_front();
    for (int d = 0; d < 4; ++d) {
      const int nx = p.x + kDx[d];
      const int ny = p.y + kDy[d];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height)
        continue;
      if (dil_img.at<uint8_t>(ny, nx) && !main_img.at<uint8_t>(ny, nx)) {
        main_img.at<uint8_t>(ny, nx) = 255;
        bfs.emplace_back(nx, ny);
      }
    }
  }

  RoiPolygon roi;
  roi.cameras_total = cam_centers.size();
  for (uint64_t key : occ_cells) {
    const auto [ix, iy] = UnpackGridCell2D(key);
    if (main_img.at<uint8_t>(static_cast<int>(iy - oy), static_cast<int>(ix - ox)))
      ++roi.occupied_cells;
  }
  uint64_t kept_raw_points = 0;
  for (const auto& [key, count] : raw_cell_counts) {
    const auto [ix, iy] = UnpackGridCell2D(key);
    const int64_t gx = static_cast<int64_t>(ix) - ox;
    const int64_t gy = static_cast<int64_t>(iy) - oy;
    if (gx >= 0 && gx < width && gy >= 0 && gy < height &&
        main_img.at<uint8_t>(static_cast<int>(gy), static_cast<int>(gx)))
      kept_raw_points += count;
  }
  roi.kept_las_fraction =
      total_raw_points > 0 ? static_cast<double>(kept_raw_points) / static_cast<double>(total_raw_points)
                           : 1.0;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(main_img.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.empty())
    throw std::runtime_error("ROI contour extraction produced no foreground region");

  auto to_world = [&](const cv::Point& gp) {
    return Eigen::Vector2d((static_cast<double>(ox + gp.x) + 0.5) * params.cell_size_m,
                           (static_cast<double>(oy + gp.y) + 0.5) * params.cell_size_m);
  };

  std::vector<Eigen::Vector2d> world_pts;
  if (contours.size() == 1) {
    world_pts.reserve(contours[0].size());
    for (const auto& gp : contours[0])
      world_pts.push_back(to_world(gp));
  } else {
    std::vector<Eigen::Vector2d> all_pts;
    for (const auto& c : contours)
      for (const auto& gp : c)
        all_pts.push_back(to_world(gp));
    world_pts = ConvexHull2D(std::move(all_pts));
  }

  auto all_cameras_inside = [&](const std::vector<Eigen::Vector2d>& poly) {
    for (const auto& c : cam_centers)
      if (!PointInPolygon2D(poly, c.x(), c.y()))
        return false;
    return true;
  };

  std::vector<Eigen::Vector2d> chosen;
  if (world_pts.size() >= 3) {
    const Eigen::Vector2d anchor = world_pts[0];
    std::vector<cv::Point2f> local_pts;
    local_pts.reserve(world_pts.size());
    for (const auto& p : world_pts)
      local_pts.emplace_back(static_cast<float>(p.x() - anchor.x()),
                             static_cast<float>(p.y() - anchor.y()));
    double eps = 0.35;
    for (int iter = 0; iter < 32; ++iter, eps *= 1.25) {
      std::vector<cv::Point2f> approx;
      cv::approxPolyDP(local_pts, approx, eps, true);
      if (approx.size() < 3 || static_cast<int>(approx.size()) > params.max_vertices)
        continue;
      std::vector<Eigen::Vector2d> cand;
      cand.reserve(approx.size());
      for (const auto& q : approx)
        cand.emplace_back(anchor.x() + static_cast<double>(q.x),
                          anchor.y() + static_cast<double>(q.y));
      if (PolygonArea2D(cand) > 1e-3 && IsSimpleNonSelfIntersecting(cand) &&
          all_cameras_inside(cand)) {
        chosen = std::move(cand);
        break;
      }
    }
  }

  if (chosen.empty()) {
    std::vector<Eigen::Vector2d> hull_inputs = world_pts;
    for (const auto& c : cam_centers)
      hull_inputs.emplace_back(c.x(), c.y());
    auto hull = ConvexHull2D(std::move(hull_inputs));
    if (static_cast<int>(hull.size()) > params.max_vertices)
      hull = CircumscribedSimplifyConvex(std::move(hull), params.max_vertices);
    if (hull.size() >= 3 && static_cast<int>(hull.size()) <= params.max_vertices &&
        PolygonArea2D(hull) > 1e-3 && IsSimpleNonSelfIntersecting(hull) &&
        all_cameras_inside(hull)) {
      chosen = std::move(hull);
    } else {
      std::vector<Eigen::Vector2d> rect_inputs = world_pts;
      for (const auto& c : cam_centers)
        rect_inputs.emplace_back(c.x(), c.y());
      chosen = BoundingRectPolygon(rect_inputs, params.margin_xy_m);
    }
  }

  if (chosen.size() < 3 || static_cast<int>(chosen.size()) > params.max_vertices ||
      !(PolygonArea2D(chosen) > 1e-4) || !IsSimpleNonSelfIntersecting(chosen))
    throw std::runtime_error("Failed to construct valid simple 2D ROI polygon");

  roi.vertices_xy = std::move(chosen);
  roi.area_m2 = PolygonArea2D(roi.vertices_xy);
  roi.cameras_inside = 0;
  for (const auto& c : cam_centers)
    if (PointInPolygon2D(roi.vertices_xy, c.x(), c.y()))
      ++roi.cameras_inside;
  if (roi.cameras_inside != roi.cameras_total)
    throw std::runtime_error("ROI polygon does not contain all registered cameras");
  return roi;
}

void FinalizeRoiZRange(RoiPolygon& roi, std::vector<float>& inside_z_samples,
                       const Reconstruction& model, const RoiParams& params) {
  double cam_z_min = std::numeric_limits<double>::infinity();
  double cam_z_max = -std::numeric_limits<double>::infinity();
  size_t cam_count = 0;
  for (const auto& img : model.images) {
    if (!img.registered)
      continue;
    const double cz = img.pose.Center().z();
    if (!std::isfinite(cz))
      throw std::runtime_error("Nonfinite registered camera Z during ROI finalization");
    cam_z_min = std::min(cam_z_min, cz);
    cam_z_max = std::max(cam_z_max, cz);
    ++cam_count;
  }
  if (!cam_count)
    throw std::runtime_error("Cannot finalize ROI Z range without registered cameras");

  if (!inside_z_samples.empty()) {
    const size_t n = inside_z_samples.size();
    const size_t idx001 = static_cast<size_t>(std::floor(0.001 * static_cast<double>(n - 1)));
    const size_t idx999 = static_cast<size_t>(std::floor(0.999 * static_cast<double>(n - 1)));
    std::nth_element(inside_z_samples.begin(),
                     inside_z_samples.begin() + static_cast<ptrdiff_t>(idx001),
                     inside_z_samples.end());
    const double z_p001 = inside_z_samples[idx001];
    std::nth_element(inside_z_samples.begin() + static_cast<ptrdiff_t>(idx001),
                     inside_z_samples.begin() + static_cast<ptrdiff_t>(idx999),
                     inside_z_samples.end());
    const double z_p999 = inside_z_samples[idx999];
    const double z_max_las = *std::max_element(
        inside_z_samples.begin() + static_cast<ptrdiff_t>(idx999), inside_z_samples.end());

    roi.z_min = std::min(z_p001, cam_z_min - 2.0) - params.margin_z_bottom_m;
    const double z_max_raw = std::max(
        {z_p999 + params.margin_z_top_m, z_max_las + 1.0, cam_z_max + params.margin_z_top_m});
    roi.z_max = std::min(z_max_raw, std::max(z_p999, cam_z_max) + params.max_top_above_p999_m);
  } else {
    roi.z_min = cam_z_min - 2.0 - params.margin_z_bottom_m;
    roi.z_max = cam_z_max + params.margin_z_top_m;
  }

  if (!(roi.z_max > roi.z_min + 0.1) || !(roi.z_min <= cam_z_min && cam_z_max <= roi.z_max))
    throw std::runtime_error("Invalid finalized ROI Z range");

  Eigen::Vector2d lo(std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity());
  Eigen::Vector2d hi(-std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity());
  for (const auto& v : roi.vertices_xy) {
    lo = lo.cwiseMin(v);
    hi = hi.cwiseMax(v);
  }
  roi.bounds_min = Eigen::Vector3d(lo.x(), lo.y(), roi.z_min);
  roi.bounds_max = Eigen::Vector3d(hi.x(), hi.y(), roi.z_max);
}

Json RoiToJson(const RoiPolygon& roi, const RoiParams& params,
               const std::string& coordinate_frame) {
  Json vertices = Json::array();
  for (const auto& v : roi.vertices_xy)
    vertices.push_back({v.x(), v.y()});
  return {
      {"version", 1},
      {"type", "polygon_prism"},
      {"source", "las_camera_seeded_occupancy"},
      {"coordinate_frame", coordinate_frame},
      {"z_min", roi.z_min},
      {"z_max", roi.z_max},
      {"vertices_xy", std::move(vertices)},
      {"bounds_min", {roi.bounds_min.x(), roi.bounds_min.y(), roi.bounds_min.z()}},
      {"bounds_max", {roi.bounds_max.x(), roi.bounds_max.y(), roi.bounds_max.z()}},
      {"params",
       {{"cell_size_m", params.cell_size_m},
        {"min_cell_points", params.min_cell_points},
        {"margin_xy_m", params.margin_xy_m},
        {"margin_z_bottom_m", params.margin_z_bottom_m},
        {"margin_z_top_m", params.margin_z_top_m},
        {"max_top_above_p999_m", params.max_top_above_p999_m},
        {"max_vertices", params.max_vertices}}},
      {"stats",
       {{"area_m2", roi.area_m2},
        {"vertex_count", roi.vertices_xy.size()},
        {"occupied_cells", roi.occupied_cells},
        {"kept_las_fraction", roi.kept_las_fraction},
        {"cameras_inside", roi.cameras_inside},
        {"cameras_total", roi.cameras_total}}},
  };
}

RoiPolygon ValidateRoiJson(const Json& roi_json, const Reconstruction& model,
                           const std::string& expected_coordinate_frame) {
  if (!roi_json.is_object())
    throw std::runtime_error("ROI file must be a JSON object");
  if (roi_json.value("version", 0) != 1 || roi_json.value("type", "") != "polygon_prism")
    throw std::runtime_error("Unsupported ROI version or type");
  if (roi_json.value("coordinate_frame", "") != expected_coordinate_frame)
    throw std::runtime_error("ROI coordinate_frame differs from initialization");

  const double z_min = roi_json.at("z_min").get<double>();
  const double z_max = roi_json.at("z_max").get<double>();
  if (!std::isfinite(z_min) || !std::isfinite(z_max) || !(z_max > z_min))
    throw std::runtime_error("Invalid ROI Z range");

  const auto& verts = roi_json.at("vertices_xy");
  if (!verts.is_array() || verts.size() < 3 || verts.size() > 64)
    throw std::runtime_error("ROI vertices_xy must contain 3 to 64 points");

  RoiPolygon roi;
  roi.z_min = z_min;
  roi.z_max = z_max;
  roi.vertices_xy.reserve(verts.size());
  Eigen::Vector2d lo(std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity());
  Eigen::Vector2d hi(-std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity());
  for (const auto& pt : verts) {
    if (!pt.is_array() || pt.size() != 2)
      throw std::runtime_error("Invalid ROI 2D vertex");
    const double x = pt.at(0).get<double>();
    const double y = pt.at(1).get<double>();
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::runtime_error("Nonfinite ROI vertex");
    roi.vertices_xy.emplace_back(x, y);
    lo = lo.cwiseMin(roi.vertices_xy.back());
    hi = hi.cwiseMax(roi.vertices_xy.back());
  }
  roi.area_m2 = PolygonArea2D(roi.vertices_xy);
  if (!(roi.area_m2 > 1e-4) || !IsSimpleNonSelfIntersecting(roi.vertices_xy))
    throw std::runtime_error("ROI polygon is degenerate or self-intersecting");

  roi.bounds_min = Eigen::Vector3d(lo.x(), lo.y(), z_min);
  roi.bounds_max = Eigen::Vector3d(hi.x(), hi.y(), z_max);
  for (const auto& img : model.images) {
    if (!img.registered)
      continue;
    ++roi.cameras_total;
    const Eigen::Vector3d c = img.pose.Center();
    if (c.z() >= z_min && c.z() <= z_max && PointInPolygon2D(roi.vertices_xy, c.x(), c.y()))
      ++roi.cameras_inside;
  }
  if (roi.cameras_inside != roi.cameras_total)
    throw std::runtime_error("ROI polygon misses registered camera(s)");
  return roi;
}

}  // namespace geosun

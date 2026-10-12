#include "registration.h"

#include <colmap/estimators/solvers/similarity_transform.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <sstream>

#include "array_io.h"
#include "scene.h"
#include "sfm_pipeline.h"

namespace registration {
constexpr double Pi = 3.14159265358979323846;
double Quantile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.;
  std::sort(values.begin(), values.end());
  const double x = fraction * (values.size() - 1);
  const size_t i = size_t(x);
  return values[i] +
         (values[std::min(i + 1, values.size() - 1)] - values[i]) * (x - i);
}
Json Statistics(const std::vector<double>& values) {
  std::vector<double> finite;
  for (double x : values)
    if (std::isfinite(x)) finite.push_back(x);
  Json r = {{"count", finite.size()}};
  if (finite.empty()) {
    r["median"] = nullptr;
    r["p90"] = nullptr;
    r["max"] = nullptr;
  } else {
    r["median"] = Quantile(finite, .5);
    r["p90"] = Quantile(finite, .9);
    r["max"] = *std::max_element(finite.begin(), finite.end());
  }
  return r;
}
Json Matrix(const colmap::Sim3d& transform) {
  Json m = Json::array();
  const auto a = transform.ToMatrix();
  for (int i = 0; i < 3; ++i) m.push_back({a(i, 0), a(i, 1), a(i, 2), a(i, 3)});
  m.push_back({0., 0., 0., 1.});
  return m;
}
double Huber(double value, double cutoff) {
  const double a = std::abs(value);
  return a <= cutoff ? .5 * value * value : cutoff * (a - .5 * cutoff);
}
void Surface::Index() {
  tree = std::make_unique<Tree>(3, cloud,
                                nanoflann::KDTreeSingleIndexAdaptorParams(10));
  tree->buildIndex();
}
template <typename Function>
void Parallel(size_t size, int workers, Function function) {
  std::atomic<size_t> next{0};
  std::vector<std::future<void>> jobs;
  for (int i = 0; i < workers; ++i)
    jobs.push_back(std::async(std::launch::async, [&] {
      for (;;) {
        const size_t start = next.fetch_add(1024);
        if (start >= size) break;
        for (size_t j = start; j < std::min(size, start + 1024); ++j)
          function(j);
      }
    }));
  for (auto& job : jobs) job.get();
}
std::unique_ptr<Surface> Surfaces(const Points& points, double voxel,
                                  int workers) {
  struct Record {
    Tile key;
    size_t index;
  };
  std::vector<Record> records;
  records.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    Tile key;
    for (int j = 0; j < 3; ++j)
      key[j] = int64_t(std::floor(points[i][j] / voxel));
    records.push_back({key, i});
  }
  std::stable_sort(records.begin(), records.end(),
                   [](const auto& a, const auto& b) { return a.key < b.key; });
  auto surface = std::make_unique<Surface>();
  surface->voxel = voxel;
  for (size_t i = 0; i < records.size();) {
    size_t end = i + 1;
    Eigen::Vector3d sum = points[records[i].index];
    while (end < records.size() && records[end].key == records[i].key)
      sum += points[records[end++].index];
    surface->points.push_back(sum / double(end - i));
    i = end;
  }
  if (surface->points.size() < 20)
    throw std::runtime_error("Too few spatially distinct LAS points");
  surface->Index();
  const size_t size = surface->points.size();
  surface->centres.resize(size);
  surface->normals.resize(size);
  surface->noise.resize(size);
  surface->valid.resize(size);
  const size_t k = std::min<size_t>(20, size);
  Parallel(size, workers, [&](size_t i) {
    std::array<size_t, 20> ids{};
    std::array<double, 20> squared{};
    surface->tree->knnSearch(surface->points[i].data(), k, ids.data(),
                             squared.data());
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    for (size_t j = 0; j < k; ++j) centre += surface->points[ids[j]];
    centre /= double(k);
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t j = 0; j < k; ++j) {
      const Eigen::Vector3d d = surface->points[ids[j]] - centre;
      covariance += d * d.transpose();
    }
    covariance /= double(k);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solve(covariance);
    if (solve.info() != Eigen::Success)
      throw std::runtime_error("LAS plane eigensolve failed");
    const Eigen::Vector3d values = solve.eigenvalues().cwiseMax(0.);
    surface->centres[i] = centre;
    surface->normals[i] = solve.eigenvectors().col(0);
    surface->noise[i] = std::sqrt(values[0]);
    surface->valid[i] =
        values[0] / std::max(values.sum(), 1e-12) < .04 &&
        values[1] > .04 * values[2] &&
        squared[k - 1] <= std::pow(std::max(.35, voxel * 5), 2) &&
        surface->noise[i] <= .08;
  });
  return surface;
}
std::unique_ptr<Surface> CoherentSurfaces(const Points& points, double voxel,
                                          int workers) {
  auto fine = Surfaces(points, voxel, workers);
  auto coarse = Surfaces(points, 2 * voxel, workers);
  fine->coherent.resize(fine->points.size());
  Parallel(fine->points.size(), workers, [&](size_t i) {
    size_t id;
    double squared;
    coarse->tree->knnSearch(fine->points[i].data(), 1, &id, &squared);
    fine->coherent[i] =
        fine->valid[i] && coarse->valid[id] && fine->noise[i] <= .025 &&
        coarse->noise[id] <= .025 &&
        std::abs(fine->normals[i].dot(coarse->normals[id])) >=
            std::cos(15 * Pi / 180) &&
        std::abs(
            (fine->centres[i] - coarse->centres[id]).dot(fine->normals[i])) <=
            .025;
  });
  return fine;
}
std::vector<Residual> Residuals(const Points& points, const Surface& surface,
                                double maximum) {
  std::vector<Residual> result;
  result.reserve(points.size());
  for (const auto& p : points) {
    size_t id;
    double square;
    surface.tree->knnSearch(p.data(), 1, &id, &square);
    const double distance = std::sqrt(square),
                 plane = (p - surface.centres[id]).dot(surface.normals[id]);
    result.push_back(
        {id, distance, plane, distance <= maximum && bool(surface.valid[id])});
  }
  return result;
}
Regions Partition(const Points& points) {
  Regions r;
  r.origin = Eigen::Vector3d::Constant(INFINITY);
  double span = 0.;
  for (int j = 0; j < 3; ++j) {
    std::vector<double> values;
    for (const auto& p : points) {
      r.origin[j] = std::min(r.origin[j], p[j]);
      values.push_back(p[j]);
    }
    span = std::max(span, Quantile(values, .98) - Quantile(values, .02));
  }
  if (!(span > 0))
    throw std::runtime_error("Registration points have no spatial extent");
  const size_t requested = std::clamp<size_t>(points.size() / 100, 8, 256);
  double low = span / 128, high = span * 2;
  size_t best = SIZE_MAX, best_count = 0;
  for (int iteration = 0; iteration < 12; ++iteration) {
    const double size = (low + high) / 2;
    std::vector<Tile> tiles;
    std::set<Tile> unique;
    for (const auto& p : points) {
      Tile tile;
      for (int j = 0; j < 3; ++j)
        tile[j] = int64_t(std::floor((p[j] - r.origin[j]) / size));
      tiles.push_back(tile);
      unique.insert(tile);
    }
    const size_t count = unique.size(), error = count > requested
                                                    ? count - requested
                                                    : requested - count;
    if (error < best || (error == best && size < r.size)) {
      best = error;
      r.size = size;
      r.tiles = std::move(tiles);
      best_count = count;
    }
    if (count > requested)
      low = size;
    else
      high = size;
  }
  r.report = {{"method", "occupied regions from robust extent and point count"},
              {"tile_size", r.size},
              {"origin", {r.origin.x(), r.origin.y(), r.origin.z()}},
              {"regions", best_count},
              {"requested_regions", requested},
              {"points", points.size()},
              {"frozen_in_initial_frame", true}};
  return r;
}
colmap::Sim3d RefineSimilarity(const Points& source, const Points& target,
                               colmap::Sim3d transform, const Regions& regions,
                               const std::vector<double>& voxels,
                               const std::vector<double>& distances,
                               int workers, Json& optimization,
                               std::unique_ptr<Surface>& surface) {
  Json log = Json::array();
  int minimum = 8;
  bool solved = false;
  for (size_t scale = 0; scale < voxels.size(); ++scale) {
    surface = Surfaces(target, voxels[scale], workers);
    const double maximum = distances.at(scale);
    for (int iteration = 0; iteration < 12; ++iteration) {
      Points points;
      for (const auto& p : source) points.push_back(transform * p);
      const auto residuals = Residuals(points, *surface, maximum);
      std::vector<size_t> keep;
      Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
      std::map<Tile, size_t> counts;
      for (size_t i = 0; i < points.size(); ++i)
        if (residuals[i].supported && std::abs(residuals[i].plane) <= maximum) {
          keep.push_back(i);
          pivot += points[i];
          ++counts[regions.tiles[i]];
        }
      if (keep.size() < 100) {
        log.push_back({{"voxel", voxels[scale]},
                       {"iteration", iteration},
                       {"status", "too_few_surface_matches"},
                       {"matches", keep.size()}});
        break;
      }
      pivot /= double(keep.size());
      const double cutoff = std::max(.015, std::min(.08, maximum / 4));
      Eigen::Matrix<double, 7, 7> information =
          Eigen::Matrix<double, 7, 7>::Zero();
      Eigen::Matrix<double, 7, 1> rhs = Eigen::Matrix<double, 7, 1>::Zero();
      double old_cost = 0.;
      for (size_t i : keep) {
        const auto& r = residuals[i];
        const auto& normal = surface->normals[r.nearest];
        const Eigen::Vector3d centred = points[i] - pivot;
        Eigen::Matrix<double, 7, 1> row;
        row.head<3>() = centred.cross(normal);
        row.segment<3>(3) = normal;
        row[6] = centred.dot(normal);
        const double base = 1. / counts.at(regions.tiles[i]),
                     weight = base *
                              std::min(1., cutoff / std::max(std::abs(r.plane),
                                                             1e-12));
        information += weight * row * row.transpose();
        rhs -= weight * row * r.plane;
        old_cost += base * Huber(r.plane, cutoff);
      }
      const Eigen::Matrix<double, 7, 1> norm =
          information.diagonal().cwiseMax(1e-24).cwiseSqrt();
      const Eigen::Matrix<double, 7, 7> normalized =
          norm.cwiseInverse().asDiagonal() * information *
          norm.cwiseInverse().asDiagonal();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>> eigen(
          normalized);
      if (eigen.info() != Eigen::Success)
        throw std::runtime_error("Similarity information eigensolve failed");
      const auto values = eigen.eigenvalues();
      int rank = 0;
      for (int j = 0; j < 7; ++j)
        if (values[j] > values[6] * 1e-6) ++rank;
      minimum = std::min(minimum, rank);
      solved = true;
      if (rank < 7) {
        log.push_back({{"voxel", voxels[scale]},
                       {"iteration", iteration},
                       {"status", "unobservable_similarity"},
                       {"matches", keep.size()},
                       {"rank", rank}});
        break;
      }
      Eigen::Matrix<double, 7, 1> delta = information.ldlt().solve(rhs);
      if (!delta.allFinite())
        throw std::runtime_error("Nonfinite similarity step");
      const double limit = std::max(
          {1., delta.head<3>().norm() / (5 * Pi / 180),
           delta.segment<3>(3).norm() / maximum, std::abs(delta[6]) / .03});
      delta /= limit;
      bool accepted = false;
      double new_cost = old_cost;
      Eigen::Matrix<double, 7, 1> step;
      for (int backtrack = 0; backtrack < 8; ++backtrack) {
        step = delta * std::pow(.5, backtrack);
        const double angle = step.head<3>().norm();
        const Eigen::Quaterniond rotation =
            angle > 0 ? Eigen::Quaterniond(
                            Eigen::AngleAxisd(angle, step.head<3>() / angle))
                      : Eigen::Quaterniond::Identity();
        const double s = std::exp(step[6]);
        colmap::Sim3d increment(
            s, rotation, pivot + step.segment<3>(3) - s * (rotation * pivot));
        const auto candidate = increment * transform;
        new_cost = 0.;
        for (size_t i : keep) {
          const auto& r = residuals[i];
          const double residual =
              ((candidate * source[i]) - surface->centres[r.nearest])
                  .dot(surface->normals[r.nearest]);
          new_cost += Huber(residual, cutoff) / counts.at(regions.tiles[i]);
        }
        if (new_cost <= old_cost) {
          transform = candidate;
          accepted = true;
          break;
        }
      }
      log.push_back(
          {{"voxel", voxels[scale]},
           {"iteration", iteration},
           {"status", accepted ? "updated" : "no_descent"},
           {"matches", keep.size()},
           {"rank", rank},
           {"weighted_cost_before", old_cost},
           {"weighted_cost_after", new_cost},
           {"step", std::vector<double>(step.data(), step.data() + 7)}});
      if (!accepted ||
          (step.head<3>().norm() < 1e-5 && step.segment<3>(3).norm() < 1e-4 &&
           std::abs(step[6]) < 1e-5))
        break;
    }
  }
  optimization = {
      {"iterations", log},
      {"full_rank_at_all_solved_iterations", solved && minimum == 7},
      {"minimum_rank", solved ? Json(minimum) : Json(nullptr)},
      {"pose_residuals_used_for_refinement", false}};
  return transform;
}
namespace {
Points Table(const fs::path& path, std::vector<double>& weeks,
             std::vector<double>& times) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Missing navigation file: " + path.u8string());
  Points p;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    std::istringstream row(line);
    double week, time, x, y, z, extra;
    if (!(row >> week >> time >> x >> y >> z))
      throw std::runtime_error("Navigation row requires at least five fields");
    while (row >> extra)
      if (!std::isfinite(extra))
        throw std::runtime_error("Nonfinite navigation field");
    if (!std::isfinite(week) || !std::isfinite(time) ||
        !Eigen::Vector3d(x, y, z).allFinite())
      throw std::runtime_error("Nonfinite navigation position");
    weeks.push_back(week);
    times.push_back(time);
    p.emplace_back(x, y, z);
  }
  return p;
}
std::pair<colmap::Sim3d, Json> Navigation(
    const Json& scene, const colmap::Reconstruction& rec,
    std::map<colmap::image_t, Eigen::Vector3d>& positions) {
  const auto root = fs::u8path(scene.at("input_root").get<std::string>());
  fs::path pose;
  for (const auto& e : fs::directory_iterator(root))
    if (e.path().filename().u8string().find("_CameraPos_C2E.txt") !=
        std::string::npos) {
      if (!pose.empty())
        throw std::runtime_error("Expected one camera-position file");
      pose = e.path();
    }
  if (pose.empty()) throw std::runtime_error("Missing camera-position file");
  const auto filename = pose.filename().u8string(),
             stem = filename.substr(
                 0, filename.size() - std::string("_CameraPos_C2E.txt").size());
  const auto nav_path = root / fs::u8path(stem + "_pos.txt"),
             enh_path = root / fs::u8path(stem + "_pos_ENH.txt");
  std::vector<double> weeks, times, enh_weeks, enh_times;
  const auto nav = Table(nav_path, weeks, times),
             enh = Table(enh_path, enh_weeks, enh_times);
  if (nav.size() != enh.size() || nav.size() < 3 || weeks != enh_weeks)
    throw std::runtime_error("Navigation row identities differ");
  std::vector<double> differences;
  Eigen::Vector3d centre = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < nav.size(); ++i) {
    if (weeks[i] != weeks[0] || (i && (!(times[i] > times[i - 1]) ||
                                       !(enh_times[i] > enh_times[i - 1]))))
      throw std::runtime_error("Invalid navigation time sequence");
    differences.push_back(times[i] - enh_times[i]);
    centre += nav[i];
  }
  if (*std::max_element(differences.begin(), differences.end()) -
          *std::min_element(differences.begin(), differences.end()) >
      .02)
    throw std::runtime_error("Navigation row time differences vary too much");
  centre /= double(nav.size());
  Points a, b;
  Eigen::Matrix3d spread = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < nav.size(); ++i) {
    a.push_back(nav[i] - centre);
    b.push_back(enh[i] - enh[0]);
    spread += a.back() * a.back().transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(spread);
  if (eigen.eigenvalues()[1] < eigen.eigenvalues()[2] * 1e-8)
    throw std::runtime_error("Navigation coordinates are collinear");
  colmap::Sim3d conversion;
  if (!colmap::EstimateSim3d(a, b, conversion) || conversion.scale() <= .999 ||
      conversion.scale() >= 1.001)
    throw std::runtime_error("Navigation coordinate conversion does not fit");
  conversion.translation() -=
      conversion.scale() * (conversion.rotation() * centre);
  std::vector<double> conversion_errors;
  for (size_t i = 0; i < nav.size(); ++i) {
    const double error = ((conversion * nav[i]) - (enh[i] - enh[0])).norm();
    if (error > .02)
      throw std::runtime_error(
          "Navigation coordinate conversion error exceeds limit");
    conversion_errors.push_back(error);
  }
  std::map<std::string, Json> rows;
  for (const auto& row : scene.at("images"))
    rows[row.at("name").get<std::string>()] = row;
  Points centres, target;
  Json names = Json::array();
  std::vector<double> exposure_offsets;
  auto ids = rec.RegImageIds();
  std::sort(ids.begin(), ids.end(), [&](auto x, auto y) {
    return rec.Image(x).Name() < rec.Image(y).Name();
  });
  for (auto id : ids) {
    const auto& image = rec.Image(id);
    const auto& row = rows.at(image.Name());
    const auto& prior = row.at("prior");
    if (prior.at("week") != row.at("week") ||
        prior.at("week").get<double>() != weeks[0])
      throw std::runtime_error("Camera/navigation GPS week mismatch");
    const auto raw = prior.at("center_raw").get<std::array<double, 3>>();
    const Eigen::Vector3d position =
        conversion * Eigen::Vector3d(raw[0], raw[1], raw[2]);
    positions[id] = position;
    centres.push_back(image.ProjectionCenter());
    target.push_back(position);
    names.push_back(image.Name());
    exposure_offsets.push_back(prior.at("seconds").get<double>() -
                               row.at("seconds").get<double>());
  }
  Eigen::Vector3d average = Eigen::Vector3d::Zero();
  for (const auto& p : centres) average += p;
  average /= double(centres.size());
  spread.setZero();
  for (const auto& p : centres) {
    const Eigen::Vector3d d = p - average;
    spread += d * d.transpose();
  }
  eigen.compute(spread);
  if (eigen.eigenvalues()[1] < std::max(eigen.eigenvalues()[2] * 1e-8, 1e-16))
    throw std::runtime_error("Camera trajectory is collinear");
  colmap::RANSACOptions opts;
  opts.max_error = .5;
  opts.random_seed = 0;
  opts.min_inlier_ratio = .25;
  colmap::Sim3d initial;
  const auto fit = colmap::EstimateSim3dRobust(centres, target, opts, initial);
  if (!fit.success || fit.support.num_inliers < 3)
    throw std::runtime_error(
        "Position correspondences do not support a similarity");
  std::vector<double> errors;
  for (size_t i = 0; i < centres.size(); ++i)
    errors.push_back(((initial * centres[i]) - target[i]).norm());
  return {
      initial,
      {{"kind", "vendor_navigation_frame_hypothesis"},
       {"hypothesis",
        "LAS XYZ approximately equals ENH minus first ENH navigation position"},
       {"vendor_coordinate_convention_verified", false},
       {"las_origin_verified", false},
       {"timestamps_modified", false},
       {"quaternions_used", false},
       {"navigation_to_candidate_las_matrix", Matrix(conversion)},
       {"enh_origin", {enh[0].x(), enh[0].y(), enh[0].z()}},
       {"coordinate_conversion_error", Statistics(conversion_errors)},
       {"coordinate_conversion_scale", conversion.scale()},
       {"camera_position_fit",
        {{"threshold", .5},
         {"inliers", fit.support.num_inliers},
         {"positions", centres.size()},
         {"position_error", Statistics(errors)}}},
       {"matched_image_names", names},
       {"navigation_row_time_difference_seconds", Statistics(differences)},
       {"camera_pose_minus_filename_seconds", Statistics(exposure_offsets)},
       {"input_files",
        {{pose.u8string(), FileMetadata(pose)},
         {nav_path.u8string(), FileMetadata(nav_path)},
         {enh_path.u8string(), FileMetadata(enh_path)}}}}};
}
Json Acceptance(const std::vector<Residual>& rows, const Regions& regions,
                double minimum_fraction, double max_median, double max_p90) {
  std::vector<double> plane, distance;
  std::map<Tile, std::vector<double>> tiles;
  std::set<Tile> all_tiles(regions.tiles.begin(), regions.tiles.end());
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (!std::isfinite(r.plane) || !std::isfinite(r.distance))
      throw std::runtime_error("Nonfinite registration residuals");
    distance.push_back(r.distance);
    if (r.supported) {
      plane.push_back(std::abs(r.plane));
      tiles[regions.tiles[i]].push_back(std::abs(r.plane));
    }
  }
  Json failures = Json::array(), regional = Json::array();
  const double fraction = double(plane.size()) / rows.size();
  if (plane.size() < 100 ||
      tiles.size() < std::min<size_t>(5, all_tiles.size()))
    failures.push_back(
        "Insufficient stable surface support or spatial coverage");
  if (fraction < minimum_fraction)
    failures.push_back("Stable surface support fraction below minimum");
  if (plane.empty() || Quantile(plane, .5) > max_median ||
      Quantile(plane, .9) > max_p90)
    failures.push_back("Supported plane error exceeds delivery limits");
  for (const auto& [tile, values] : tiles)
    if (values.size() >= 30) {
      auto stats = Statistics(values);
      stats["tile"] = tile;
      regional.push_back(stats);
      if (Quantile(values, .9) > .5)
        failures.push_back("Regional plane P90 exceeds limit");
    }
  return {{"passed", failures.empty()},
          {"failures", failures},
          {"fit_points", rows.size()},
          {"supported_fit_points", plane.size()},
          {"supported_fit_fraction", fraction},
          {"supported_fit_tiles", tiles.size()},
          {"point_to_plane_m", Statistics(plane)},
          {"nearest_distance", Statistics(distance)},
          {"regions", regional},
          {"limits",
           {{"support_fraction", minimum_fraction},
            {"median_m", max_median},
            {"p90_m", max_p90}}}};
}
template <typename T>
geosunpre::Array Array(const std::vector<T>& values, std::vector<size_t> shape,
                       const std::string& dtype) {
  geosunpre::Array a{dtype, std::move(shape), {}};
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  a.bytes.assign(bytes, bytes + values.size() * sizeof(T));
  return a;
}
std::vector<double> Flatten(const Points& points) {
  std::vector<double> values;
  values.reserve(points.size() * 3);
  for (const auto& p : points)
    values.insert(values.end(), {p.x(), p.y(), p.z()});
  return values;
}
void ProjectionInvariant(const colmap::Reconstruction& before,
                         const colmap::Reconstruction& after) {
  if (before.NumPoints3D() != after.NumPoints3D() ||
      before.ComputeNumObservations() != after.ComputeNumObservations())
    throw std::runtime_error("Similarity changed topology");
  for (const auto& [id, image] : before.Images()) {
    const auto& other = after.Image(id);
    const auto& camera = before.Camera(image.CameraId());
    if (camera.params != after.Camera(other.CameraId()).params)
      throw std::runtime_error("Similarity changed intrinsics");
    size_t checked = 0;
    for (const auto& point : image.Points2D())
      if (point.HasPoint3D() &&
          ++checked % std::max<size_t>(1, image.NumPoints3D() / 20) == 0) {
        const auto p = camera.ImgFromCam(image.CamFromWorld() *
                                         before.Point3D(point.point3D_id).xyz);
        const auto q = camera.ImgFromCam(other.CamFromWorld() *
                                         after.Point3D(point.point3D_id).xyz);
        if (!p || !q || !p->allFinite() || !q->allFinite() ||
            (*p - *q).norm() > 1e-7)
          throw std::runtime_error(
              "Similarity did not preserve visual projections");
      }
  }
}
}  // namespace
}  // namespace registration
void RunRegistration(const Arguments& args) {
  using namespace registration;
  args.Validate({"--input", "--manifest", "--las", "--output", "--workers",
                 "--minimum-support", "--maximum-median", "--maximum-p90",
                 "--maximum-median-px", "--maximum-p90-px", "--blas-threads",
                 "--joint"});
  const auto output = args.Path("--output"), model = args.Path("--input"),
             las = args.Path("--las");
  if (fs::exists(output))
    throw std::runtime_error("Registration output already exists");
  auto pending = output;
  pending += ".pending";
  if (fs::exists(pending))
    throw std::runtime_error("Unfinished registration exists");
  fs::create_directories(pending);
  const auto source_las = FileMetadata(las);
  Json source_model = Json::object();
  for (const char* name :
       {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"})
    source_model[name] = FileMetadata(model / name);
  colmap::Reconstruction rec;
  rec.Read(model);
  const auto scene = ReadJson(args.Path("--manifest"));
  const Json image_limits = {
      {"quality_max_median_px", args.Number("--maximum-median-px", 2.)},
      {"quality_max_p90_px", args.Number("--maximum-p90-px", 4.)}};
  VisualQuality(rec, scene, image_limits);
  std::map<colmap::image_t, Eigen::Vector3d> positions;
  const auto [initial, audit] = Navigation(scene, rec, positions);
  const auto header = ReadLasHeader(las);
  Json sampled;
  const auto target = SampleLas(las, header, sampled);
  Surface density;
  density.points = target;
  density.Index();
  std::vector<double> spacing;
  for (size_t i = 0; i < std::min<size_t>(4096, target.size()); ++i) {
    const size_t index = size_t(double(i) * (target.size() - 1) /
                                (std::min<size_t>(4096, target.size()) - 1));
    std::array<size_t, 8> ids;
    std::array<double, 8> squared;
    density.tree->knnSearch(target[index].data(), 8, ids.data(),
                            squared.data());
    for (double square : squared)
      if (square > 0) {
        spacing.push_back(std::sqrt(square));
        break;
      }
  }
  if (spacing.empty())
    throw std::runtime_error("LAS samples have no distinct support");
  const double fine = std::max(
      2 * header.scale.maxCoeff(),
      std::min(2 * Quantile(spacing, .5), args.Number("--maximum-median", .1)));
  const double coarse = std::max(8 * fine, 2 * audit.at("camera_position_fit")
                                                   .at("position_error")
                                                   .at("p90")
                                                   .get<double>());
  const std::vector<double> voxels{4 * fine, 2 * fine, fine},
      distances{coarse, std::max(4 * fine, coarse / 2), 2 * fine};
  const double support = args.Number("--maximum-p90", .3);
  std::vector<colmap::point3D_t> sorted;
  for (const auto& [id, p] : rec.Points3D()) sorted.push_back(id);
  std::sort(sorted.begin(), sorted.end());
  std::vector<colmap::point3D_t> ids;
  Points source, partition;
  for (auto id : sorted) {
    const auto& p = rec.Point3D(id);
    const Eigen::Vector3d transformed = initial * p.xyz;
    if (p.error <= 2. && p.track.Length() >= 4 &&
        ((transformed - header.lower).array() >= -coarse).all() &&
        ((header.upper - transformed).array() >= -coarse).all()) {
      ids.push_back(id);
      source.push_back(p.xyz);
      partition.push_back(transformed);
    }
  }
  if (source.size() < 100)
    throw std::runtime_error(
        "Insufficient eligible visual points in LAS overlap");
  const auto regions = Partition(partition);
  std::cerr << "Native registration: " << source.size() << " visual points, "
            << target.size() << " LAS samples\n";
  Json optimization;
  std::unique_ptr<Surface> surface;
  const int workers = args.Integer("--workers", 4);
  const auto transform =
      RefineSimilarity(source, target, initial, regions, voxels, distances,
                       workers, optimization, surface);
  Points final;
  for (const auto& p : source) final.push_back(transform * p);
  const auto residuals = Residuals(final, *surface, support);
  const auto acceptance =
      Acceptance(residuals, regions, args.Number("--minimum-support", .25),
                 args.Number("--maximum-median", .1), support);
  if (!acceptance.at("passed").get<bool>())
    throw std::runtime_error("LAS registration acceptance rejected: " +
                             acceptance.at("failures").dump());
  colmap::Reconstruction transformed = rec;
  auto selected_transform = transform;
  auto selected_acceptance = acceptance;
  auto selected_points = final;
  auto selected_residuals = residuals;
  Json joint = {{"status", "not_requested"}},
       selection = {{"selected_stage", "global_registration"}};
  if (args.Get("--joint") == "true") {
    auto candidate =
        RefineJoint(rec, transform, initial, positions, target, ids, regions,
                    fine, std::max(distances.back(), support), workers);
    joint = candidate.report;
    if (candidate.model) {
      Points candidate_points;
      for (auto id : ids)
        candidate_points.push_back(candidate.transform *
                                   candidate.model->Point3D(id).xyz);
      const auto next = Residuals(candidate_points, *surface, support);
      const auto candidate_acceptance =
          Acceptance(next, regions, args.Number("--minimum-support", .25),
                     args.Number("--maximum-median", .1), support);
      std::vector<double> old, newer;
      size_t before = 0, after = 0, common = 0;
      for (size_t i = 0; i < next.size(); ++i) {
        before += residuals[i].supported;
        after += next[i].supported;
        if (next[i].supported && residuals[i].supported) {
          ++common;
          old.push_back(std::abs(residuals[i].plane));
          newer.push_back(std::abs(next[i].plane));
        }
      }
      bool regression =
          double(after) / next.size() >= double(before) / next.size() - .02 &&
          common >= before * .9 &&
          Quantile(newer, .5) <= Quantile(old, .5) + .005 &&
          Quantile(newer, .9) <= Quantile(old, .9) + .02;
      for (const auto& tile :
           std::set<Tile>(regions.tiles.begin(), regions.tiles.end())) {
        old.clear();
        newer.clear();
        for (size_t i = 0; i < next.size(); ++i)
          if (regions.tiles[i] == tile && next[i].supported &&
              residuals[i].supported) {
            old.push_back(std::abs(residuals[i].plane));
            newer.push_back(std::abs(next[i].plane));
          }
        if (old.size() >= 30 &&
            (Quantile(newer, .5) > Quantile(old, .5) + .005 ||
             Quantile(newer, .9) > Quantile(old, .9) + .02))
          regression = false;
      }
      for (int orientation = 0; orientation < 2; ++orientation) {
        old.clear();
        newer.clear();
        for (size_t i = 0; i < next.size(); ++i) {
          const double z = std::abs(surface->normals[residuals[i].nearest].z());
          if ((orientation == 0
                   ? z >= std::cos(25 * 3.14159265358979323846 / 180)
                   : z <= std::sin(25 * 3.14159265358979323846 / 180)) &&
              next[i].supported && residuals[i].supported) {
            old.push_back(std::abs(residuals[i].plane));
            newer.push_back(std::abs(next[i].plane));
          }
        }
        if (old.size() >= 30 &&
            (Quantile(newer, .5) > Quantile(old, .5) + .005 ||
             Quantile(newer, .9) > Quantile(old, .9) + .02))
          regression = false;
      }
      bool visual_passed = true;
      try {
        joint["visual_quality"] =
            VisualQuality(*candidate.model, scene, image_limits);
      } catch (const VisualAcceptanceRejected& error) {
        visual_passed = false;
        joint["visual_rejection"] = error.what();
      }
      joint["surface_acceptance"] = candidate_acceptance;
      joint["geometric_regression_passed"] = regression;
      if (candidate_acceptance.at("passed").get<bool>() && regression &&
          visual_passed) {
        transformed = *candidate.model;
        selected_transform = candidate.transform;
        selected_acceptance = candidate_acceptance;
        selected_points = candidate_points;
        selected_residuals = next;
        selection["selected_stage"] = "pose_las";
      }
    }
  }
  const auto unaligned = transformed;
  transformed.Transform(selected_transform);
  fs::create_directories(pending / "sparse/0");
  transformed.Write(pending / "sparse/0");
  colmap::Reconstruction reread;
  reread.Read(pending / "sparse/0");
  if (!reread.IsValid() || Cheirality(reread).at("invalid_depth") != 0)
    throw std::runtime_error("Registered model has invalid depth or structure");
  ProjectionInvariant(unaligned, reread);
  Json scales = {{"nearest_spacing_median_m", Quantile(spacing, .5)},
                 {"coordinate_resolution_m", header.scale.maxCoeff()},
                 {"voxel_sizes", voxels},
                 {"correspondence_distances", distances},
                 {"support_distance", support}};
  Json report = {{"version", 1},
                 {"runtime", "C++"},
                 {"initialization", audit},
                 {"las", sampled},
                 {"regions", regions.report},
                 {"sampling_scales", scales},
                 {"optimization", optimization},
                 {"acceptance", selected_acceptance},
                 {"global_acceptance", acceptance},
                 {"joint", joint},
                 {"selection", selection},
                 {"scope",
                  "Fit-region delivery checks; no independent absolute "
                  "accuracy validation"}};
  WriteJson(pending / "report.json", report);
  WriteJson(pending / "quality.json",
            VisualQuality(unaligned, scene, image_limits));
  WriteJson(pending / "transform.json", {{"matrix", Matrix(selected_transform)},
                                         {"initial_matrix", Matrix(initial)}});
  std::vector<int64_t> tiles;
  std::vector<double> nearest, plane;
  std::vector<uint8_t> supported;
  Points normals;
  for (size_t i = 0; i < ids.size(); ++i) {
    tiles.insert(tiles.end(), regions.tiles[i].begin(), regions.tiles[i].end());
    nearest.push_back(selected_residuals[i].distance);
    plane.push_back(selected_residuals[i].plane);
    supported.push_back(selected_residuals[i].supported);
    normals.push_back(surface->normals[selected_residuals[i].nearest]);
  }
  geosunpre::WriteArrays(
      pending / "registration_points.npz",
      {{"point_ids", Array(ids, {ids.size()}, "<u8")},
       {"tiles", Array(tiles, {ids.size(), 3}, "<i8")},
       {"partition_xyz", Array(Flatten(partition), {ids.size(), 3}, "<f8")},
       {"final_xyz", Array(Flatten(selected_points), {ids.size(), 3}, "<f8")},
       {"nearest_distance", Array(nearest, {ids.size()}, "<f8")},
       {"signed_plane_error", Array(plane, {ids.size()}, "<f8")},
       {"stable_surface_supported", Array(supported, {ids.size()}, "|b1")},
       {"surface_normals", Array(Flatten(normals), {ids.size(), 3}, "<f8")}},
      true);
  if (source_las != FileMetadata(las))
    throw std::runtime_error("LAS changed during registration");
  for (const auto& [name, file] : source_model.items())
    if (FileMetadata(model / fs::u8path(name)) != file)
      throw std::runtime_error("Visual model changed during registration");
  for (const auto& [path, file] : audit.at("input_files").items())
    if (FileMetadata(fs::u8path(path)) != file)
      throw std::runtime_error("Navigation changed during registration");
  fs::rename(pending, output);
  std::cout << Json{{"selection", selection},
                    {"acceptance", selected_acceptance},
                    {"joint", joint}}
                   .dump()
            << '\n';
}

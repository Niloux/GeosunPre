#include "alignment/alignment.h"
#include "io/las.h"
#include "sfm/optimization.h"
#include <nanoflann.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <atomic>
#include <future>
#include <set>
#include <random>
#include <unordered_map>
namespace geosun {
namespace {
using Tile = std::array<int64_t, 3>;
using Points = std::vector<Eigen::Vector3d>;
struct Cloud {
  const Points* points;
  size_t kdtree_get_point_count() const { return points->size(); }
  double kdtree_get_pt(size_t i, size_t axis) const { return (*points)[i][axis]; }
  template <class Box>
  bool kdtree_get_bbox(Box&) const {
    return false;
  }
};
using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud,
                                                 3, size_t>;
struct Surface {
  Points points, centres, normals;
  std::vector<double> noise;
  std::vector<uint8_t> valid, coherent;
  Cloud cloud{&points};
  std::unique_ptr<Tree> tree;
  double voxel = 0.;
  void Index();
};
struct Regions {
  Eigen::Vector3d origin;
  double size;
  std::vector<Tile> tiles;
  Json report;
};
struct Residual {
  size_t nearest;
  double distance, plane;
  bool supported;
};
double Huber(double value, double cutoff) {
  const double a = std::abs(value);
  return a <= cutoff ? .5 * value * value : cutoff * (a - .5 * cutoff);
}
void Surface::Index() {
  tree = std::make_unique<Tree>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  tree->buildIndex();
}
std::unique_ptr<Surface> Surfaces(const Points& points, double voxel, int workers) {
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
  ParallelFor(size, workers, 1024, [&](size_t i) {
    std::array<size_t, 20> ids{};
    std::array<double, 20> squared{};
    surface->tree->knnSearch(surface->points[i].data(), k, ids.data(), squared.data());
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    for (size_t j = 0; j < k; ++j)
      centre += surface->points[ids[j]];
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
        values[0] / std::max(values.sum(), 1e-12) < .04 && values[1] > .04 * values[2] &&
        squared[k - 1] <= std::pow(std::max(.35, voxel * 5), 2) && surface->noise[i] <= .08;
  });
  return surface;
}
std::unique_ptr<Surface> CoherentSurfaces(const Points& points, double voxel, int workers) {
  auto fine = Surfaces(points, voxel, workers);
  auto coarse = Surfaces(points, 2 * voxel, workers);
  fine->coherent.resize(fine->points.size());
  ParallelFor(fine->points.size(), workers, 1024, [&](size_t i) {
    size_t id;
    double squared;
    coarse->tree->knnSearch(fine->points[i].data(), 1, &id, &squared);
    fine->coherent[i] =
        fine->valid[i] && coarse->valid[id] && fine->noise[i] <= .025 &&
        coarse->noise[id] <= .025 &&
        std::abs(fine->normals[i].dot(coarse->normals[id])) >= std::cos(15 * Pi / 180) &&
        std::abs((fine->centres[i] - coarse->centres[id]).dot(fine->normals[i])) <= .025;
  });
  return fine;
}
std::vector<Residual> Residuals(const Points& points, const Surface& surface, double maximum) {
  std::vector<Residual> result;
  result.reserve(points.size());
  for (const auto& p : points) {
    size_t id;
    double square;
    surface.tree->knnSearch(p.data(), 1, &id, &square);
    const double distance = std::sqrt(square),
                 plane = (p - surface.centres[id]).dot(surface.normals[id]);
    result.push_back({id, distance, plane, distance <= maximum && bool(surface.valid[id])});
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
    const size_t count = unique.size(),
                 error = count > requested ? count - requested : requested - count;
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
Similarity RefineSimilarity(const Points& source, const Points& target, Similarity transform,
                            const Regions& regions, const std::vector<double>& voxels,
                            const std::vector<double>& distances, int workers, Json& optimization,
                            std::unique_ptr<Surface>& surface) {
  Json log = Json::array();
  int minimum = 8;
  bool solved = false;
  for (size_t scale = 0; scale < voxels.size(); ++scale) {
    surface = Surfaces(target, voxels[scale], workers);
    const double maximum = distances.at(scale);
    for (int iteration = 0; iteration < 12; ++iteration) {
      Points points;
      for (const auto& p : source)
        points.push_back(transform * p);
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
      Eigen::Matrix<double, 7, 7> information = Eigen::Matrix<double, 7, 7>::Zero();
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
                     weight = base * std::min(1., cutoff / std::max(std::abs(r.plane), 1e-12));
        information += weight * row * row.transpose();
        rhs -= weight * row * r.plane;
        old_cost += base * Huber(r.plane, cutoff);
      }
      const Eigen::Matrix<double, 7, 1> norm = information.diagonal().cwiseMax(1e-24).cwiseSqrt();
      const Eigen::Matrix<double, 7, 7> normalized =
          norm.cwiseInverse().asDiagonal() * information * norm.cwiseInverse().asDiagonal();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>> eigen(normalized);
      if (eigen.info() != Eigen::Success)
        throw std::runtime_error("Similarity information eigensolve failed");
      const auto values = eigen.eigenvalues();
      int rank = 0;
      for (int j = 0; j < 7; ++j)
        if (values[j] > values[6] * 1e-6)
          ++rank;
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
      const double limit =
          std::max({1., delta.head<3>().norm() / (5 * Pi / 180),
                    delta.segment<3>(3).norm() / maximum, std::abs(delta[6]) / .03});
      delta /= limit;
      bool accepted = false;
      double new_cost = old_cost;
      Eigen::Matrix<double, 7, 1> step;
      for (int backtrack = 0; backtrack < 8; ++backtrack) {
        step = delta * std::pow(.5, backtrack);
        const double angle = step.head<3>().norm();
        const Eigen::Quaterniond rotation =
            angle > 0 ? Eigen::Quaterniond(Eigen::AngleAxisd(angle, step.head<3>() / angle))
                      : Eigen::Quaterniond::Identity();
        const double s = std::exp(step[6]);
        Similarity increment{s, rotation, pivot + step.segment<3>(3) - s * (rotation * pivot)};
        const auto candidate = increment * transform;
        new_cost = 0.;
        for (size_t i : keep) {
          const auto& r = residuals[i];
          const double residual = ((candidate * source[i]) - surface->centres[r.nearest])
                                      .dot(surface->normals[r.nearest]);
          new_cost += Huber(residual, cutoff) / counts.at(regions.tiles[i]);
        }
        if (new_cost <= old_cost) {
          transform = candidate;
          accepted = true;
          break;
        }
      }
      log.push_back({{"voxel", voxels[scale]},
                     {"iteration", iteration},
                     {"status", accepted ? "updated" : "no_descent"},
                     {"matches", keep.size()},
                     {"rank", rank},
                     {"weighted_cost_before", old_cost},
                     {"weighted_cost_after", new_cost},
                     {"step", std::vector<double>(step.data(), step.data() + 7)}});
      if (!accepted || (step.head<3>().norm() < 1e-5 && step.segment<3>(3).norm() < 1e-4 &&
                        std::abs(step[6]) < 1e-5))
        break;
    }
  }
  optimization = {{"iterations", log},
                  {"full_rank_at_all_solved_iterations", solved && minimum == 7},
                  {"minimum_rank", solved ? Json(minimum) : Json(nullptr)},
                  {"pose_residuals_used_for_refinement", false}};
  return transform;
}
Json Acceptance(const std::vector<Residual>& rows, const Regions& regions, double minimum_fraction,
                double max_median, double max_p90) {
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
  if (plane.size() < 100 || tiles.size() < std::min<size_t>(5, all_tiles.size()))
    failures.push_back("Insufficient stable surface support or spatial coverage");
  if (fraction < minimum_fraction)
    failures.push_back("Stable surface support fraction below minimum");
  if (plane.empty() || Quantile(plane, .5) > max_median || Quantile(plane, .9) > max_p90)
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
           {{"support_fraction", minimum_fraction}, {"median_m", max_median}, {"p90_m", max_p90}}}};
}

}  // namespace

namespace {
Similarity NavigationFit(const std::vector<Eigen::Vector3d>& centers,
                         const std::vector<Eigen::Vector3d>& positions) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> choose(0, centers.size() - 1);
  size_t best = 0;
  Similarity selected;
  for (int trial = 0; trial < 1000; ++trial) {
    const size_t a = choose(rng), b = choose(rng), c = choose(rng);
    if (a == b || a == c || b == c ||
        (centers[b] - centers[a]).cross(centers[c] - centers[a]).norm() < 1e-6)
      continue;
    const auto fit = FitSimilarity({centers[a], centers[b], centers[c]},
                                   {positions[a], positions[b], positions[c]});
    size_t support = 0;
    for (size_t i = 0; i < centers.size(); ++i)
      if ((fit.Apply(centers[i]) - positions[i]).norm() <= .5)
        ++support;
    if (support > best) {
      best = support;
      selected = fit;
    }
  }
  if (best < std::max<size_t>(3, centers.size() / 4))
    throw std::runtime_error("Navigation does not support a metric similarity");
  for (int pass = 0; pass < 3; ++pass) {
    Points a, b;
    for (size_t i = 0; i < centers.size(); ++i)
      if ((selected.Apply(centers[i]) - positions[i]).norm() <= .5) {
        a.push_back(centers[i]);
        b.push_back(positions[i]);
      }
    selected = FitSimilarity(a, b);
  }
  return selected;
}
double FixedCameraPositionSigma(const Reconstruction& model, const ModelPoint& point) {
  // Local uncertainty proxy with camera parameters fixed, not absolute accuracy.
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  for (const auto& o : point.track) {
    const auto& image = model.images[o.image];
    const Eigen::Matrix<double, 2, 3> j =
        model.cameras[image.camera].ProjectionJacobian(image.pose.Apply(point.xyz)) *
        image.pose.Rotation().toRotationMatrix();
    information.noalias() += j.transpose() * j;
  }
  const auto eigen =
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(information).eigenvalues().eval();
  return std::max(.5, point.error) / std::sqrt(std::max(eigen[0], 1e-18));
}
std::vector<double> LasSupportDistances(const Scene& scene, const Reconstruction& model,
                                        const LasHeader& header, double maximum,
                                        Json& report) {
  const auto started = Clock::now();
  const Eigen::Vector3d origin = (header.lower + header.upper) / 2;
  Json sampling;
  Surface sample;
  sample.points = SampleLas(scene.las, header, sampling);
  const double sample_seconds = Seconds(started);
  for (auto& point : sample.points)
    point -= origin;
  sample.Index();
  std::vector<double> distances(model.points.size(), INFINITY);
  std::vector<uint64_t> unresolved;
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    if (model.points[id].track.empty())
      continue;
    const Eigen::Vector3d local = model.points[id].xyz - origin;
    size_t nearest;
    double squared;
    sample.tree->knnSearch(local.data(), 1, &nearest, &squared);
    distances[id] = std::sqrt(squared);
    if (distances[id] > maximum)
      unresolved.push_back(id);
  }
  const size_t rechecked = unresolved.size();
  const double search_seconds = Seconds(started) - sample_seconds;
  // A missing neighbour in the bounded sample is not evidence of missing LAS
  // support. Recheck only those points against every valid record, one chunk
  // at a time, before removing them. No plane assumption is made here.
  if (!unresolved.empty()) {
    sample.tree.reset();
    sample.points.clear();
    // Cells as wide as the support distance: every record within it of a point lies in one of
    // the 27 cells around the point, where the point is listed, so one lookup per record finds
    // it. A point resolved by a chunk keeps that chunk's nearest record, as before.
    constexpr int64_t limit = int64_t(1) << 20;
    auto cell = [&](const Eigen::Vector3d& xyz, std::array<int64_t, 3>& index) {
      for (int k = 0; k < 3; ++k) {
        index[k] = int64_t(std::floor(xyz[k] / maximum));
        if (std::abs(index[k]) >= limit - 1)
          return false;
      }
      return true;
    };
    auto key = [&](const std::array<int64_t, 3>& index) {
      return uint64_t(index[0] + limit) << 42 | uint64_t(index[1] + limit) << 21 |
             uint64_t(index[2] + limit);
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
    std::vector<Eigen::Vector3d> local(unresolved.size());
    for (uint32_t i = 0; i < unresolved.size(); ++i) {
      local[i] = model.points[unresolved[i]].xyz - origin;
      std::array<int64_t, 3> centre;
      // Beyond the grid a point is far outside the LAS bounds and stays unsupported.
      if (!cell(local[i], centre))
        continue;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            grid[key({centre[0] + dx, centre[1] + dy, centre[2] + dz})].push_back(i);
    }
    std::vector<uint8_t> resolved(unresolved.size());
    std::vector<double> nearest(unresolved.size());
    ScanLas(scene.las, header, [&](uint64_t, const std::vector<LasPoint>& records) {
      std::fill(nearest.begin(), nearest.end(), INFINITY);
      for (const auto& record : records) {
        if (!KeepLasPoint(record))
          continue;
        const Eigen::Vector3d xyz = LasCoordinates(header, record) - origin;
        std::array<int64_t, 3> index;
        if (!cell(xyz, index))
          continue;
        const auto found = grid.find(key(index));
        if (found == grid.end())
          continue;
        for (const uint32_t i : found->second)
          if (!resolved[i])
            nearest[i] = std::min(nearest[i], (xyz - local[i]).squaredNorm());
      }
      for (uint32_t i = 0; i < unresolved.size(); ++i)
        if (!resolved[i] && nearest[i] <= maximum * maximum) {
          distances[unresolved[i]] = std::sqrt(nearest[i]);
          resolved[i] = 1;
        }
    });
    std::vector<uint64_t> remaining;
    for (uint32_t i = 0; i < unresolved.size(); ++i)
      if (!resolved[i])
        remaining.push_back(unresolved[i]);
    unresolved = std::move(remaining);
  }
  report = {{"sample", sampling},
            {"rechecked_against_full_las", rechecked},
            {"rescued_from_sample_gaps", rechecked - unresolved.size()},
            {"unsupported_after_full_scan", unresolved.size()},
            {"distance_scope",
             "distance to a searched valid LAS record; upper bound on full-cloud nearest distance"},
            {"sample_seconds", sample_seconds},
            {"sample_search_seconds", search_seconds},
            {"recheck_seconds", Seconds(started) - sample_seconds - search_seconds},
            {"seconds", Seconds(started)}};
  return distances;
}
}  // namespace
Json PlaneErrors(const Reconstruction& model, const std::vector<PlaneConstraint>& planes) {
  std::vector<double> errors;
  for (const auto& plane : planes)
    if (!model.points[plane.point].track.empty())
      errors.push_back(std::abs((model.points[plane.point].xyz - plane.center).dot(plane.normal)));
  return Statistics(errors);
}
Json FailedChecks(const Json& checks) {
  Json failed = Json::array();
  for (const auto& [name, passed] : checks.items())
    if (!passed.get<bool>())
      failed.push_back(name);
  return failed;
}
std::vector<PlaneConstraint> LasPlaneConstraints(const Scene& scene, const Reconstruction& model,
                                                 int workers) {
  const auto header = ReadLasHeader(scene.las);
  const Eigen::Vector3d origin = (header.lower + header.upper) / 2;
  Json sampling;
  auto target = SampleLas(scene.las, header, sampling);
  for (auto& p : target)
    p -= origin;
  auto surface = CoherentSurfaces(target, .025, workers);
  std::vector<PlaneConstraint> planes;
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& p = model.points[id];
    if (p.track.size() < 3 || p.error > 2)
      continue;
    const Eigen::Vector3d local = p.xyz - origin;
    size_t nearest;
    double squared;
    surface->tree->knnSearch(local.data(), 1, &nearest, &squared);
    if (squared > .01 || !surface->coherent[nearest])
      continue;
    planes.push_back({id, surface->centres[nearest] + origin, surface->normals[nearest],
                      std::max({.01, surface->noise[nearest], header.scale.maxCoeff()})});
  }
  return planes;
}
Json FilterSparseStructure(const Scene& scene, Reconstruction& model,
                           const AlignmentLimits& limits) {
  // Camera refinement requires >=3 views; reliable two-view points can still
  // seed 3DGS when parallax, depth precision and LAS support pass.
  constexpr size_t minimum_track_length = 2;
  constexpr double minimum_parallax_degrees = 1.5;
  const auto header = ReadLasHeader(scene.las);
  const double max_error = std::min(3., limits.p90_px),
               max_sigma = std::max(.5 * limits.median_m, 3 * header.scale.maxCoeff()),
               max_distance = std::max(limits.median_m, 3 * header.scale.maxCoeff());
  size_t before = 0, short_tracks = 0, uncertain = 0, unsupported = 0, retained = 0;
  for (const auto& p : model.points)
    before += !p.track.empty();
  model.Filter(max_error, minimum_parallax_degrees);
  for (auto& point : model.points) {
    if (point.track.empty())
      continue;
    if (point.track.size() < minimum_track_length) {
      point.track.clear();
      ++short_tracks;
      continue;
    }
    const double sigma = FixedCameraPositionSigma(model, point);
    if (sigma > max_sigma) {
      point.track.clear();
      ++uncertain;
      continue;
    }
  }
  Json support;
  auto support_distances =
      LasSupportDistances(scene, model, header, max_distance, support);
  Json support_passes = Json::array({support});
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    auto& point = model.points[id];
    if (point.track.empty())
      continue;
    if (support_distances[id] > max_distance) {
      point.track.clear();
      ++unsupported;
      continue;
    }
    ++retained;
  }
  const size_t initially_retained = retained;
  Json removed_images = Json::array();
  for (;;) {
    model.RebuildReferences();
    bool changed = false;
    for (uint32_t id = 0; id < model.images.size(); ++id) {
      auto& image = model.images[id];
      if (!image.registered)
        continue;
      size_t count = 0;
      std::set<int> cells;
      const auto& camera = model.cameras[image.camera];
      for (size_t feature = 0; feature < image.points.size(); ++feature)
        if (image.points[feature] != NoPoint) {
          ++count;
          cells.insert(camera.CoverageCell(image.pixels[feature]));
        }
      if (count < MinimumImageObservations || cells.size() < MinimumImageCells) {
        image.registered = false;
        changed = true;
        removed_images.push_back({{"image_index", id},
                                  {"name", scene.images[id].name},
                                  {"observations", count},
                                  {"occupied_cells", cells.size()}});
      }
    }
    if (!changed)
      break;
    for (auto& point : model.points) {
      const size_t before_track = point.track.size();
      point.track.erase(
          std::remove_if(point.track.begin(), point.track.end(),
                         [&](const auto& o) { return !model.images[o.image].registered; }),
          point.track.end());
      if (point.track.size() < minimum_track_length)
        point.track.clear();
      else if (point.track.size() != before_track)
        RefinePoint(model, point.track, point.xyz);
    }
    model.Filter(max_error, minimum_parallax_degrees);
  }
  std::vector<double> distances, uncertainty;
  retained = 0;
  if (!removed_images.empty()) {
    support_distances = LasSupportDistances(scene, model, header, max_distance, support);
    support_passes.push_back(support);
  }
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    auto& point = model.points[id];
    if (point.track.empty())
      continue;
    const double sigma = FixedCameraPositionSigma(model, point);
    if (point.track.size() < minimum_track_length || sigma > max_sigma ||
        support_distances[id] > max_distance) {
      point.track.clear();
      continue;
    }
    ++retained;
    distances.push_back(support_distances[id]);
    uncertainty.push_back(sigma);
  }
  model.RebuildReferences();
  return {{"points_before", before},
          {"points_after", retained},
          {"removed_reprojection_or_parallax",
           before - initially_retained - short_tracks - uncertain - unsupported},
          {"removed_short_tracks", short_tracks},
          {"removed_uncertain_depth", uncertain},
          {"removed_las_unsupported", unsupported},
          {"removed_with_weak_images", initially_retained - retained},
          {"removed_images", removed_images},
          {"nearest_las_m", Statistics(distances)},
          {"las_support_passes", support_passes},
          {"fixed_camera_position_sigma_proxy_m", Statistics(uncertainty)},
          {"limits",
           {{"minimum_track_length", minimum_track_length},
            {"minimum_parallax_degrees", minimum_parallax_degrees},
            {"maximum_reprojection_px", max_error},
            {"maximum_sigma_proxy_m", max_sigma},
            {"maximum_nearest_las_m", max_distance}}}};
}
Json Align(const Scene& scene, Reconstruction& model, int workers, const AlignmentLimits& limits,
           int ba_workers) {
  if (!ba_workers)
    ba_workers = workers;
  const auto header = ReadLasHeader(scene.las);
  Json sampling;
  const Eigen::Vector3d origin = (header.lower + header.upper) / 2;
  auto target = SampleLas(scene.las, header, sampling);
  for (auto& p : target)
    p -= origin;
  const auto navigation = NavigationPositions(scene);
  Points centers, positions;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered) {
      centers.push_back(model.images[i].pose.Center());
      positions.push_back(navigation[i] - origin);
    }
  const auto initial = NavigationFit(centers, positions);
  std::vector<double> nav_errors;
  for (size_t i = 0; i < centers.size(); ++i)
    nav_errors.push_back((initial.Apply(centers[i]) - positions[i]).norm());
  Surface density;
  density.points = target;
  density.Index();
  std::vector<double> spacing;
  for (size_t i = 0; i < std::min<size_t>(4096, target.size()); ++i) {
    const size_t index = i * (target.size() - 1) / (std::min<size_t>(4096, target.size()) - 1);
    std::array<size_t, 8> ids;
    std::array<double, 8> squared;
    density.tree->knnSearch(target[index].data(), 8, ids.data(), squared.data());
    for (double d : squared)
      if (d > 0) {
        spacing.push_back(std::sqrt(d));
        break;
      }
  }
  if (spacing.empty())
    throw std::runtime_error("LAS samples have no distinct support");
  const double fine =
      std::max(2 * header.scale.maxCoeff(), std::min(2 * Quantile(spacing, .5), limits.median_m));
  const double coarse = std::max(8 * fine, 2 * Quantile(nav_errors, .9));
  const std::vector<double> voxels{4 * fine, 2 * fine, fine},
      distances{coarse, std::max(4 * fine, coarse / 2), 2 * fine};
  Points source, partition;
  std::vector<uint64_t> identities;
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& p = model.points[id];
    if (p.track.size() < 4 || p.error > 2)
      continue;
    const auto v = initial.Apply(p.xyz);
    if (((v - (header.lower - origin)).array() >= -coarse).all() &&
        (((header.upper - origin) - v).array() >= -coarse).all()) {
      source.push_back(p.xyz);
      partition.push_back(v);
      identities.push_back(id);
    }
  }
  if (source.size() < 100)
    throw std::runtime_error("Insufficient visual points in LAS overlap");
  Log("LAS alignment: " + std::to_string(source.size()) + " visual points, " +
      std::to_string(target.size()) + " samples");
  const auto regions = Partition(partition);
  Json optimization;
  std::unique_ptr<Surface> surface;
  auto transform = RefineSimilarity(source, target, initial, regions, voxels, distances, workers,
                                    optimization, surface);
  if (!optimization.at("full_rank_at_all_solved_iterations").get<bool>())
    throw std::runtime_error("LAS geometry cannot constrain all seven similarity parameters");
  Points aligned;
  for (const auto& p : source)
    aligned.push_back(transform.Apply(p));
  auto acceptance = Acceptance(Residuals(aligned, *surface, limits.p90_m), regions,
                               limits.support_fraction, limits.median_m, limits.p90_m);
  if (!acceptance.at("passed").get<bool>())
    throw std::runtime_error("LAS alignment rejected: " + acceptance.at("failures").dump());
  transform.translation += origin;
  model.Apply(transform);
  const auto base_quality = model.Quality(scene, limits.median_px, limits.p90_px);
  if (!base_quality.at("passed").get<bool>())
    throw std::runtime_error("Aligned visual model failed acceptance");
  auto reliable = CoherentSurfaces(target, fine, workers);
  const auto residuals = Residuals(aligned, *reliable, limits.p90_m);
  std::vector<PlaneConstraint> constraints;
  std::vector<double> normalized;
  Eigen::Matrix3d directions = Eigen::Matrix3d::Zero();
  std::map<Tile, size_t> regional_counts;
  for (size_t i = 0; i < residuals.size(); ++i)
    if (residuals[i].supported && reliable->coherent[residuals[i].nearest])
      ++regional_counts[regions.tiles[i]];
  for (size_t i = 0; i < residuals.size(); ++i) {
    const auto& r = residuals[i];
    if (!r.supported || !reliable->coherent[r.nearest])
      continue;
    const auto& normal = reliable->normals[r.nearest];
    const double sigma = std::max({.01, reliable->noise[r.nearest], header.scale.maxCoeff()});
    normalized.push_back(std::abs(r.plane) / sigma);
    directions += normal * normal.transpose();
    const double balance = std::sqrt(
        double(regional_counts[regions.tiles[i]]) /
        std::max<size_t>(1, residuals.size() / std::max<size_t>(1, regional_counts.size())));
    constraints.push_back({identities[i], reliable->centres[r.nearest] + origin, normal,
                           sigma * std::max(.5, balance)});
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(directions);
  Json joint = {{"used", false},
                {"reliable_correspondences", constraints.size()},
                {"normalized_plane_error", Statistics(normalized)}};
  const bool directional =
      constraints.size() >= 200 && eigen.eigenvalues()[0] > 1e-4 * eigen.eigenvalues()[2];
  if (!directional)
    joint["reason"] = "insufficient reliable surface directions";
  else if (Quantile(normalized, .5) <= 1 && Quantile(normalized, .9) <= 2)
    joint["reason"] = "already within surface noise proxy";
  else {
    Reconstruction candidate = model;
    const auto solve =
        Optimize(candidate, scene.cameras, "fixed", ba_workers, 100, {}, constraints, navigation);
    const auto quality = candidate.Quality(scene, limits.median_px, limits.p90_px);
    Points candidate_points;
    std::vector<double> displacement;
    for (uint64_t id : identities) {
      candidate_points.push_back(candidate.points[id].xyz - origin);
      displacement.push_back((candidate.points[id].xyz - model.points[id].xyz).norm());
    }
    const auto proposed = Acceptance(Residuals(candidate_points, *surface, limits.p90_m), regions,
                                     limits.support_fraction, limits.median_m, limits.p90_m);
    const double before = base_quality.at("reprojection_px").at("p90"),
                 after = quality.at("reprojection_px").at("p90");
    bool image_stable = true;
    Json unstable_images = Json::array();
    for (size_t i = 0; i < quality.at("images").size(); ++i) {
      if (!model.images[i].registered)
        continue;
      const auto& previous = base_quality.at("images")[i];
      const auto& next = quality.at("images")[i];
      const double a = previous.at("reprojection_px").at("p90"),
                   b = next.at("reprojection_px").at("p90");
      if (b > a * 1.1 + .1) {
        image_stable = false;
        unstable_images.push_back({{"name", next.at("name")},
                                   {"before_p90_px", a},
                                   {"after_p90_px", b},
                                   {"limit_px", a * 1.1 + .1}});
      }
    }
    const bool improved = proposed.at("point_to_plane_m").at("median").get<double>() <
                          acceptance.at("point_to_plane_m").at("median").get<double>();
    const Json checks = {{"visual_acceptance", quality.at("passed")},
                         {"alignment_acceptance", proposed.at("passed")},
                         {"global_reprojection", after <= before * 1.05 + .05},
                         {"per_image_reprojection", image_stable},
                         {"point_displacement", Quantile(displacement, .9) <= .1},
                         {"plane_improvement", improved}};
    const auto failed_checks = FailedChecks(checks);
    const bool accepted = failed_checks.empty();
    joint["used"] = accepted;
    joint["reason"] =
        accepted ? "accepted" : "candidate did not preserve visual fit or improve LAS fit";
    joint["optimization"] = solve;
    joint["candidate_alignment"] = proposed;
    joint["candidate_quality"] = quality;
    joint["checks"] = checks;
    joint["failed_checks"] = failed_checks;
    joint["unstable_images"] = unstable_images;
    joint["point_displacement_m"] = Statistics(displacement);
    joint["comparison"] = {{"before_reprojection_p90_px", before},
                           {"after_reprojection_p90_px", after},
                           {"reprojection_limit_px", before * 1.05 + .05},
                           {"point_displacement_p90_limit_m", .1}};
    if (accepted) {
      model = std::move(candidate);
      acceptance = proposed;
    }
  }
  for (auto& point : model.points)
    if (!point.track.empty()) {
      point.error = 0;
      for (const auto& observation : point.track) {
        const auto& image = model.images[observation.image];
        point.error += (model.cameras[image.camera].Project(image.pose.Apply(point.xyz)) -
                        image.pixels[observation.feature])
                           .norm();
      }
      point.error /= point.track.size();
    }
  nav_errors.clear();
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered)
      nav_errors.push_back((model.images[i].pose.Center() - navigation[i]).norm());
  return {
      {"coordinate_frame", "input_las_xyz"},
      {"navigation_frame_assumption", "ENH minus first ENH position; vendor convention unverified"},
      {"navigation_error_m", Statistics(nav_errors)},
      {"transform", transform.ToJson()},
      {"local_origin", {origin.x(), origin.y(), origin.z()}},
      {"sampling", sampling},
      {"regions", regions.report},
      {"similarity_rank", optimization.at("minimum_rank")},
      {"alignment", acceptance},
      {"joint", joint},
      {"quality", model.Quality(scene, limits.median_px, limits.p90_px)}};
}
}  // namespace geosun

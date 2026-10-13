#include "sfm/photometric.h"
#include "sfm/optimization.h"
#include "alignment/alignment.h"
#include "io/cache.h"
#include "io/colmap.h"
#include "frontend/frontend.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <atomic>
#include <future>
#include <iostream>
#include <numeric>
#include <deque>
#include <map>

namespace geosun {
namespace {
constexpr int Radius = 12, Search = 6;
bool Contains(const fs::path& parent, const fs::path& child) {
  const auto p = fs::absolute(parent).lexically_normal(), c = fs::absolute(child).lexically_normal();
  auto a = p.begin(), b = c.begin();
  for (; a != p.end(); ++a, ++b) {
    if (b == c.end())
      return false;
#ifdef _WIN32
    if (_wcsicmp(a->c_str(), b->c_str()))
      return false;
#else
    if (*a != *b)
      return false;
#endif
  }
  return true;
}
template <class Function>
void Parallel(size_t count, int workers, Function function) {
  std::atomic<size_t> next{0};
  std::vector<std::future<void>> jobs;
  for (int i = 0; i < workers; ++i)
    jobs.push_back(std::async(std::launch::async, [&] {
      for (;;) {
        const auto id = next.fetch_add(1);
        if (id >= count)
          break;
        function(id);
      }
    }));
  for (auto& job : jobs)
    job.get();
}
// Coordinates use the project's half-pixel convention. Sample the original
// grayscale image in double precision, without quantizing the warp to 1/32 px.
cv::Mat Sample(const cv::Mat& image, const Eigen::Vector2d& center,
               const Eigen::Matrix2d& affine, int radius) {
  cv::Mat patch(2 * radius + 1, 2 * radius + 1, CV_32FC1);
  for (int y = -radius; y <= radius; ++y)
    for (int x = -radius; x <= radius; ++x) {
      const Eigen::Vector2d p = center + affine * Eigen::Vector2d(x, y) -
                                Eigen::Vector2d::Constant(.5);
      const int ix = int(std::floor(p.x())), iy = int(std::floor(p.y()));
      if (ix < 0 || iy < 0 || ix + 1 >= image.cols || iy + 1 >= image.rows)
        return {};
      const double dx = p.x() - ix, dy = p.y() - iy;
      patch.at<float>(y + radius, x + radius) = float(
          (1 - dy) * ((1 - dx) * image.at<uint8_t>(iy, ix) + dx * image.at<uint8_t>(iy, ix + 1)) +
          dy * ((1 - dx) * image.at<uint8_t>(iy + 1, ix) + dx * image.at<uint8_t>(iy + 1, ix + 1)));
    }
  return patch;
}
cv::Mat SmoothedSample(const cv::Mat& image, const Eigen::Vector2d& center,
                       const Eigen::Matrix2d& affine, int radius) {
  auto patch = Sample(image, center, affine, radius);
  if (!patch.empty())
    cv::GaussianBlur(patch, patch, cv::Size(5, 5), .7);
  return patch;
}
bool Textured(const cv::Mat& patch) {
  if (patch.empty())
    return false;
  cv::Scalar mean, deviation;
  cv::meanStdDev(patch, mean, deviation);
  if (deviation[0] < 8)
    return false;
  Eigen::Matrix2d gradients = Eigen::Matrix2d::Zero();
  for (int y = 1; y + 1 < patch.rows; ++y)
    for (int x = 1; x + 1 < patch.cols; ++x) {
      const Eigen::Vector2d g((patch.at<float>(y, x + 1) - patch.at<float>(y, x - 1)) / 2,
                              (patch.at<float>(y + 1, x) - patch.at<float>(y - 1, x)) / 2);
      gradients.noalias() += g * g.transpose();
    }
  const auto values = Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(gradients).eigenvalues().eval();
  return values[0] / ((patch.rows - 2) * (patch.cols - 2)) > 6 && values[0] > .02 * values[1];
}
bool Warp(const Reconstruction& model, const Eigen::Vector3d& xyz, const Observation& a,
          const Observation& b, const Eigen::Vector3d& plane, Eigen::Matrix2d& affine) {
  const auto& ia = model.images[a.image];
  const auto& ib = model.images[b.image];
  const Eigen::Vector3d va = (xyz - ia.pose.Center()).normalized(),
                        vb = (xyz - ib.pose.Center()).normalized();
  if (va.dot(vb) < .866025403784)
    return false;
  Eigen::Vector3d normal = plane.squaredNorm() > .5 ? plane : (va + vb).normalized();
  if (std::abs(normal.dot(va)) < .3 || std::abs(normal.dot(vb)) < .3)
    return false;
  const Eigen::Vector3d u = normal.unitOrthogonal(), v = normal.cross(u);
  Eigen::Matrix<double, 3, 2> basis;
  basis << u, v;
  const Eigen::Matrix2d ja = model.cameras[ia.camera].ProjectionJacobian(ia.pose.Apply(xyz)) *
                            ia.pose.Rotation().toRotationMatrix() * basis,
                        jb = model.cameras[ib.camera].ProjectionJacobian(ib.pose.Apply(xyz)) *
                            ib.pose.Rotation().toRotationMatrix() * basis;
  if (std::abs(ja.determinant()) < 1e-6)
    return false;
  affine = jb * ja.inverse();
  const auto scales = Eigen::JacobiSVD<Eigen::Matrix2d>(affine).singularValues().eval();
  return affine.allFinite() && affine.determinant() > 0 && scales[0] < 1.8 && scales[1] > .55 &&
         scales[0] < 2.2 * scales[1];
}
bool MatchPatch(const cv::Mat& raw_reference, const cv::Mat& image, const Eigen::Vector2d& pixel,
           const Eigen::Matrix2d& affine, Eigen::Vector2d& refined) {
  cv::Mat reference;
  cv::GaussianBlur(raw_reference, reference, cv::Size(5, 5), .7);
  const auto search = SmoothedSample(image, pixel, affine, Radius + Search);
  if (search.empty())
    return false;
  cv::Mat scores;
  cv::matchTemplate(search, reference, scores, cv::TM_CCOEFF_NORMED);
  double score;
  cv::Point peak;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &peak);
  if (score < .94 || peak.x == 0 || peak.y == 0 || peak.x + 1 == scores.cols ||
      peak.y + 1 == scores.rows)
    return false;
  // Repeated text and straight edges must have an unambiguous correlation peak.
  for (int y = 0; y < scores.rows; ++y)
    for (int x = 0; x < scores.cols; ++x)
      if (std::abs(x - peak.x) > 2 || std::abs(y - peak.y) > 2)
        if (scores.at<float>(y, x) > score - .012)
          return false;
  Eigen::Vector2d delta(peak.x - Search, peak.y - Search);
  for (int axis = 0; axis < 2; ++axis) {
    const double low = scores.at<float>(peak.y - (axis == 1), peak.x - (axis == 0)),
                 high = scores.at<float>(peak.y + (axis == 1), peak.x + (axis == 0));
    const double curvature = low + high - 2 * score;
    if (curvature > -1e-5)
      return false;
    delta[axis] += (low - high) / (2 * curvature);
  }
  // Refine the correlation peak against continuous image samples, jointly
  // fitting exposure gain and offset instead of rounding to a pixel grid.
  auto target = SmoothedSample(image, pixel + affine * delta, affine, Radius);
  cv::Scalar reference_mean, reference_sd, target_mean, target_sd;
  cv::meanStdDev(reference, reference_mean, reference_sd);
  cv::meanStdDev(target, target_mean, target_sd);
  double gain = target_sd[0] / reference_sd[0], offset = target_mean[0] - gain * reference_mean[0];
  for (int iteration = 0; iteration < 6; ++iteration) {
    Eigen::Matrix4d information = Eigen::Matrix4d::Zero();
    Eigen::Vector4d rhs = Eigen::Vector4d::Zero();
    const auto center = pixel + affine * delta;
    target = SmoothedSample(image, center, affine, Radius);
    const auto left = SmoothedSample(image, center - Eigen::Vector2d::UnitX(), affine, Radius),
               right = SmoothedSample(image, center + Eigen::Vector2d::UnitX(), affine, Radius),
               up = SmoothedSample(image, center - Eigen::Vector2d::UnitY(), affine, Radius),
               down = SmoothedSample(image, center + Eigen::Vector2d::UnitY(), affine, Radius);
    if (target.empty() || left.empty() || right.empty() || up.empty() || down.empty())
      return false;
    for (int y = 0; y < reference.rows; ++y)
      for (int x = 0; x < reference.cols; ++x) {
        const double source = reference.at<float>(y, x);
        const double residual = target.at<float>(y, x) - gain * source - offset;
        const Eigen::Vector2d gradient((right.at<float>(y, x) - left.at<float>(y, x)) / 2,
                                        (down.at<float>(y, x) - up.at<float>(y, x)) / 2);
        Eigen::Vector4d j;
        j << gradient.dot(affine.col(0)), gradient.dot(affine.col(1)), -source, -1;
        const double weight = std::min(1., 12 / std::max(std::abs(residual), 1e-9));
        information.noalias() += weight * j * j.transpose();
        rhs.noalias() -= weight * j * residual;
      }
    const Eigen::Vector4d step = information.ldlt().solve(rhs);
    if (!step.allFinite() || step.head<2>().norm() > 1.5)
      return false;
    delta += step.head<2>();
    gain += step[2];
    offset += step[3];
    if (gain < .4 || gain > 2.5)
      return false;
    if (step.head<2>().norm() < .005)
      break;
  }
  const Eigen::Vector2d shift = affine * delta;
  if (!shift.allFinite() || shift.norm() > 4)
    return false;
  refined = pixel + shift;
  return true;
}
Json PointErrors(Reconstruction& model, const std::set<uint64_t>& ids, bool refit = true) {
  std::vector<double> errors;
  for (auto id : ids) {
    auto& point = model.points[id];
    if (refit)
      RefinePoint(model, point.track, point.xyz);
    for (const auto& o : point.track) {
      const auto& image = model.images[o.image];
      const auto v = image.pose.Apply(point.xyz);
      if (v.z() <= 0)
        throw std::runtime_error("Photometric refinement produced nonpositive depth");
      errors.push_back((model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm());
    }
  }
  return Statistics(errors);
}
Json PlaneErrors(const Reconstruction& model, const std::vector<PlaneConstraint>& planes) {
  std::vector<double> errors;
  for (const auto& plane : planes)
    if (!model.points[plane.point].track.empty())
      errors.push_back(std::abs((model.points[plane.point].xyz - plane.center).dot(plane.normal)));
  return Statistics(errors);
}
class GrayCache {
  const Scene& scene;
  std::map<uint32_t, cv::Mat> images;
  std::deque<uint32_t> order;

 public:
  explicit GrayCache(const Scene& source) : scene(source) {}
  cv::Mat Get(uint32_t id) {
    const auto found = images.find(id);
    if (found != images.end()) {
      order.erase(std::find(order.begin(), order.end(), id));
      order.push_back(id);
      return found->second;
    }
    if (images.size() == 12) {
      images.erase(order.front());
      order.pop_front();
    }
    auto gray = ReadImage(scene.images[id].source, cv::IMREAD_GRAYSCALE);
    images.emplace(id, gray);
    order.push_back(id);
    return gray;
  }
};
std::vector<std::vector<Observation>> RefineObservationGraph(
    const Scene& scene, Reconstruction& model, const std::vector<Eigen::Vector3d>& normals,
    int workers, Json& report) {
  struct Edge {
    uint64_t point;
    uint32_t a, b;
    Eigen::Matrix2d affine;
    Eigen::Vector2d shift;
    double round_trip = 0;
    bool accepted = false;
  };
  std::vector<Edge> edges;
  std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> pairs;
  std::vector<std::vector<size_t>> by_point(model.points.size());
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& point = model.points[id];
    if (point.track.size() < 3 || point.error > 2)
      continue;
    std::vector<uint32_t> sequence(point.track.size());
    std::iota(sequence.begin(), sequence.end(), 0);
    std::sort(sequence.begin(), sequence.end(), [&](uint32_t a, uint32_t b) {
      return point.track[a].image < point.track[b].image;
    });
    // Consecutive views plus the preceding second view create local triangles.
    // Long tracks can propagate accurate correspondences without a distant,
    // fixed reference patch or loading all original images into memory.
    for (size_t b = 1; b < sequence.size(); ++b)
      for (size_t gap = 1; gap <= 2 && gap <= b; ++gap) {
        const uint32_t ia = sequence[b - gap], ib = sequence[b];
        const auto key = std::make_pair(point.track[ia].image, point.track[ib].image);
        pairs[key].push_back(edges.size());
        by_point[id].push_back(edges.size());
        edges.push_back({id, ia, ib, Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()});
      }
  }
  GrayCache cache(scene);
  auto checkpoint = Clock::now();
  size_t completed = 0, accepted_edges = 0, sparse_pairs = 0;
  const size_t active_pairs = std::count_if(pairs.begin(), pairs.end(),
                                            [](const auto& pair) { return pair.second.size() >= 8; });
  std::vector<double> round_trips;
  for (const auto& [key, indices] : pairs) {
    if (indices.size() < 8) {
      ++sparse_pairs;
      continue;
    }
    const auto source = cache.Get(key.first), target = cache.Get(key.second);
    Parallel(indices.size(), workers, [&](size_t index) {
      auto& edge = edges[indices[index]];
      const auto& point = model.points[edge.point];
      const auto& a = point.track[edge.a];
      const auto& b = point.track[edge.b];
      const auto& pa = model.images[a.image].pixels[a.feature];
      const auto& pb = model.images[b.image].pixels[b.feature];
      if (!Warp(model, point.xyz, a, b, normals[edge.point], edge.affine))
        return;
      const auto reference = Sample(source, pa, Eigen::Matrix2d::Identity(), Radius);
      Eigen::Vector2d refined, reverse;
      if (!Textured(reference) || !MatchPatch(reference, target, pb, edge.affine, refined))
        return;
      const auto back_reference = Sample(target, refined, Eigen::Matrix2d::Identity(), Radius);
      if (!Textured(back_reference) ||
          !MatchPatch(back_reference, source, pa, edge.affine.inverse(), reverse))
        return;
      edge.round_trip = (reverse - pa).norm();
      if (edge.round_trip > .35)
        return;
      edge.shift = refined - pb;
      edge.accepted = true;
    });
    for (size_t id : indices)
      if (edges[id].accepted) {
        ++accepted_edges;
        round_trips.push_back(edges[id].round_trip);
      }
    ++completed;
    if (Seconds(checkpoint) > 25) {
      Log("Photometric view pairs " + std::to_string(completed) + "/" + std::to_string(active_pairs) +
          ", accepted edges " + std::to_string(accepted_edges));
      checkpoint = Clock::now();
    }
  }
  std::vector<std::vector<Observation>> refined_tracks(model.points.size());
  std::vector<std::vector<double>> changes(model.points.size()), cycle_errors(model.points.size());
  std::vector<uint8_t> rejected_cycles(model.points.size());
  Parallel(model.points.size(), workers, [&](size_t id) {
    const auto& point = model.points[id];
    if (by_point[id].empty())
      return;
    const size_t count = point.track.size();
    std::vector<uint32_t> parent(count), degree(count), component_size(count);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](uint32_t node) {
      while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
      }
      return node;
    };
    for (size_t edge_id : by_point[id]) {
      const auto& edge = edges[edge_id];
      if (edge.accepted) {
        parent[root(edge.b)] = root(edge.a);
        ++degree[edge.a];
        ++degree[edge.b];
      }
    }
    for (uint32_t node = 0; node < count; ++node)
      ++component_size[root(node)];
    const uint32_t largest = uint32_t(std::max_element(component_size.begin(), component_size.end()) -
                                      component_size.begin());
    const size_t size = component_size[largest];
    if (size < 3)
      return;
    std::vector<int> positions(count, -1);
    size_t anchor = 0, best_degree = 0, index = 0;
    for (uint32_t node = 0; node < count; ++node)
      if (root(node) == largest) {
        positions[node] = int(index);
        if (degree[node] > best_degree) {
          anchor = index;
          best_degree = degree[node];
        }
        ++index;
      }
    Eigen::MatrixXd information = Eigen::MatrixXd::Zero(2 * size, 2 * size);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2 * size);
    for (size_t edge_id : by_point[id]) {
      const auto& edge = edges[edge_id];
      if (!edge.accepted || positions[edge.a] < 0 || positions[edge.b] < 0)
        continue;
      const int a = 2 * positions[edge.a], b = 2 * positions[edge.b];
      information.block<2, 2>(a, a).noalias() += edge.affine.transpose() * edge.affine;
      information.block<2, 2>(a, b) -= edge.affine.transpose();
      information.block<2, 2>(b, a) -= edge.affine;
      information.block<2, 2>(b, b) += Eigen::Matrix2d::Identity();
      rhs.segment<2>(a).noalias() -= edge.affine.transpose() * edge.shift;
      rhs.segment<2>(b) += edge.shift;
    }
    information.block<2, 2>(2 * anchor, 2 * anchor) += 1e6 * Eigen::Matrix2d::Identity();
    const Eigen::VectorXd shift = information.ldlt().solve(rhs);
    if (!shift.allFinite())
      throw std::runtime_error("Nonfinite photometric graph solution");
    std::vector<double> residuals;
    for (size_t edge_id : by_point[id]) {
      const auto& edge = edges[edge_id];
      if (edge.accepted && positions[edge.a] >= 0 && positions[edge.b] >= 0)
        residuals.push_back((shift.segment<2>(2 * positions[edge.b]) -
                             edge.affine * shift.segment<2>(2 * positions[edge.a]) - edge.shift).norm());
    }
    if (Quantile(residuals, .9) > .4 || *std::max_element(residuals.begin(), residuals.end()) > .9) {
      rejected_cycles[id] = 1;
      return;
    }
    for (size_t i = 0; i < size; ++i)
      if (shift.segment<2>(2 * i).norm() > 4)
        return;
    for (uint32_t node = 0; node < count; ++node)
      if (positions[node] >= 0) {
        const auto& o = point.track[node];
        const Eigen::Vector2d delta = shift.segment<2>(2 * positions[node]);
        model.images[o.image].pixels[o.feature] += delta;
        refined_tracks[id].push_back(o);
        changes[id].push_back(delta.norm());
      }
    cycle_errors[id] = std::move(residuals);
  });
  size_t tracks = 0, observations = 0;
  std::vector<double> shifts, cycles;
  for (size_t id = 0; id < refined_tracks.size(); ++id) {
    tracks += !refined_tracks[id].empty();
    observations += refined_tracks[id].size();
    shifts.insert(shifts.end(), changes[id].begin(), changes[id].end());
    cycles.insert(cycles.end(), cycle_errors[id].begin(), cycle_errors[id].end());
  }
  report = {{"candidate_edges", edges.size()}, {"accepted_edges", accepted_edges},
            {"matching", {{"image_sigma_px", .7}, {"minimum_ncc", .94},
                           {"maximum_round_trip_px", .35}, {"maximum_cycle_p90_px", .4},
                           {"minimum_shared_tracks", 8}}},
            {"view_pairs", completed}, {"skipped_sparse_pairs", sparse_pairs},
            {"refined_tracks", tracks},
            {"refined_observations", observations}, {"pixel_shift_px", Statistics(shifts)},
            {"round_trip_px", Statistics(round_trips)}, {"cycle_residual_px", Statistics(cycles)},
            {"cycle_rejected_tracks", std::accumulate(rejected_cycles.begin(), rejected_cycles.end(), size_t(0))}};
  return refined_tracks;
}
Json FailedChecks(const Json& checks) {
  Json failed = Json::array();
  for (const auto& [name, passed] : checks.items())
    if (!passed.get<bool>())
      failed.push_back(name);
  return failed;
}
Json StructureCoverage(const Reconstruction& before, const Reconstruction& after,
                       const std::vector<PlaneConstraint>& planes) {
  std::array<size_t, 2> previous{}, next{};
  for (size_t id = 0; id < before.images.size(); ++id) {
    previous[before.images[id].camera] += before.images[id].registered;
    next[after.images[id].camera] += after.images[id].registered;
  }
  size_t retained = 0;
  for (const auto& plane : planes)
    retained += after.points[plane.point].track.size() >= 3;
  std::vector<uint32_t> parent(after.images.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](uint32_t id) {
    while (parent[id] != id) {
      parent[id] = parent[parent[id]];
      id = parent[id];
    }
    return id;
  };
  for (const auto& point : after.points)
    for (size_t i = 1; i < point.track.size(); ++i)
      parent[root(point.track[i].image)] = root(point.track[0].image);
  std::set<uint32_t> components;
  for (uint32_t id = 0; id < after.images.size(); ++id)
    if (after.images[id].registered)
      components.insert(root(id));
  const bool camera_coverage = next[0] >= .98 * previous[0] && next[1] >= .98 * previous[1];
  return {{"checks", {{"registered_image_retention", after.Registered() >= .99 * before.Registered()},
                       {"per_camera_image_retention", camera_coverage},
                       {"image_graph_connected", components.size() == 1},
                       {"las_supported_track_retention", retained >= .9 * planes.size()}}},
          {"registered_images_before", before.Registered()},
          {"registered_images_after", after.Registered()},
          {"camera_images_before", previous}, {"camera_images_after", next},
          {"las_supported_tracks_before", planes.size()},
          {"las_supported_tracks_after", retained},
          {"limits", {{"minimum_image_retention", .99}, {"minimum_per_camera_retention", .98},
                      {"minimum_las_supported_track_retention", .9}}}};
}
}  // namespace
PhotometricRefinement RefinePhotometric(const Scene& scene, const Reconstruction& source,
                                      int workers, const AlignmentLimits& limits) {
  PhotometricRefinement refined;
  refined.candidate = source;
  auto& model = refined.candidate;
  const auto before_quality = source.Quality(scene, limits.median_px, limits.p90_px);
  if (!before_quality.at("passed").get<bool>())
    throw std::runtime_error("Source reconstruction failed visual acceptance");
  const auto started = Clock::now();
  Log("Original-resolution photometric refinement: " + std::to_string(model.Registered()) +
      " images");
  const auto planes = RefinementPlanes(scene, model, workers);
  std::vector<Eigen::Vector3d> normals(model.points.size(), Eigen::Vector3d::Zero());
  for (const auto& plane : planes)
    normals[plane.point] = plane.normal;
  Json graph;
  const auto refined_tracks = RefineObservationGraph(scene, model, normals, workers, graph);
  auto precise = model;
  std::set<uint64_t> training, validation, photo_training;
  std::vector<size_t> coverage(model.images.size());
  std::vector<std::vector<uint8_t>> precise_observations(model.images.size());
  for (uint32_t image = 0; image < model.images.size(); ++image)
    precise_observations[image].resize(model.images[image].pixels.size());
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    auto& point = precise.points[id];
    if (point.track.size() < 3 || point.error > 2) {
      point.track.clear();
      continue;
    }
    if (id % 5 == 0) {
      if (refined_tracks[id].size() >= 3) {
        point.track = refined_tracks[id];
        validation.insert(id);
      }
    } else {
      training.insert(id);
      if (refined_tracks[id].size() >= 3)
        photo_training.insert(id);
      for (const auto& o : refined_tracks[id])
        precise_observations[o.image][o.feature] = 1;
      for (const auto& o : point.track)
        ++coverage[o.image];
    }
  }
  precise.RebuildReferences();
  auto& report = refined.report;
  report = {{"method", "cycle-consistent original-image graph with subpixel exposure-aware refinement"},
                 {"refinement_version", 3}, {"accepted", false},
                 {"limits", {{"las_support_fraction", limits.support_fraction},
                             {"las_median_m", limits.median_m}, {"las_p90_m", limits.p90_m}}},
                 {"pixel_shift_px", graph.at("pixel_shift_px")},
                 {"training_tracks", training.size()}, {"validation_tracks", validation.size()},
                 {"photo_training_tracks", photo_training.size()}, {"graph", graph},
                 {"validation_point_ids", validation}, {"holdout_rule", "all point IDs divisible by five excluded from camera optimization"},
                 {"training_observations_per_image", coverage}, {"surface_constraints", planes.size()}};
  if (training.size() < 200 || validation.size() < 50) {
    report["reason"] = "insufficient unambiguous multi-view photometric tracks";
    report["failed_checks"] = {"photometric_support"};
    report["structure_filter"] = {{"used", false}, {"reason", "photometric support insufficient"}};
    report["seconds"] = Seconds(started);
    return refined;
  }
  const auto validation_before = PointErrors(precise, validation);
  std::vector<PlaneConstraint> training_planes;
  for (const auto& plane : planes)
    if (training.count(plane.point))
      training_planes.push_back(plane);
  report["validation_before_px"] = validation_before;
  report["optimization"] = Optimize(precise, scene.cameras, "principal", workers, 150, {},
                                     training_planes, NavigationPositions(scene), training, precise_observations);
  const auto validation_after = PointErrors(precise, validation);
  report["validation_after_px"] = validation_after;
  report["training_error_px"] = PointErrors(precise, training, false);
  const auto plane_before = PlaneErrors(source, planes);
  model.cameras = precise.cameras;
  for (uint32_t image = 0; image < model.images.size(); ++image)
    model.images[image].pose = precise.images[image].pose;
  // Keep the jointly optimized points and their LAS constraints. Held-out
  // points are independently refit; their observations never optimize cameras.
  Parallel(model.points.size(), workers, [&](size_t id) {
    auto& point = model.points[id];
    if (training.count(id) || validation.count(id))
      point.xyz = precise.points[id].xyz;
    else if (point.track.size() >= 2)
      RefinePoint(model, point.track, point.xyz);
  });
  model.RebuildReferences();
  report["plane_before_m"] = plane_before;
  report["plane_after_m"] = PlaneErrors(model, planes);
  for (auto& point : model.points)
    if (!point.track.empty()) {
      point.error = 0;
      for (const auto& o : point.track) {
        const auto& image = model.images[o.image];
        point.error += (model.cameras[image.camera].Project(image.pose.Apply(point.xyz)) -
                        image.pixels[o.feature]).norm();
      }
      point.error /= point.track.size();
    }
  // Apply the same geometric validity filter as SfM. Strong pruning is a
  // separate candidate, and cannot make an unsuccessful camera solve pass.
  model.Filter();
  auto quality = model.Quality(scene, limits.median_px, limits.p90_px);
  const double previous_p90 = before_quality.at("reprojection_px").at("p90");
  const bool surface_fit = planes.empty() ||
      report.at("plane_after_m").at("p90").get<double>() <=
          plane_before.at("p90").get<double>() * 1.25 + .01;
  report["checks"] = {{"visual_acceptance", quality.at("passed")},
                       {"held_out_improvement", validation_after.at("median").get<double>() <
                                                  validation_before.at("median").get<double>() &&
                                                validation_after.at("p90").get<double>() <
                                                  validation_before.at("p90").get<double>()},
                       {"full_reprojection", quality.at("reprojection_px").at("p90").is_number() &&
                                                quality.at("reprojection_px").at("p90").get<double>() <=
                                                previous_p90 * 1.1 + .1},
                       {"surface_fit", surface_fit},
                       {"registered_images_preserved", model.Registered() == source.Registered()}};
  report["failed_checks"] = FailedChecks(report.at("checks"));
  refined.accepted = report.at("failed_checks").empty();
  report["accepted"] = refined.accepted;
  report["optimization_quality"] = quality;
  report["reason"] = refined.accepted ? "accepted" : "photometric candidate failed acceptance";
  if (refined.accepted && planes.size() >= 200) {
    Reconstruction cleaned = model;
    auto filtering = FilterRefinementStructure(scene, cleaned, limits);
    const auto cleaned_quality = cleaned.Quality(scene, limits.median_px, limits.p90_px);
    filtering["coverage"] = StructureCoverage(model, cleaned, planes);
    std::vector<PlaneConstraint> retained_planes;
    for (const auto& plane : planes)
      if (!cleaned.points[plane.point].track.empty())
        retained_planes.push_back(plane);
    filtering["plane_before_m"] = PlaneErrors(model, retained_planes);
    filtering["plane_after_m"] = PlaneErrors(cleaned, retained_planes);
    auto checks = filtering.at("coverage").at("checks");
    checks["visual_acceptance"] = cleaned_quality.at("passed");
    checks["full_reprojection"] = cleaned_quality.at("reprojection_px").at("p90").is_number() &&
        cleaned_quality.at("reprojection_px").at("p90").get<double>() <=
            quality.at("reprojection_px").at("p90").get<double>() * 1.05 + .05;
    checks["surface_fit"] = !retained_planes.empty() &&
        filtering.at("plane_after_m").at("p90").get<double>() <=
            filtering.at("plane_before_m").at("p90").get<double>() * 1.25 + .01;
    filtering["checks"] = checks;
    filtering["failed_checks"] = FailedChecks(checks);
    const bool used = filtering.at("failed_checks").empty();
    filtering["used"] = used;
    filtering["reason"] = used ? "accepted" : "pruning candidate failed coverage or geometry acceptance";
    filtering["candidate_quality"] = cleaned_quality;
    report["structure_filter"] = std::move(filtering);
    if (used) {
      model = std::move(cleaned);
      quality = cleaned_quality;
    }
  } else {
    report["structure_filter"] = {{"used", false},
        {"reason", refined.accepted ? "insufficient reliable LAS support for pruning" :
                                     "photometric candidate rejected"}};
  }
  report["candidate_quality"] = quality;
  report["seconds"] = Seconds(started);
  return refined;
}
Json RefinementResult(const Scene& scene, const Reconstruction& selected, const Json& original,
                      const Json& refinement) {
  std::vector<double> navigation_errors;
  const auto navigation = NavigationPositions(scene);
  for (uint32_t image = 0; image < selected.images.size(); ++image)
    if (selected.images[image].registered)
      navigation_errors.push_back((selected.images[image].pose.Center() - navigation[image]).norm());
  Json result = original;
  result["quality"] = selected.Quality(scene, original.at("quality").at("limits").at("median_px"),
                                     original.at("quality").at("limits").at("p90_px"));
  result["refinement"] = refinement;
  result["selected_model"] = "refined";
  result["navigation_error_m"] = Statistics(navigation_errors);
  result["build"] = BuildInfo();
  return result;
}
void RefinePhotometric(const Arguments& args) {
  args.Validate({"--work", "--output", "--workers"});
  const auto source = args.Path("--work"), output = args.Path("--output");
  const int workers = args.Integer("--workers", 4);
  if (workers < 1 || workers > 32)
    throw std::runtime_error("workers must be 1-32");
  if (fs::exists(output))
    throw std::runtime_error("Refinement output already exists; choose a new work directory");
  const auto scene = Scene::FromJson(ReadJson(source / "scene.json"));
  if (Contains(source, output) || Contains(scene.input, output))
    throw std::runtime_error("Refinement output must be outside source work and input");
  if (ReadScene(scene.input, scene.las).ToJson() != scene.ToJson())
    throw std::runtime_error("Source inputs changed; reconstruct in a new work directory");
  const auto original = ReadJson(source / "result.json");
  const auto model = ReadModel(SelectedModelPath(source, original));
  AlignmentLimits limits;
  if (original.contains("alignment")) {
    const auto& saved = original.at("alignment").at("limits");
    limits.support_fraction = saved.at("support_fraction");
    limits.median_m = saved.at("median_m");
    limits.p90_m = saved.at("p90_m");
  } else if (original.at("refinement").contains("limits")) {
    const auto& saved = original.at("refinement").at("limits");
    limits.support_fraction = saved.at("las_support_fraction");
    limits.median_m = saved.at("las_median_m");
    limits.p90_m = saved.at("las_p90_m");
  }
  limits.median_px = original.at("quality").at("limits").at("median_px");
  limits.p90_px = original.at("quality").at("limits").at("p90_px");
  fs::create_directories(output / "cache");
  auto refinement = RefinePhotometric(scene, model, workers, limits);
  refinement.report["source_work"] = source.u8string();
  const auto& selected = refinement.accepted ? refinement.candidate : model;
  const auto result = RefinementResult(scene, selected, original, refinement.report);
  WriteModel(output / "cache/aligned.bin", model);
  WriteJson(output / "cache/alignment.json", original);
  WriteModel(output / "cache/refinement_candidate.bin", refinement.candidate);
  WriteModel(output / "cache/refined.bin", selected);
  WriteJson(output / "cache/refinement.json", refinement.report);
  WriteJson(output / "scene.json", scene.ToJson());
  WriteJson(output / "result.json", result);
  for (const auto& image : scene.images) {
    if (!selected.images[image.id].registered)
      continue;
    const auto mask = MaskPath(output, image.id);
    fs::create_directories(mask.parent_path());
    fs::copy_file(MaskPath(source, image.id), mask);
  }
  ExportBundle(scene, selected, output, output / "colmap", true, .05, result);
  Log("Photometric refinement " + refinement.report.at("reason").get<std::string>());
  std::cout << Json({{"work", output.u8string()}, {"accepted", refinement.accepted},
                     {"reason", refinement.report.at("reason")}}).dump(2) << '\n';
}
}  // namespace geosun

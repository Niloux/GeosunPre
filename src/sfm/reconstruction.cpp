#include "sfm/reconstruction.h"
#include "sfm/optimization.h"
#include "io/cache.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <atomic>
#include <mutex>
#include <thread>
#include <numeric>
#include <unordered_map>
#include <set>
#include <random>

namespace geosun {
namespace {
constexpr double Pi = 3.14159265358979323846;
Eigen::Matrix3d Matrix(const cv::Mat& input) {
  cv::Mat m;
  input.convertTo(m, CV_64F);
  Eigen::Matrix3d out;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = m.at<double>(r, c);
  return out;
}
bool Triangulate(const Pose& a, const Pose& b, const Eigen::Vector3d& ra, const Eigen::Vector3d& rb,
                 Eigen::Vector3d& point) {
  Eigen::Matrix<double, 3, 4> pa, pb;
  pa.leftCols<3>() = a.Rotation().toRotationMatrix();
  pa.col(3) = a.Translation();
  pb.leftCols<3>() = b.Rotation().toRotationMatrix();
  pb.col(3) = b.Translation();
  Eigen::Matrix4d design;
  design.row(0) = ra.x() * pa.row(2) - ra.z() * pa.row(0);
  design.row(1) = ra.y() * pa.row(2) - ra.z() * pa.row(1);
  design.row(2) = rb.x() * pb.row(2) - rb.z() * pb.row(0);
  design.row(3) = rb.y() * pb.row(2) - rb.z() * pb.row(1);
  const auto v = design.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3).eval();
  if (std::abs(v[3]) < 1e-12)
    return false;
  point = v.head<3>() / v[3];
  return point.allFinite() && a.Apply(point).z() > 0 && b.Apply(point).z() > 0;
}
struct Pair {
  uint32_t a, b;
  Pose relative;
  std::vector<Match> inliers;
  double angle = 0;
};
struct Measurement {
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 3, 2> derivative;
};
Measurement Measure(const Camera& camera, const Eigen::Vector2d& pixel) {
  Measurement m;
  m.ray = camera.Ray(pixel);
  const auto& p = camera.params;
  const Eigen::Vector2d xy((pixel.x() - p[2]) / p[0], (pixel.y() - p[3]) / p[1]);
  const double r = xy.norm();
  if (r < 1e-7) {
    m.derivative.setZero();
    m.derivative(0, 0) = 1 / p[0];
    m.derivative(1, 1) = 1 / p[1];
    return m;
  }
  const double theta = std::atan2(m.ray.head<2>().norm(), m.ray.z()), t2 = theta * theta;
  const double slope = 1 + t2 * (3 * p[4] + t2 * (5 * p[5] + t2 * (7 * p[6] + t2 * 9 * p[7])));
  const double sine = std::sin(theta), cosine = std::cos(theta);
  m.derivative.topRows<2>() = Eigen::Matrix2d::Identity() * (sine / r) +
                              (cosine / (slope * r * r) - sine / (r * r * r)) * xy * xy.transpose();
  m.derivative.row(2) = -sine / (slope * r) * xy.transpose();
  m.derivative.col(0) /= p[0];
  m.derivative.col(1) /= p[1];
  return m;
}
double EpipolarSquared(const Eigen::Matrix3d& e, const Measurement& a, const Measurement& b) {
  const Eigen::Vector3d ea = e * a.ray, eb = e.transpose() * b.ray;
  const double value = b.ray.dot(ea);
  const double gradient =
      (a.derivative.transpose() * eb).squaredNorm() + (b.derivative.transpose() * ea).squaredNorm();
  return value * value / std::max(gradient, 1e-24);
}
bool RefitEssential(const std::vector<Measurement>& a, const std::vector<Measurement>& b,
                    const Eigen::Matrix3d& current, Eigen::Matrix3d& candidate) {
  Eigen::Matrix<double, 9, 9> normal = Eigen::Matrix<double, 9, 9>::Zero();
  size_t count = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (EpipolarSquared(current, a[i], b[i]) > 9)
      continue;
    const double gradient =
        (a[i].derivative.transpose() * current.transpose() * b[i].ray).squaredNorm() +
        (b[i].derivative.transpose() * current * a[i].ray).squaredNorm();
    Eigen::Matrix<double, 9, 1> row;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        row[3 * r + c] = b[i].ray[r] * a[i].ray[c];
    normal.noalias() += row * row.transpose() / std::max(gradient, 1e-24);
    ++count;
  }
  if (count < 8)
    return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solve(normal);
  if (solve.info() != Eigen::Success || solve.eigenvalues()[1] <= solve.eigenvalues()[8] * 1e-10)
    return false;
  const Eigen::Matrix<double, 9, 1> v = solve.eigenvectors().col(0);
  Eigen::Matrix3d e;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      e(r, c) = v[3 * r + c];
  const auto svd = e.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  candidate = svd.matrixU() * Eigen::Vector3d(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
  return candidate.allFinite();
}
Pair Verify(const Scene& scene, const Reconstruction& model, const MatchSet& matches) {
  Pair result{matches.a, matches.b};
  if (matches.matches.size() < 24)
    return result;
  const auto& ca = model.cameras[scene.images[matches.a].camera];
  const auto& cb = model.cameras[scene.images[matches.b].camera];
  std::vector<cv::Point2d> na, nb;
  std::vector<Measurement> ra, rb;
  for (const auto& m : matches.matches) {
    const auto x = Measure(ca, model.images[matches.a].pixels.at(m.a)),
               y = Measure(cb, model.images[matches.b].pixels.at(m.b));
    ra.push_back(x);
    rb.push_back(y);
    na.emplace_back(x.ray.x() / x.ray.z(), x.ray.y() / x.ray.z());
    nb.emplace_back(y.ray.x() / y.ray.z(), y.ray.y() / y.ray.z());
  }
  const double focal = (ca.params[0] + ca.params[1] + cb.params[0] + cb.params[1]) / 4;
  Eigen::Matrix3d essential;
  size_t best = 0;
  double best_cost = INFINITY;
  auto score = [&](const Eigen::Matrix3d& e) {
    size_t support = 0;
    double cost = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      const double error = EpipolarSquared(e, ra[i], rb[i]);
      support += error <= 9;
      cost += std::min(error, 9.);
    }
    if (support > best || (support == best && cost < best_cost)) {
      best = support;
      best_cost = cost;
      essential = e;
      return true;
    }
    return false;
  };
  const cv::Mat initial =
      cv::findEssentialMat(na, nb, 1., cv::Point2d(), cv::USAC_MAGSAC, .999, 3 / focal);
  for (int row = 0; row + 3 <= initial.rows; row += 3)
    score(Matrix(initial.rowRange(row, row + 3)));
  // Five-point hypotheses use perspective coordinates only for the minimal
  // solve. Consensus and local refinement use the native fisheye pixel metric.
  std::mt19937 rng(matches.a * 73856093u ^ matches.b * 19349663u);
  std::uniform_int_distribution<size_t> choose(0, ra.size() - 1);
  size_t trials = 1024;
  for (size_t trial = 0; trial < trials; ++trial) {
    std::array<size_t, 5> ids;
    std::vector<cv::Point2d> sa, sb;
    for (size_t k = 0; k < ids.size(); ++k) {
      size_t id;
      do {
        id = choose(rng);
      } while (std::find(ids.begin(), ids.begin() + k, id) != ids.begin() + k);
      ids[k] = id;
      sa.push_back(na[id]);
      sb.push_back(nb[id]);
    }
    const cv::Mat hypotheses =
        cv::findEssentialMat(sa, sb, 1., cv::Point2d(), cv::RANSAC, .999, 1e-6, 1);
    for (int row = 0; row + 3 <= hypotheses.rows; row += 3)
      score(Matrix(hypotheses.rowRange(row, row + 3)));
    if (best >= 24) {
      const double success = std::pow(double(best) / ra.size(), 5);
      const double needed = success >= 1 ? 32 : std::ceil(std::log(.001) / std::log1p(-success));
      trials = std::min(trials, size_t(std::clamp(needed, 32., 1024.)));
    }
  }
  if (best < 24)
    return result;
  for (int round = 0; round < 3; ++round) {
    Eigen::Matrix3d candidate;
    if (!RefitEssential(ra, rb, essential, candidate) || !score(candidate))
      break;
  }
  cv::Mat E(3, 3, CV_64F), mask(int(ra.size()), 1, CV_8UC1);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      E.at<double>(r, c) = essential(r, c);
  for (size_t i = 0; i < ra.size(); ++i)
    mask.at<uint8_t>(int(i)) = EpipolarSquared(essential, ra[i], rb[i]) <= 9 ? 255 : 0;
  cv::Mat rotation, translation;
  cv::recoverPose(E, na, nb, cv::Mat::eye(3, 3, CV_64F), rotation, translation, 10000, mask);
  if (rotation.empty())
    return result;
  result.relative.Set(
      Eigen::Quaterniond(Matrix(rotation)),
      {translation.at<double>(0), translation.at<double>(1), translation.at<double>(2)});
  const auto R = result.relative.Rotation();
  std::vector<double> angles;
  Pose origin;
  for (size_t i = 0; i < matches.matches.size(); ++i) {
    if (!mask.at<uint8_t>(int(i)))
      continue;
    const auto& m = matches.matches[i];
    const auto& xa = model.images[matches.a].pixels[m.a];
    const auto& xb = model.images[matches.b].pixels[m.b];
    Eigen::Vector3d point;
    if (Triangulate(origin, result.relative, ra[i].ray, rb[i].ray, point) &&
        (ca.Project(point) - xa).norm() <= 6 &&
        (cb.Project(result.relative.Apply(point)) - xb).norm() <= 6) {
      result.inliers.push_back(m);
      angles.push_back(std::acos(std::clamp(ra[i].ray.dot(R.conjugate() * rb[i].ray), -1., 1.)) *
                       180 / Pi);
    }
  }
  result.angle = angles.empty() ? 0 : Quantile(angles, .5);
  if (result.inliers.size() < 24)
    result.inliers.clear();
  return result;
}
struct Tracks {
  std::vector<std::vector<Observation>> tracks;
  std::vector<std::vector<uint32_t>> lookup;
  explicit Tracks(const Reconstruction& model, const std::vector<Pair>& pairs) {
    std::vector<size_t> offset{0};
    for (const auto& image : model.images)
      offset.push_back(offset.back() + image.pixels.size());
    std::vector<size_t> parent(offset.back()), sizes(offset.back(), 1);
    std::iota(parent.begin(), parent.end(), 0);
    std::unordered_map<size_t, std::vector<uint32_t>> labels;
    auto root = [&](size_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto images = [&](size_t r) {
      const auto it = labels.find(r);
      return it != labels.end()
                 ? it->second
                 : std::vector<uint32_t>{uint32_t(
                       std::upper_bound(offset.begin(), offset.end(), r) - offset.begin() - 1)};
    };
    for (const auto& pair : pairs)
      for (const auto& match : pair.inliers) {
        auto a = root(offset[pair.a] + match.a), b = root(offset[pair.b] + match.b);
        if (a == b)
          continue;
        auto ia = images(a), ib = images(b);
        bool conflict = false;
        for (uint32_t id : ia)
          if (std::binary_search(ib.begin(), ib.end(), id)) {
            conflict = true;
            break;
          }
        if (conflict)
          continue;
        if (sizes[a] < sizes[b]) {
          std::swap(a, b);
          std::swap(ia, ib);
        }
        std::vector<uint32_t> merged;
        std::merge(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(merged));
        parent[b] = a;
        sizes[a] += sizes[b];
        labels[a] = std::move(merged);
        labels.erase(b);
      }
    lookup.resize(model.images.size());
    std::unordered_map<size_t, uint32_t> ids;
    for (uint32_t im = 0; im < model.images.size(); ++im) {
      lookup[im].assign(model.images[im].pixels.size(), UINT32_MAX);
      for (uint32_t k = 0; k < model.images[im].pixels.size(); ++k) {
        const auto r = root(offset[im] + k);
        if (sizes[r] < 2)
          continue;
        auto it = ids.find(r);
        if (it == ids.end()) {
          const uint32_t id = uint32_t(tracks.size());
          tracks.emplace_back();
          it = ids.emplace(r, id).first;
        }
        lookup[im][k] = it->second;
        tracks[it->second].push_back({im, k});
      }
    }
  }
};
double PointCost(const Reconstruction& model, const std::vector<Observation>& track,
                 const Eigen::Vector3d& point) {
  double cost = 0;
  for (const auto& o : track) {
    const auto& image = model.images[o.image];
    const Eigen::Vector3d v = image.pose.Apply(point);
    if (v.z() <= 0)
      return INFINITY;
    const double error = (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm();
    cost += error <= 2 ? .5 * error * error : 2 * (error - 1);
  }
  return cost;
}
void RefinePoint(const Reconstruction& model, const std::vector<Observation>& track,
                 Eigen::Vector3d& point) {
  double cost = PointCost(model, track, point);
  for (int iteration = 0; iteration < 6; ++iteration) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (const auto& o : track) {
      const auto& image = model.images[o.image];
      const auto& camera = model.cameras[image.camera];
      const Eigen::Vector3d v = image.pose.Apply(point);
      const Eigen::Vector2d residual = camera.Project(v) - image.pixels[o.feature];
      const Eigen::Matrix<double, 2, 3> j =
          camera.ProjectionJacobian(v) * image.pose.Rotation().toRotationMatrix();
      const double weight = std::min(1., 2 / std::max(residual.norm(), 1e-12));
      normal.noalias() += weight * j.transpose() * j;
      rhs.noalias() -= weight * j.transpose() * residual;
    }
    const Eigen::Vector3d step = normal.ldlt().solve(rhs);
    if (!step.allFinite())
      throw std::runtime_error("Nonfinite triangulation refinement");
    bool accepted = false;
    for (int backtrack = 0; backtrack < 6; ++backtrack) {
      const Eigen::Vector3d candidate = point + step * std::pow(.5, backtrack);
      const double next = PointCost(model, track, candidate);
      if (std::isfinite(next) && next < cost) {
        point = candidate;
        const double improvement = cost - next;
        cost = next;
        accepted = true;
        if (improvement < 1e-7 * std::max(1., cost))
          return;
        break;
      }
    }
    if (!accepted)
      break;
  }
}
bool AddPoint(Reconstruction& model, const std::vector<Observation>& observations,
              uint64_t& identity, bool refine = false) {
  if (identity != NoPoint && !model.points[identity].track.empty()) {
    auto& point = model.points[identity];
    std::set<uint32_t> seen;
    for (const auto& o : point.track)
      seen.insert(o.image);
    const size_t before = point.track.size();
    for (const auto& o : observations) {
      const auto& image = model.images[o.image];
      if (!image.registered || seen.count(o.image) ||
          (image.points[o.feature] != NoPoint && image.points[o.feature] != identity))
        continue;
      const auto v = image.pose.Apply(point.xyz);
      if (v.z() > 0 &&
          (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm() <= 4.5) {
        point.track.push_back(o);
        model.images[o.image].points[o.feature] = identity;
        seen.insert(o.image);
      }
    }
    if (refine || point.track.size() > before)
      RefinePoint(model, point.track, point.xyz);
    return true;
  }
  std::vector<Observation> track;
  for (const auto& o : observations)
    if (model.images[o.image].registered && (model.images[o.image].points[o.feature] == NoPoint ||
                                             model.images[o.image].points[o.feature] == identity))
      track.push_back(o);
  if (track.size() < 2)
    return false;
  double best = 1;
  size_t first = 0, second = 0;
  std::vector<Eigen::Vector3d> rays, world;
  for (const auto& o : track) {
    const auto& image = model.images[o.image];
    const auto ray = model.cameras[image.camera].Ray(image.pixels[o.feature]);
    rays.push_back(ray);
    world.push_back(image.pose.Rotation().conjugate() * ray);
  }
  for (size_t i = 0; i < track.size(); ++i)
    for (size_t j = i + 1; j < track.size(); ++j) {
      const double cosine = world[i].dot(world[j]);
      if (cosine < best) {
        best = cosine;
        first = i;
        second = j;
      }
    }
  if (best > std::cos(.5 * Pi / 180))
    return false;
  Eigen::Vector3d point;
  if (!Triangulate(model.images[track[first].image].pose, model.images[track[second].image].pose,
                   rays[first], rays[second], point))
    return false;
  std::vector<Observation> good;
  for (const auto& o : track) {
    const auto& im = model.images[o.image];
    const auto v = im.pose.Apply(point);
    if (v.z() > 0 && (model.cameras[im.camera].Project(v) - im.pixels[o.feature]).norm() <= 6)
      good.push_back(o);
  }
  if (good.size() < 2)
    return false;
  RefinePoint(model, good, point);
  if (identity == NoPoint) {
    identity = model.points.size();
    model.points.push_back({point, {127, 127, 127}, 0, good});
  } else {
    model.points[identity].xyz = point;
    model.points[identity].track = std::move(good);
  }
  for (const auto& o : model.points[identity].track)
    model.images[o.image].points[o.feature] = identity;
  return true;
}
Json CompleteTracks(Reconstruction& model, const Tracks& graph, std::vector<uint64_t>& identities) {
  const auto started = Clock::now();
  const size_t before = model.Observations();
  size_t merged = 0;
  for (size_t i = 0; i < graph.tracks.size(); ++i) {
    std::set<uint64_t> existing;
    for (const auto& o : graph.tracks[i])
      if (model.images[o.image].points[o.feature] != NoPoint)
        existing.insert(model.images[o.image].points[o.feature]);
    if (existing.size() == 1)
      identities[i] = *existing.begin();
    else if (existing.size() > 1) {
      std::map<uint32_t, uint32_t> features;
      std::vector<Observation> track;
      bool conflict = false;
      uint64_t winner = *existing.begin();
      for (uint64_t id : existing) {
        if (model.points[id].track.size() > model.points[winner].track.size())
          winner = id;
        for (const auto& o : model.points[id].track) {
          const auto inserted = features.emplace(o.image, o.feature);
          if (!inserted.second && inserted.first->second != o.feature)
            conflict = true;
          if (inserted.second)
            track.push_back(o);
        }
      }
      if (conflict)
        continue;
      Eigen::Vector3d point = model.points[winner].xyz;
      RefinePoint(model, track, point);
      bool valid = true;
      for (const auto& o : track) {
        const auto& image = model.images[o.image];
        const auto v = image.pose.Apply(point);
        valid = valid && v.z() > 0 &&
                (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm() <= 4.5;
      }
      if (!valid)
        continue;
      for (uint64_t id : existing)
        if (id != winner) {
          model.points[id].track.clear();
          ++merged;
        }
      model.points[winner].xyz = point;
      model.points[winner].track = std::move(track);
      for (const auto& o : model.points[winner].track)
        model.images[o.image].points[o.feature] = winner;
      identities[i] = winner;
    }
    AddPoint(model, graph.tracks[i], identities[i], true);
  }
  const size_t completed = model.Observations();
  model.Filter();
  return {{"added_observations", completed - before},
          {"removed_observations", completed - model.Observations()},
          {"merged_points", merged},
          {"seconds", Seconds(started)}};
}
size_t Correspondences(const Reconstruction& model, const std::vector<Pair>& pairs,
                       const std::vector<size_t>& neighbors, uint32_t id,
                       std::vector<cv::Point3d>* xyz = nullptr,
                       std::vector<cv::Point2d>* xy = nullptr,
                       std::vector<uint32_t>* features = nullptr,
                       std::vector<uint64_t>* identities = nullptr) {
  const auto& im = model.images[id];
  const auto& cam = model.cameras[im.camera];
  std::vector<uint8_t> supported(im.pixels.size());
  struct Vote {
    size_t count = 0;
    double score = 0;
  };
  std::map<uint32_t, std::map<uint64_t, Vote>> votes;
  for (size_t index : neighbors) {
    const auto& pair = pairs[index];
    const bool first = pair.a == id;
    const auto& other = model.images[first ? pair.b : pair.a];
    if (!other.registered)
      continue;
    for (const auto& match : pair.inliers) {
      const uint32_t feature = first ? match.a : match.b;
      const uint64_t point = other.points[first ? match.b : match.a];
      if (point == NoPoint)
        continue;
      supported[feature] = 1;
      if (xyz) {
        auto& vote = votes[feature][point];
        ++vote.count;
        vote.score += match.score;
      }
    }
  }
  if (!xyz)
    return std::count(supported.begin(), supported.end(), uint8_t(1));
  struct Candidate {
    uint32_t feature;
    uint64_t point;
    Vote vote;
  };
  std::vector<Candidate> candidates;
  for (const auto& [feature, options] : votes) {
    Candidate best{feature, NoPoint, {}};
    for (const auto& [point, vote] : options)
      if (vote.count > best.vote.count ||
          (vote.count == best.vote.count && vote.score > best.vote.score))
        best = {feature, point, vote};
    candidates.push_back(best);
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    if (a.vote.count != b.vote.count)
      return a.vote.count > b.vote.count;
    if (a.vote.score != b.vote.score)
      return a.vote.score > b.vote.score;
    return a.feature < b.feature;
  });
  std::set<uint64_t> used;
  for (const auto& candidate : candidates) {
    if (used.insert(candidate.point).second) {
      const auto& p = model.points[candidate.point].xyz;
      xyz->emplace_back(p.x(), p.y(), p.z());
      const auto ray = cam.Ray(im.pixels[candidate.feature]);
      xy->emplace_back(ray.x() / ray.z(), ray.y() / ray.z());
      features->push_back(candidate.feature);
      identities->push_back(candidate.point);
    }
  }
  return xyz->size();
}
bool Register(Reconstruction& model, const std::vector<Pair>& pairs,
              const std::vector<size_t>& neighbors, uint32_t id) {
  std::vector<cv::Point3d> xyz;
  std::vector<cv::Point2d> xy;
  std::vector<uint32_t> features;
  std::vector<uint64_t> identities;
  if (Correspondences(model, pairs, neighbors, id, &xyz, &xy, &features, &identities) < 24)
    return false;
  auto& image = model.images[id];
  const auto& cam = model.cameras[image.camera];
  Pose selected;
  size_t maximum = 0;
  for (double threshold : {4., 12., 32.}) {
    cv::Mat rv, tv, inliers;
    if (!cv::solvePnPRansac(xyz, xy, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rv, tv, false, 2000,
                            float(threshold / std::min(cam.params[0], cam.params[1])), .999,
                            inliers, cv::SOLVEPNP_EPNP) ||
        inliers.rows < 24)
      continue;
    cv::Mat R;
    cv::Rodrigues(rv, R);
    Pose pose;
    pose.Set(Eigen::Quaterniond(Matrix(R)), {tv.at<double>(0), tv.at<double>(1), tv.at<double>(2)});
    std::vector<Eigen::Vector3d> fit_points;
    std::vector<Eigen::Vector2d> fit_pixels;
    for (size_t i = 0; i < xyz.size(); ++i) {
      const Eigen::Vector3d point(xyz[i].x, xyz[i].y, xyz[i].z);
      const auto v = pose.Apply(point);
      const auto& pixel = image.pixels[features[i]];
      if (v.z() > 0 && (cam.Project(v) - pixel).norm() <= 12) {
        fit_points.push_back(point);
        fit_pixels.push_back(pixel);
      }
    }
    if (fit_points.size() < 24)
      continue;
    RefinePose(cam, pose, fit_points, fit_pixels);
    size_t support = 0;
    std::set<int> cells;
    for (size_t i = 0; i < xyz.size(); ++i) {
      const auto v = pose.Apply({xyz[i].x, xyz[i].y, xyz[i].z});
      const auto& pixel = image.pixels[features[i]];
      if (v.z() > 0 && (cam.Project(v) - pixel).norm() <= 6) {
        ++support;
        cells.insert(int(pixel.x() / cam.width * 8) + 8 * int(pixel.y() / cam.height * 8));
      }
    }
    if (support >= 32 && cells.size() >= 8 && support > maximum) {
      selected = pose;
      maximum = support;
    }
    if (maximum >= std::max<size_t>(100, xyz.size() / 2))
      break;
  }
  if (!maximum)
    return false;
  image.pose = selected;
  image.registered = true;
  for (size_t i = 0; i < xyz.size(); ++i) {
    const auto v = selected.Apply(model.points[identities[i]].xyz);
    if (v.z() > 0 && (cam.Project(v) - image.pixels[features[i]]).norm() <= 6) {
      model.points[identities[i]].track.push_back({id, features[i]});
      image.points[features[i]] = identities[i];
    }
  }
  return true;
}
}  // namespace
Reconstruction Reconstruct(const Scene& scene, const fs::path& work, int workers, Json& report) {
  const auto started = Clock::now();
  Reconstruction m;
  const auto checkpoint = work / "cache/incremental.bin";
  const bool resumed = fs::exists(checkpoint);
  if (resumed) {
    m = ReadModel(checkpoint);
    if (m.images.size() != scene.images.size())
      throw std::runtime_error("Incremental image count differs");
    Log("Resume SfM at " + std::to_string(m.Registered()) + "/" + std::to_string(m.images.size()));
  } else {
    m.cameras = scene.cameras;
    m.images.resize(scene.images.size());
  }
  for (uint32_t id = 0; id < m.images.size(); ++id) {
    const auto f = ReadFeatures(FeaturePath(work, id));
    if (f.image != id)
      throw std::runtime_error("Feature/image identity differs");
    if (resumed) {
      if (m.images[id].camera != scene.images[id].camera ||
          m.images[id].pixels.size() != f.pixels.size())
        throw std::runtime_error("Incremental feature identity differs");
      for (size_t k = 0; k < f.pixels.size(); ++k)
        if (m.images[id].pixels[k] != Eigen::Vector2d(f.pixels[k][0], f.pixels[k][1]))
          throw std::runtime_error("Incremental feature coordinates differ");
    } else {
      m.images[id].camera = scene.images[id].camera;
      for (const auto& p : f.pixels)
        m.images[id].pixels.emplace_back(p[0], p[1]);
      m.images[id].points.assign(f.pixels.size(), NoPoint);
    }
  }
  const auto catalog = ReadJson(work / "cache/pairs.json");
  std::vector<std::array<uint32_t, 2>> candidates;
  for (const auto& pair : catalog.at("pairs"))
    candidates.push_back(pair.get<std::array<uint32_t, 2>>());
  std::vector<Pair> pairs(candidates.size());
  const auto geometry_started = Clock::now();
  std::atomic<size_t> next{0}, done{0};
  std::exception_ptr error;
  std::mutex mutex;
  std::vector<std::thread> threads;
  for (int worker = 0; worker < workers; ++worker)
    threads.emplace_back([&] {
      try {
        for (;;) {
          const size_t i = next.fetch_add(1);
          if (i >= candidates.size())
            break;
          const auto p = candidates[i];
          pairs[i] = Verify(scene, m, ReadMatches(MatchPath(work, p[0], p[1])));
          const auto n = done.fetch_add(1) + 1;
          if (n % 500 == 0) {
            std::lock_guard<std::mutex> lock(mutex);
            Log("Geometry " + std::to_string(n) + "/" + std::to_string(candidates.size()));
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error)
          error = std::current_exception();
        next = candidates.size();
      }
    });
  for (auto& t : threads)
    t.join();
  if (error)
    std::rethrow_exception(error);
  const double geometry_seconds = Seconds(geometry_started);
  Tracks graph(m, pairs);
  std::vector<uint64_t> points(graph.tracks.size(), NoPoint);
  std::vector<std::vector<size_t>> adjacency(m.images.size());
  for (size_t i = 0; i < pairs.size(); ++i)
    if (!pairs[i].inliers.empty()) {
      adjacency[pairs[i].a].push_back(i);
      adjacency[pairs[i].b].push_back(i);
    }
  Json optimization = Json::array();
  Json completion = Json::array();
  if (resumed) {
    for (size_t i = 0; i < graph.tracks.size(); ++i) {
      std::set<uint64_t> existing;
      for (const auto& o : graph.tracks[i])
        if (m.images[o.image].points[o.feature] != NoPoint)
          existing.insert(m.images[o.image].points[o.feature]);
      if (existing.size() == 1)
        points[i] = *existing.begin();
    }
  } else {
    const Pair* seed = nullptr;
    double best = 0;
    const auto positions = NavigationPositions(scene);
    for (const auto& p : pairs)
      if (p.inliers.size() >= 100 && p.angle >= 1. &&
          (positions[p.a] - positions[p.b]).norm() > .2) {
        const double score = p.inliers.size() * std::min(p.angle, 8.);
        if (score > best) {
          seed = &p;
          best = score;
        }
      }
    if (!seed)
      throw std::runtime_error("No nondegenerate reconstruction seed");
    m.gauge_image = seed->a;
    m.scale_image = seed->b;
    m.gauge_baseline = 1;
    m.images[seed->a].registered = true;
    m.images[seed->b].registered = true;
    m.images[seed->b].pose = seed->relative;
    for (uint32_t i = 0; i < graph.tracks.size(); ++i)
      AddPoint(m, graph.tracks[i], points[i]);
    m.RebuildReferences();
    optimization.push_back(Optimize(m, scene.cameras, "fixed", workers, 100));
    m.Filter();
    Log("Seed " + scene.images[seed->a].name + " + " + scene.images[seed->b].name);
  }
  size_t last_global = m.Registered();
  std::set<uint32_t> recent;
  if (!resumed)
    recent = {m.gauge_image, m.scale_image};
  std::array<size_t, 2> resumed_counts{};
  for (const auto& image : m.images)
    if (image.registered)
      ++resumed_counts[image.camera];
  bool calibrated =
      resumed && m.Registered() >= 32 && resumed_counts[0] >= 8 && resumed_counts[1] >= 8;
  while (m.Registered() < m.images.size()) {
    std::vector<std::pair<size_t, uint32_t>> options;
    for (uint32_t id = 0; id < m.images.size(); ++id)
      if (!m.images[id].registered)
        options.emplace_back(Correspondences(m, pairs, adjacency[id], id), id);
    std::sort(options.rbegin(), options.rend());
    bool added = false;
    for (const auto& [count, id] : options) {
      if (count < 24)
        break;
      if (!Register(m, pairs, adjacency[id], id))
        continue;
      added = true;
      recent.insert(id);
      std::set<uint32_t> touched;
      for (uint32_t track : graph.lookup[id])
        if (track != UINT32_MAX)
          touched.insert(track);
      for (uint32_t track : touched)
        AddPoint(m, graph.tracks[track], points[track]);
      const auto registered = m.Registered();
      if (registered % 8 == 0) {
        std::set<uint32_t> local = recent;
        std::map<uint32_t, size_t> connections;
        for (uint32_t image : recent)
          for (uint64_t point : m.images[image].points)
            if (point != NoPoint)
              for (const auto& o : m.points[point].track)
                if (!recent.count(o.image))
                  ++connections[o.image];
        std::vector<std::pair<size_t, uint32_t>> neighbors;
        for (const auto& [image, shared_points] : connections)
          neighbors.emplace_back(shared_points, image);
        std::sort(neighbors.rbegin(), neighbors.rend());
        for (size_t k = 0; k < std::min<size_t>(8, neighbors.size()); ++k)
          local.insert(neighbors[k].second);
        optimization.push_back(Optimize(m, scene.cameras, "fixed", workers, 30, local));
        m.Filter();
        recent.clear();
      }
      if (registered >= 32 && !calibrated) {
        std::array<size_t, 2> counts{};
        for (const auto& image : m.images)
          if (image.registered)
            ++counts[image.camera];
        if (counts[0] >= 8 && counts[1] >= 8) {
          optimization.push_back(Optimize(m, scene.cameras, "focal", workers, 100));
          m.Filter();
          completion.push_back(CompleteTracks(m, graph, points));
          optimization.push_back(Optimize(m, scene.cameras, "radial", workers, 100));
          m.Filter();
          completion.push_back(CompleteTracks(m, graph, points));
          calibrated = true;
          last_global = registered;
        }
      } else if (registered >= std::max(last_global + 40, size_t(last_global * 1.35))) {
        optimization.push_back(
            Optimize(m, scene.cameras, calibrated ? "radial" : "fixed", workers, 100));
        m.Filter();
        completion.push_back(CompleteTracks(m, graph, points));
        last_global = registered;
      }
      if (registered % 16 == 0)
        Log("Registered " + std::to_string(registered) + "/" + std::to_string(m.images.size()));
      if (registered % 8 == 0)
        WriteModel(checkpoint, m);
      break;
    }
    if (!added) {
      if (m.Registered() > last_global) {
        optimization.push_back(Optimize(m, scene.cameras, "radial", workers, 150));
        m.Filter();
        completion.push_back(CompleteTracks(m, graph, points));
        last_global = m.Registered();
        continue;
      }
      Json missing = Json::array();
      for (uint32_t id = 0; id < m.images.size(); ++id)
        if (!m.images[id].registered)
          missing.push_back(id);
      WriteJson(work / "cache/unregistered.json", missing);
      WriteModel(checkpoint, m);
      for (uint32_t id : missing.get<std::vector<uint32_t>>())
        Log("Unregistered " + scene.images[id].name + ": " +
            std::to_string(Correspondences(m, pairs, adjacency[id], id)) +
            " candidate observations");
      Log("SfM stalled at " + std::to_string(m.Registered()) + "/" +
          std::to_string(m.images.size()));
      throw RegistrationIncomplete(missing.get<std::vector<uint32_t>>());
    }
  }
  // Growth has already refined focal length and distortion. Calibrate the
  // complete image set once, hold intrinsics while incorporating new tracks,
  // then finish with a strict joint refinement of all parameters.
  optimization.push_back(Optimize(m, scene.cameras, "principal", workers, 150));
  for (int round = 0; round < 3; ++round) {
    const size_t before = m.Observations();
    m.Filter();
    const size_t removed = before - m.Observations();
    auto completed = CompleteTracks(m, graph, points);
    const size_t total_removed = removed + completed.at("removed_observations").get<size_t>();
    const size_t changed = total_removed + completed.at("added_observations").get<size_t>();
    completed["removed_observations"] = total_removed;
    completion.push_back(std::move(completed));
    if (round == 2 || (changed <= std::max<size_t>(1, size_t(before * .0005)) &&
                       optimization.back().at("converged").get<bool>()))
      break;
    optimization.push_back(Optimize(m, scene.cameras, "fixed", workers, 100));
  }
  optimization.push_back(Optimize(m, scene.cameras, "principal", workers, 200));
  m.Filter();
  for (uint32_t id = 0; id < m.images.size(); ++id) {
    const auto rgb = ReadImage(scene.images[id].source, cv::IMREAD_COLOR);
    for (uint32_t k = 0; k < m.images[id].points.size(); ++k)
      if (m.images[id].points[k] != NoPoint) {
        auto& p = m.points[m.images[id].points[k]];
        if (p.track.front().image != id)
          continue;
        const auto& xy = m.images[id].pixels[k];
        const auto color = rgb.at<cv::Vec3b>(std::clamp(int(xy.y()), 0, rgb.rows - 1),
                                             std::clamp(int(xy.x()), 0, rgb.cols - 1));
        p.color = {color[2], color[1], color[0]};
      }
  }
  report = {{"verified_pairs", std::count_if(pairs.begin(), pairs.end(),
                                             [](const auto& p) { return !p.inliers.empty(); })},
            {"tracks", graph.tracks.size()},
            {"seconds", Seconds(started)},
            {"geometry_seconds", geometry_seconds},
            {"completion", completion},
            {"optimization", optimization}};
  fs::remove(work / "cache/unregistered.json");
  return m;
}
}  // namespace geosun

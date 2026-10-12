#include "sfm/reconstruction.h"
#include "sfm/optimization.h"
#include "io/cache.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SVD>
#include <atomic>
#include <mutex>
#include <thread>
#include <numeric>
#include <unordered_map>
#include <set>

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
Pair Verify(const Scene& scene, const Reconstruction& model, const MatchSet& matches) {
  Pair result{matches.a, matches.b};
  if (matches.matches.size() < 24)
    return result;
  const auto& ca = model.cameras[scene.images[matches.a].camera];
  const auto& cb = model.cameras[scene.images[matches.b].camera];
  std::vector<cv::Point2d> na, nb;
  std::vector<Eigen::Vector3d> ra, rb;
  for (const auto& m : matches.matches) {
    const auto x = ca.Ray(model.images[matches.a].pixels.at(m.a)),
               y = cb.Ray(model.images[matches.b].pixels.at(m.b));
    ra.push_back(x);
    rb.push_back(y);
    na.emplace_back(x.x() / x.z(), x.y() / x.z());
    nb.emplace_back(y.x() / y.z(), y.y() / y.z());
  }
  const double focal = (ca.params[0] + ca.params[1] + cb.params[0] + cb.params[1]) / 4;
  cv::Mat mask;
  cv::Mat E =
      cv::findEssentialMat(na, nb, 1., cv::Point2d(), cv::USAC_MAGSAC, .999, 3 / focal, mask);
  if (E.empty() || E.rows != 3 || E.cols != 3)
    return result;
  cv::Mat rotation, translation;
  cv::recoverPose(E, na, nb, cv::Mat::eye(3, 3, CV_64F), rotation, translation, 10000, mask);
  if (rotation.empty())
    return result;
  result.relative.Set(
      Eigen::Quaterniond(Matrix(rotation)),
      {translation.at<double>(0), translation.at<double>(1), translation.at<double>(2)});
  const auto essential = Matrix(E);
  const auto R = result.relative.Rotation();
  std::vector<double> angles;
  Pose origin;
  for (size_t i = 0; i < matches.matches.size(); ++i) {
    if (!mask.at<uint8_t>(int(i)))
      continue;
    const auto& m = matches.matches[i];
    const auto& xa = model.images[matches.a].pixels[m.a];
    const auto& xb = model.images[matches.b].pixels[m.b];
    double gradient = 0;
    for (int axis = 0; axis < 2; ++axis) {
      Eigen::Vector2d ap = xa, am = xa, bp = xb, bm = xb;
      ap[axis] += .25;
      am[axis] -= .25;
      bp[axis] += .25;
      bm[axis] -= .25;
      const double ga = rb[i].dot(essential * (ca.Ray(ap) - ca.Ray(am)) / .5),
                   gb = ((cb.Ray(bp) - cb.Ray(bm)) / .5).dot(essential * ra[i]);
      gradient += ga * ga + gb * gb;
    }
    const double error =
        std::abs(rb[i].dot(essential * ra[i])) / std::sqrt(std::max(gradient, 1e-24));
    Eigen::Vector3d point;
    if (error <= 3 && Triangulate(origin, result.relative, ra[i], rb[i], point) &&
        (ca.Project(point) - xa).norm() <= 6 &&
        (cb.Project(result.relative.Apply(point)) - xb).norm() <= 6) {
      result.inliers.push_back(m);
      angles.push_back(std::acos(std::clamp(ra[i].dot(R.conjugate() * rb[i]), -1., 1.)) * 180 / Pi);
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
bool AddPoint(Reconstruction& model, const std::vector<Observation>& observations,
              uint64_t& identity) {
  if (identity != NoPoint && !model.points[identity].track.empty()) {
    auto& point = model.points[identity];
    std::set<uint32_t> seen;
    for (const auto& o : point.track)
      seen.insert(o.image);
    for (const auto& o : observations) {
      const auto& image = model.images[o.image];
      if (!image.registered || seen.count(o.image) ||
          (image.points[o.feature] != NoPoint && image.points[o.feature] != identity))
        continue;
      const auto v = image.pose.Apply(point.xyz);
      if (v.z() > 0 &&
          (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm() <= 6) {
        point.track.push_back(o);
        seen.insert(o.image);
      }
    }
    return true;
  }
  std::vector<Observation> track;
  for (const auto& o : observations)
    if (model.images[o.image].registered && (model.images[o.image].points[o.feature] == NoPoint ||
                                             model.images[o.image].points[o.feature] == identity))
      track.push_back(o);
  if (track.size() < 2)
    return false;
  double best = 0;
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
      const double angle = std::acos(std::clamp(world[i].dot(world[j]), -1., 1.));
      if (angle > best) {
        best = angle;
        first = i;
        second = j;
      }
    }
  if (best < .5 * Pi / 180)
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
  if (identity == NoPoint) {
    identity = model.points.size();
    model.points.push_back({point, {127, 127, 127}, 0, good});
  } else {
    model.points[identity].xyz = point;
    model.points[identity].track = std::move(good);
  }
  return true;
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
    if (v.z() > 0 && (cam.Project(v) - image.pixels[features[i]]).norm() <= 6)
      model.points[identities[i]].track.push_back({id, features[i]});
  }
  model.RebuildReferences();
  return true;
}
}  // namespace
Reconstruction Reconstruct(const Scene& scene, const fs::path& work, int workers, Json& report) {
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
  Tracks graph(m, pairs);
  std::vector<uint64_t> points(graph.tracks.size(), NoPoint);
  std::vector<std::vector<size_t>> adjacency(m.images.size());
  for (size_t i = 0; i < pairs.size(); ++i)
    if (!pairs[i].inliers.empty()) {
      adjacency[pairs[i].a].push_back(i);
      adjacency[pairs[i].b].push_back(i);
    }
  Json optimization = Json::array();
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
      m.RebuildReferences();
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
          optimization.push_back(Optimize(m, scene.cameras, "radial", workers, 100));
          m.Filter();
          calibrated = true;
          last_global = registered;
        }
      } else if (registered >= std::max(last_global + 40, size_t(last_global * 1.35))) {
        optimization.push_back(
            Optimize(m, scene.cameras, calibrated ? "radial" : "fixed", workers, 100));
        m.Filter();
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
  optimization.push_back(Optimize(m, scene.cameras, "focal", workers, 150));
  m.Filter();
  optimization.push_back(Optimize(m, scene.cameras, "radial", workers, 150));
  m.Filter();
  optimization.push_back(Optimize(m, scene.cameras, "principal", workers, 200));
  m.Filter();
  for (uint32_t i = 0; i < graph.tracks.size(); ++i)
    if (points[i] == NoPoint || m.points[points[i]].track.empty())
      AddPoint(m, graph.tracks[i], points[i]);
  m.RebuildReferences();
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
            {"optimization", optimization}};
  fs::remove(work / "cache/unregistered.json");
  return m;
}
}  // namespace geosun

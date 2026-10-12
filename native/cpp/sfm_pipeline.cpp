#include "sfm_pipeline.h"

#include <colmap/estimators/pose.h>
#include <colmap/scene/database.h>
#include <colmap/scene/two_view_geometry.h>

#include <atomic>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>

#include "array_io.h"
#include "scene.h"

namespace {
constexpr double Pi = 3.14159265358979323846;
using Key = std::pair<std::string, std::string>;
using Node = std::pair<std::string, colmap::point2D_t>;
double Quantile(std::vector<double> values, double q) {
  if (values.empty()) return 0.;
  std::sort(values.begin(), values.end());
  const double position = q * (values.size() - 1);
  const size_t lo = size_t(position), hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (position - lo);
}
Json Statistics(const std::vector<double>& values) {
  std::vector<double> finite;
  double sum = 0.;
  for (double v : values)
    if (std::isfinite(v)) {
      finite.push_back(v);
      sum += v;
    }
  Json r = {{"count", values.size()}, {"finite_count", finite.size()}};
  if (finite.empty())
    for (const char* field : {"median", "p90", "mean", "max"})
      r[field] = nullptr;
  else {
    r["median"] = Quantile(finite, .5);
    r["p90"] = Quantile(finite, .9);
    r["mean"] = sum / finite.size();
    r["max"] = *std::max_element(finite.begin(), finite.end());
  }
  return r;
}
colmap::Camera ColmapCamera(const Camera& cam) {
  auto c = colmap::Camera::CreateFromModelName(cam.label == "L" ? 1 : 2,
                                               "OPENCV_FISHEYE", cam.params[0],
                                               cam.width, cam.height);
  c.params.assign(cam.params.begin(), cam.params.end());
  c.has_prior_focal_length = true;
  return c;
}
Eigen::Vector3d Vector(const cv::Point3d& p) { return {p.x, p.y, p.z}; }
struct Feature {
  std::vector<cv::Point2d> pixels;
  std::vector<int> bins;
};
struct Input {
  Json scene, data;
  std::map<std::string, Json> records;
  std::map<std::string, Camera> cameras;
  std::map<std::string, Feature> features;
  Input(const fs::path& scene_path, const fs::path& data_path)
      : scene(ReadJson(scene_path)), data(ReadJson(data_path)) {
    for (const auto& [label, value] : scene.at("cameras").items())
      cameras.emplace(label, Camera::FromJson(value));
    for (const auto& row : scene.at("images")) {
      const auto name = row.at("name").get<std::string>();
      records[name] = row;
      const auto arrays = geosunpre::ReadArrays(
          fs::u8path(
              data.at("features").at(name).at("path").get<std::string>()),
          {"keypoints"});
      const auto& array = arrays.at("keypoints");
      if (array.dtype != "<f4" || array.shape.size() != 2 ||
          array.shape[1] != 2)
        throw std::runtime_error("Invalid keypoints: " + name);
      auto& feature = features[name];
      const auto* p = reinterpret_cast<const float*>(array.bytes.data());
      for (size_t i = 0; i < array.shape[0]; ++i)
        feature.pixels.emplace_back(p[2 * i], p[2 * i + 1]);
      const auto rays =
          cameras.at(row.at("camera").get<std::string>()).Rays(feature.pixels);
      for (const auto& ray : rays) {
        const double theta = std::acos(std::clamp(ray.z, -1., 1.));
        const int radial = std::min(
            3, int(theta /
                   (scene.at("fov_degrees").get<double>() / 2 * Pi / 180) * 4));
        const int sector =
            int((std::atan2(ray.y, ray.x) + Pi) / (2 * Pi) * 8) % 8;
        feature.bins.push_back(radial * 8 + sector);
      }
    }
  }
};
struct Matches {
  colmap::FeatureMatches pairs;
  std::vector<float> scores;
};
Matches LoadMatches(const Json& entry) {
  auto arrays = geosunpre::ReadArrays(
      fs::u8path(entry.at("path").get<std::string>()), {"matches", "scores"});
  const auto& m = arrays.at("matches");
  const auto& s = arrays.at("scores");
  if (m.shape.size() != 2 || m.shape[1] != 2 || s.dtype != "<f4" ||
      s.Count() != m.shape[0])
    throw std::runtime_error("Invalid match arrays");
  Matches result;
  const auto* score = reinterpret_cast<const float*>(s.bytes.data());
  for (size_t i = 0; i < m.shape[0]; ++i) {
    uint64_t a, b;
    if (m.ItemSize() == 4) {
      const auto* ids = reinterpret_cast<const uint32_t*>(m.bytes.data());
      a = ids[2 * i];
      b = ids[2 * i + 1];
    } else if (m.ItemSize() == 8) {
      const auto* ids = reinterpret_cast<const uint64_t*>(m.bytes.data());
      a = ids[2 * i];
      b = ids[2 * i + 1];
    } else
      throw std::runtime_error("Invalid match integer type");
    result.pairs.emplace_back(colmap::point2D_t(a), colmap::point2D_t(b));
    result.scores.push_back(score[i]);
  }
  return result;
}
struct Fit {
  Key key;
  std::string label;
  colmap::Rigid3d pose;
  Eigen::Matrix3d essential;
  colmap::FeatureMatches inliers;
  std::vector<Eigen::Vector3d> xyz;
  std::vector<double> angles;
  std::set<int> covered[2];
  Json report;
  double closure = 0.;
  size_t triangles = 0;
};
Fit Verify(const Input& input, const Json& entry) {
  Fit fit;
  fit.key = {entry.at("a"), entry.at("b")};
  auto label = [&](const std::string& name) {
    const auto& row = input.records.at(name);
    std::ostringstream value;
    value << row.at("camera").get<std::string>() << std::setfill('0')
          << std::setw(5) << row.at("frame_index").get<int>();
    return value.str();
  };
  fit.label = label(fit.key.first) + "__" + label(fit.key.second);
  const auto raw = LoadMatches(entry);
  const auto& fa = input.features.at(fit.key.first);
  const auto& fb = input.features.at(fit.key.second);
  const auto& ca = input.cameras.at(
      input.records.at(fit.key.first).at("camera").get<std::string>());
  const auto& cb = input.cameras.at(
      input.records.at(fit.key.second).at("camera").get<std::string>());
  fit.report = {{"a", fit.key.first},
                {"b", fit.key.second},
                {"label", fit.label},
                {"success", false},
                {"raw_matches", raw.pairs.size()},
                {"verified", 0}};
  if (raw.pairs.size() < 24) {
    fit.report["reason"] = "fewer than 24 matches";
    return fit;
  }
  const auto cma = ColmapCamera(ca), cmb = ColmapCamera(cb);
  std::vector<cv::Point2d> pa, pb;
  std::vector<colmap::CamRayWithJac> ja, jb;
  for (const auto& match : raw.pairs) {
    const auto a = fa.pixels.at(match.point2D_idx1),
               b = fb.pixels.at(match.point2D_idx2);
    pa.push_back(a);
    pb.push_back(b);
    const auto ra = cma.CamRayFromImgWithJac({a.x, a.y}),
               rb = cmb.CamRayFromImgWithJac({b.x, b.y});
    if (!ra || !rb) throw std::runtime_error("Fisheye unprojection failed");
    ja.push_back(*ra);
    jb.push_back(*rb);
  }
  colmap::RANSACOptions opts;
  opts.max_error = 3.;
  opts.min_num_trials = 100;
  opts.max_num_trials = 10000;
  opts.confidence = .9999;
  opts.random_seed = 271828;
  size_t inliers = 0;
  std::vector<char> mask;
  if (!colmap::EstimateRelativePose(opts, ja, jb, &fit.pose, &inliers, &mask)) {
    fit.report["reason"] = "relative pose estimation failed";
    return fit;
  }
  const auto R = fit.pose.rotation().toRotationMatrix();
  const Eigen::Vector3d t = fit.pose.translation();
  Eigen::Matrix3d cross;
  cross << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
  fit.essential = cross * R;
  const auto ra = ca.Rays(pa), rb = cb.Rays(pb);
  std::vector<double> gradient(pa.size(), 0.);
  for (int axis = 0; axis < 2; ++axis) {
    auto ap = pa, am = pa, bp = pb, bm = pb;
    for (size_t i = 0; i < pa.size(); ++i) {
      (axis ? ap[i].y : ap[i].x) += .25;
      (axis ? am[i].y : am[i].x) -= .25;
      (axis ? bp[i].y : bp[i].x) += .25;
      (axis ? bm[i].y : bm[i].x) -= .25;
    }
    const auto aplus = ca.Rays(ap), aminus = ca.Rays(am), bplus = cb.Rays(bp),
               bminus = cb.Rays(bm);
    for (size_t i = 0; i < pa.size(); ++i) {
      const double ga = Vector(rb[i]).dot(
          fit.essential * (Vector(aplus[i]) - Vector(aminus[i])) / .5);
      const double gb = ((Vector(bplus[i]) - Vector(bminus[i])) / .5)
                            .dot(fit.essential * Vector(ra[i]));
      gradient[i] += ga * ga + gb * gb;
    }
  }
  cv::Mat points_a(2, int(pa.size()), CV_64F),
      points_b(2, int(pa.size()), CV_64F);
  for (size_t i = 0; i < pa.size(); ++i) {
    points_a.at<double>(0, int(i)) = ra[i].x / ra[i].z;
    points_a.at<double>(1, int(i)) = ra[i].y / ra[i].z;
    points_b.at<double>(0, int(i)) = rb[i].x / rb[i].z;
    points_b.at<double>(1, int(i)) = rb[i].y / rb[i].z;
  }
  cv::Matx34d A = cv::Matx34d::zeros(), B;
  for (int j = 0; j < 3; ++j) {
    A(j, j) = 1;
    for (int k = 0; k < 3; ++k) B(j, k) = R(j, k);
    B(j, 3) = t[j];
  }
  cv::Mat hom;
  cv::triangulatePoints(A, B, points_a, points_b, hom);
  std::vector<cv::Point3d> xyz_a, xyz_b;
  std::vector<bool> positive;
  for (size_t i = 0; i < pa.size(); ++i) {
    const double w = hom.at<double>(3, int(i));
    Eigen::Vector3d xyz;
    if (std::abs(w) > 1e-14)
      xyz = {hom.at<double>(0, int(i)) / w, hom.at<double>(1, int(i)) / w,
             hom.at<double>(2, int(i)) / w};
    else
      xyz.setConstant(NAN);
    const Eigen::Vector3d other = R * xyz + t;
    positive.push_back(xyz.allFinite() && xyz.z() > 0 && other.z() > 0);
    xyz_a.emplace_back(xyz.x(), xyz.y(), xyz.z());
    xyz_b.emplace_back(other.x(), other.y(), other.z());
  }
  const auto pixels_a = ca.Project(xyz_a), pixels_b = cb.Project(xyz_b);
  std::vector<double> epi, errors;
  size_t epi_count = 0;
  for (size_t i = 0; i < pa.size(); ++i) {
    const double e =
        gradient[i] > 1e-24
            ? std::abs(Vector(rb[i]).dot(fit.essential * Vector(ra[i]))) /
                  std::sqrt(gradient[i])
            : INFINITY;
    if (e <= 3.) ++epi_count;
    const double error =
        std::max(cv::norm(pixels_a[i] - pa[i]), cv::norm(pixels_b[i] - pb[i]));
    if (e <= 3. && positive[i] && error <= 6.) {
      fit.inliers.push_back(raw.pairs[i]);
      fit.xyz.push_back(Vector(xyz_a[i]));
      fit.angles.push_back(
          std::acos(std::clamp(Vector(ra[i]).dot(R.transpose() * Vector(rb[i])),
                               -1., 1.)) *
          180 / Pi);
      fit.covered[0].insert(fa.bins.at(raw.pairs[i].point2D_idx1));
      fit.covered[1].insert(fb.bins.at(raw.pairs[i].point2D_idx2));
      epi.push_back(e);
      errors.push_back(error);
    }
  }
  fit.report.update(
      {{"success", fit.inliers.size() >= 15},
       {"reason", "image-only geometric verification"},
       {"verified", fit.inliers.size()},
       {"ransac_inliers", inliers},
       {"epipolar_inliers", epi_count},
       {"fitted_epipolar_error_px", Statistics(epi)},
       {"verified_two_view_reprojection_max_px", Statistics(errors)},
       {"verified_triangulation_angle_degrees", Statistics(fit.angles)},
       {"estimator", "camera-aware E-only tangent Sampson LO-RANSAC"},
       {"native_pixel_threshold", 3.}});
  return fit;
}
Json SelectCalibration(const Input& input) {
  struct Edge {
    Json entry;
    double score;
  };
  std::map<Key, Edge> edges;
  std::map<std::string, std::vector<Key>> adjacent;
  std::map<std::pair<std::string, int>, std::vector<Key>> groups;
  std::map<std::string, int> ranks, counts, next;
  std::map<std::string, std::set<int>> coverage;
  for (const auto& row : input.scene.at("images"))
    ranks[row.at("name").get<std::string>()] =
        next[row.at("camera").get<std::string>()]++;
  for (const auto& entry : input.data.at("pairs")) {
    const Key k{entry.at("a"), entry.at("b")};
    const auto matches = LoadMatches(entry);
    size_t n = 0;
    std::set<int> occupied[2];
    for (size_t i = 0; i < matches.pairs.size(); ++i)
      if (matches.scores[i] >= .5f) {
        ++n;
        occupied[0].insert(
            input.features.at(k.first).bins.at(matches.pairs[i].point2D_idx1));
        occupied[1].insert(
            input.features.at(k.second).bins.at(matches.pairs[i].point2D_idx2));
      }
    if (n < 30) continue;
    edges[k] = {
        entry,
        n * std::sqrt(std::min(occupied[0].size(), occupied[1].size()) / 32.)};
    adjacent[k.first].push_back(k);
    adjacent[k.second].push_back(k);
    const auto a = input.records.at(k.first).at("camera").get<std::string>(),
               b = input.records.at(k.second).at("camera").get<std::string>();
    if (a == b)
      groups[{a, int(std::log2(std::max(
                     1, std::abs(ranks.at(k.first) - ranks.at(k.second)))))}]
          .push_back(k);
  }
  std::map<Key, Fit> evaluated;
  auto verify = [&](const Key& k) -> const Fit& {
    if (!evaluated.count(k))
      evaluated.emplace(k, Verify(input, edges.at(k).entry));
    return evaluated.at(k);
  };
  auto rank = [&](std::vector<Key> v) {
    std::sort(v.begin(), v.end(), [&](const Key& a, const Key& b) {
      return edges.at(a).score == edges.at(b).score
                 ? a < b
                 : edges.at(a).score > edges.at(b).score;
    });
    if (v.size() > 3) v.resize(3);
    return v;
  };
  Key seed;
  double best = -1;
  for (const auto& [group, keys] : groups)
    for (const auto& k : rank(keys)) {
      const auto& f = verify(k);
      const double angle = Quantile(f.angles, .5);
      if (f.inliers.size() >= 100 && angle >= 2) {
        const double score =
            f.inliers.size() * std::min(angle, 10.) *
            std::sqrt(std::min(f.covered[0].size(), f.covered[1].size()) / 32.);
        if (score > best || (score == best && k < seed)) {
          best = score;
          seed = k;
        }
      }
    }
  if (best < 0)
    throw std::runtime_error(
        "No calibration seed with sufficient verified support and parallax");
  std::set<std::string> selected{seed.first, seed.second};
  for (int i = 0; i < 2; ++i) {
    const auto label = input.records.at(i ? seed.second : seed.first)
                           .at("camera")
                           .get<std::string>();
    ++counts[label];
    coverage[label].insert(verify(seed).covered[i].begin(),
                           verify(seed).covered[i].end());
  }
  const size_t budget = std::min<size_t>(64, input.records.size());
  while (selected.size() < budget) {
    std::map<std::string, std::vector<Key>> frontier;
    for (const auto& name : selected)
      for (const auto& k : adjacent[name]) {
        const auto other = k.first == name ? k.second : k.first;
        if (!selected.count(other)) frontier[other].push_back(k);
      }
    std::vector<std::pair<double, std::string>> ordered;
    for (const auto& [name, keys] : frontier) {
      double score = 0;
      for (const auto& k : keys) score += edges.at(k).score;
      ordered.emplace_back(
          -score /
              (1 +
               counts[input.records.at(name).at("camera").get<std::string>()]),
          name);
    }
    std::sort(ordered.begin(), ordered.end());
    std::string winner;
    std::set<int> gain_best;
    double high = -1;
    int proposals = 0;
    for (const auto& [unused, name] : ordered) {
      const auto label = input.records.at(name).at("camera").get<std::string>();
      bool support = false;
      double score = 0;
      std::set<int> gain;
      for (const auto& k : rank(frontier.at(name))) {
        const auto& f = verify(k);
        const auto angle = Quantile(f.angles, .5);
        if (f.inliers.size() < 30 || angle < .5) continue;
        support = true;
        const int side = k.first == name ? 0 : 1;
        gain.insert(f.covered[side].begin(), f.covered[side].end());
        score += f.inliers.size() * std::min(angle, 5.);
      }
      if (!support) continue;
      size_t fresh = 0;
      for (int bin : gain)
        if (!coverage[label].count(bin)) ++fresh;
      score *= (1 + fresh / 8.) / (1 + counts[label]);
      if (score > high || (score == high && name < winner)) {
        high = score;
        winner = name;
        gain_best = gain;
      }
      if (++proposals == 8) break;
    }
    if (high < 0) break;
    selected.insert(winner);
    const auto label = input.records.at(winner).at("camera").get<std::string>();
    ++counts[label];
    coverage[label].insert(gain_best.begin(), gain_best.end());
  }
  for (const auto& [label, camera] : input.cameras)
    if (counts[label] < 3)
      throw std::runtime_error(
          "Calibration needs at least three connected verified views per "
          "camera");
  Json names = Json::array(), occupied = Json::object();
  for (const auto& row : input.scene.at("images"))
    if (selected.count(row.at("name").get<std::string>()))
      names.push_back(row.at("name"));
  for (const auto& [label, bins] : coverage) occupied[label] = bins.size();
  return {
      {"version", 1},
      {"method",
       "verified connected support with parallax and angular coverage"},
      {"image_names", names},
      {"image_budget", budget},
      {"seed",
       {{"images", {seed.first, seed.second}},
        {"verified_matches", verify(seed).inliers.size()},
        {"median_triangulation_angle_deg", Quantile(verify(seed).angles, .5)}}},
      {"images_per_camera", counts},
      {"occupied_angular_bins_per_camera", occupied},
      {"verified_proposals", evaluated.size()},
      {"input_pose_or_las_used", false}};
}
Json ModelFiles(const fs::path& root) {
  Json files = Json::object();
  for (const char* name :
       {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"})
    files[name] = FileMetadata(root / name);
  return files;
}
void FrozenJson(const fs::path& path, const Json& value) {
  if (fs::exists(path)) {
    if (ReadJson(path) != value)
      throw std::runtime_error("Frozen geometry inputs changed: " +
                               path.u8string());
  } else
    WriteJson(path, value);
}
void CommitModel(colmap::Reconstruction& rec, const fs::path& output,
                 const std::string& report_name, const Json& report) {
  if (fs::exists(output))
    throw std::runtime_error("Output already exists: " + output.u8string());
  auto pending = output;
  pending += ".pending";
  if (fs::exists(pending))
    throw std::runtime_error("Unfinished output exists: " + pending.u8string());
  fs::create_directories(pending / "sparse/0");
  rec.Write(pending / "sparse/0");
  colmap::Reconstruction check;
  check.Read(pending / "sparse/0");
  if (!check.IsValid())
    throw std::runtime_error("Invalid serialized visual model");
  auto saved = report;
  saved["model_files"] = ModelFiles(pending / "sparse/0");
  WriteJson(pending / report_name, saved);
  fs::rename(pending, output);
}
Json Contract(const Input& input, const fs::path& initial) {
  Json features = Json::object(), pairs = Json::array();
  for (const auto& [name, f] : input.features)
    features[name] = FileMetadata(fs::u8path(
        input.data.at("features").at(name).at("path").get<std::string>()));
  for (const auto& entry : input.data.at("pairs"))
    pairs.push_back({{"a", entry.at("a")},
                     {"b", entry.at("b")},
                     {"file", FileMetadata(fs::u8path(
                                  entry.at("path").get<std::string>()))}});
  return {
      {"version", 1},
      {"scene", input.scene},
      {"features", features},
      {"pairs", pairs},
      {"initial_model", initial.empty() ? Json(nullptr) : ModelFiles(initial)}};
}
void BuildGeometry(const Input& input, const fs::path& output,
                   const fs::path& initial, int workers) {
  const auto request = Contract(input, initial);
  if (fs::exists(output / "completion.json")) {
    const auto receipt = ReadJson(output / "completion.json");
    if (receipt.at("request") != request)
      throw std::runtime_error("Mapping inputs changed");
    for (const auto& [name, file] : receipt.at("files").items())
      if (FileMetadata(output / fs::u8path(name)) != file)
        throw std::runtime_error("Committed mapping input changed: " + name);
    return;
  }
  if (fs::exists(output))
    throw std::runtime_error("Uncontracted mapping inputs exist");
  auto pending = output;
  pending += ".pending";
  if (fs::exists(pending))
    throw std::runtime_error("Unfinished geometry exists: " +
                             pending.u8string());
  fs::create_directories(pending / "geometry");
  const auto& entries = input.data.at("pairs");
  std::vector<Fit> fits(entries.size());
  std::atomic<size_t> next{0}, completed{0};
  std::vector<std::future<void>> jobs;
  for (int worker = 0; worker < workers; ++worker)
    jobs.push_back(std::async(std::launch::async, [&] {
      for (;;) {
        const auto i = next.fetch_add(1);
        if (i >= entries.size()) break;
        fits[i] = Verify(input, entries.at(i));
        const auto n = ++completed;
        if (n % 100 == 0 || n == entries.size())
          std::cerr << "Native E verification " << n << '/' << entries.size()
                    << '\n';
      }
    }));
  for (auto& job : jobs) job.get();
  std::map<Key, size_t> lookup;
  std::map<std::string, std::set<std::string>> adjacent;
  std::vector<std::vector<double>> closure(fits.size());
  Json cycles = Json::array();
  for (size_t i = 0; i < fits.size(); ++i)
    if (fits[i].inliers.size() >= 15) {
      const auto& key = fits[i].key;
      lookup[{std::min(key.first, key.second),
              std::max(key.first, key.second)}] = i;
      adjacent[key.first].insert(key.second);
      adjacent[key.second].insert(key.first);
    }
  auto rotation = [&](const std::string& a,
                      const std::string& b) -> Eigen::Matrix3d {
    const auto& fit = fits.at(lookup.at({std::min(a, b), std::max(a, b)}));
    const auto R = fit.pose.rotation().toRotationMatrix();
    return fit.key.first == a ? R : Eigen::Matrix3d(R.transpose());
  };
  for (const auto& [a, neighbors] : adjacent)
    for (auto bi = neighbors.upper_bound(a); bi != neighbors.end(); ++bi)
      for (auto ci = std::next(bi); ci != neighbors.end(); ++ci)
        if (adjacent[*bi].count(*ci)) {
          const Eigen::Matrix3d R =
              rotation(*ci, a) * rotation(*bi, *ci) * rotation(a, *bi);
          const double error = Eigen::AngleAxisd(R).angle() * 180 / Pi;
          cycles.push_back(
              {{"images", {a, *bi, *ci}}, {"rotation_closure_degrees", error}});
          for (const auto& k : {Key{a, *bi}, Key{*bi, *ci}, Key{a, *ci}})
            closure.at(lookup.at(k)).push_back(error);
        }
  std::vector<size_t> accepted;
  size_t successful = 0;
  Json pair_rows = Json::array(), excluded = Json::array();
  for (size_t i = 0; i < fits.size(); ++i) {
    auto& f = fits[i];
    f.triangles = closure[i].size();
    f.closure = Quantile(closure[i], .5);
    const bool reject = f.triangles >= 3 && f.closure > 5.;
    f.report["excluded_by_rotation_consensus"] = reject;
    WriteJson(pending / "geometry" / fs::u8path(f.label + ".json"), f.report);
    if (f.inliers.size() >= 15) {
      ++successful;
      pair_rows.push_back(
          {{"images", {f.key.first, f.key.second}},
           {"triangle_count", f.triangles},
           {"rotation_closure_degrees", Statistics(closure[i])}});
      if (reject) excluded.push_back(f.label);
    }
    if (f.inliers.size() >= 30 && !reject) accepted.push_back(i);
  }
  WriteJson(pending / "rotation_cycles.json",
            {{"pairs", pair_rows}, {"triangles", cycles}});
  WriteJson(pending / "geometry_summary.json",
            {{"pairs", entries.size()},
             {"successful_E_pairs", successful},
             {"database_pairs", accepted.size()},
             {"rotation_consensus_excluded", excluded}});
  size_t seed_index = SIZE_MAX;
  double high = -1;
  if (initial.empty())
    for (size_t i : accepted) {
      const auto& f = fits[i];
      const double angle = Quantile(f.angles, .5);
      if (input.records.at(f.key.first).at("camera") !=
              input.records.at(f.key.second).at("camera") ||
          f.inliers.size() < 100 || angle < 2 || f.closure >= 2)
        continue;
      const double score = f.inliers.size() * std::min(angle, 10.);
      if (score > high) {
        high = score;
        seed_index = i;
      }
    }
  if (initial.empty() && seed_index == SIZE_MAX)
    throw std::runtime_error("No well-supported image-only fisheye seed");
  auto db = colmap::Database::Open(pending / "database.db");
  db->BeginTransaction();
  colmap::Reconstruction rec;
  for (const auto& [label, c] : input.cameras) {
    auto camera = ColmapCamera(c);
    db->WriteCamera(camera, true);
    colmap::Rig rig;
    rig.SetRigId(camera.camera_id);
    rig.AddRefSensor(camera.SensorId());
    db->WriteRig(rig, true);
    if (initial.empty()) {
      rec.AddCamera(camera);
      rec.AddRig(rig);
    }
  }
  for (const auto& row : input.scene.at("images")) {
    const auto name = row.at("name").get<std::string>();
    colmap::Image image;
    image.SetName(name);
    image.SetImageId(row.at("image_id"));
    image.SetCameraId(row.at("camera") == "L" ? 1 : 2);
    image.SetFrameId(image.ImageId());
    colmap::Frame frame;
    frame.SetFrameId(image.FrameId());
    frame.SetRigId(image.CameraId());
    frame.AddDataId(image.DataId());
    db->WriteFrame(frame, true);
    db->WriteImage(image, true);
    colmap::FeatureKeypoints points;
    for (const auto& xy : input.features.at(name).pixels)
      points.emplace_back(float(xy.x), float(xy.y));
    db->WriteKeypoints(image.ImageId(), points);
  }
  for (size_t i : accepted) {
    const auto& f = fits[i];
    colmap::TwoViewGeometry g;
    g.config = colmap::TwoViewGeometry::CALIBRATED;
    g.E = f.essential;
    g.cam2_from_cam1 = f.pose;
    g.inlier_matches = f.inliers;
    g.tri_angle = Quantile(f.angles, .5) * Pi / 180;
    const colmap::image_t a = input.records.at(f.key.first).at("image_id"),
                          b = input.records.at(f.key.second).at("image_id");
    db->WriteMatches(a, b, f.inliers);
    db->WriteTwoViewGeometry(a, b, g);
  }
  db->EndTransaction();
  db.reset();
  if (!initial.empty()) {
    rec.Read(initial);
    if (!rec.IsValid() || Cheirality(rec).at("invalid_depth") != 0)
      throw std::runtime_error("Invalid initial reconstruction");
    for (const auto& [cid, c] : rec.Cameras()) {
      const auto& reference = input.cameras.at(cid == 1 ? "L" : "R");
      if (c.ModelName() != "OPENCV_FISHEYE" || c.width != reference.width ||
          c.height != reference.height ||
          c.params != std::vector<double>(reference.params.begin(),
                                          reference.params.end()))
        throw std::runtime_error("Initial camera identity changed");
    }
    for (const auto& [id, image] : rec.Images()) {
      const auto& row = input.records.at(image.Name());
      if (!image.HasPose() || id != row.at("image_id") ||
          image.FrameId() != id ||
          image.CameraId() != (row.at("camera") == "L" ? 1 : 2))
        throw std::runtime_error("Initial exposure identity changed");
      const auto& pixels = input.features.at(image.Name()).pixels;
      if (pixels.size() != image.NumPoints2D())
        throw std::runtime_error("Initial feature indices changed");
      for (size_t i = 0; i < pixels.size(); ++i)
        if (image.Point2D(colmap::point2D_t(i)).xy !=
            Eigen::Vector2d(pixels[i].x, pixels[i].y))
          throw std::runtime_error("Initial feature coordinates changed");
    }
  } else {
    const auto& seed = fits.at(seed_index);
    for (int side = 0; side < 2; ++side) {
      const auto name = side ? seed.key.second : seed.key.first;
      const auto& row = input.records.at(name);
      colmap::Image image;
      image.SetName(name);
      image.SetImageId(row.at("image_id"));
      image.SetCameraId(row.at("camera") == "L" ? 1 : 2);
      image.SetFrameId(image.ImageId());
      std::vector<Eigen::Vector2d> points;
      for (const auto& p : input.features.at(name).pixels)
        points.emplace_back(p.x, p.y);
      image.SetPoints2D(points);
      colmap::Frame frame;
      frame.SetFrameId(image.FrameId());
      frame.SetRigId(image.CameraId());
      frame.AddDataId(image.DataId());
      frame.SetRigFromWorld(side ? seed.pose : colmap::Rigid3d());
      rec.AddFrame(frame);
      rec.AddImage(image);
      rec.RegisterFrame(frame.FrameId());
    }
    const auto rgb = ReadImage(
        fs::u8path(
            input.records.at(seed.key.first).at("source").get<std::string>()),
        cv::IMREAD_COLOR);
    for (size_t i = 0; i < seed.inliers.size(); ++i) {
      if (!seed.xyz[i].allFinite() || seed.angles[i] < 1) continue;
      const auto& m = seed.inliers[i];
      colmap::Track track;
      track.AddElement(input.records.at(seed.key.first).at("image_id"),
                       m.point2D_idx1);
      track.AddElement(input.records.at(seed.key.second).at("image_id"),
                       m.point2D_idx2);
      const auto xy =
          input.features.at(seed.key.first).pixels.at(m.point2D_idx1);
      const auto bgr =
          rgb.at<cv::Vec3b>(int(std::floor(xy.y)), int(std::floor(xy.x)));
      rec.AddPoint3D(seed.xyz[i], track, {bgr[2], bgr[1], bgr[0]});
    }
  }
  fs::create_directories(pending / "seed");
  rec.Write(pending / "seed");
  WriteJson(pending / "seed.json",
            {{"pair", initial.empty() ? Json(fits.at(seed_index).label)
                                      : Json(nullptr)},
             {"registered", rec.NumRegImages()},
             {"points", rec.NumPoints3D()},
             {"pose_prior_used", false}});
  Json files = Json::object();
  for (const char* name : {"database.db", "seed.json", "geometry_summary.json",
                           "rotation_cycles.json"})
    files[name] = FileMetadata(pending / name);
  const auto seed_files = ModelFiles(pending / "seed");
  for (const auto& [name, value] : seed_files.items())
    files["seed/" + name] = value;
  WriteJson(pending / "completion.json",
            {{"request", request}, {"files", files}});
  fs::rename(pending, output);
}
Json Coverage(const colmap::Reconstruction& rec) {
  Json rows = Json::object(), components = Json::array();
  std::map<colmap::image_t, std::set<colmap::image_t>> adjacent;
  for (auto id : rec.RegImageIds()) adjacent[id];
  for (const auto& [id, p] : rec.Points3D())
    for (const auto& a : p.track.Elements())
      for (const auto& b : p.track.Elements())
        if (a.image_id != b.image_id) adjacent[a.image_id].insert(b.image_id);
  size_t min_observations = SIZE_MAX, min_cells = SIZE_MAX;
  for (const auto& [id, im] : rec.Images())
    if (im.HasPose()) {
      const auto& c = rec.Camera(im.CameraId());
      size_t n = 0;
      std::set<std::pair<int, int>> cells;
      for (const auto& p : im.Points2D())
        if (p.HasPoint3D()) {
          ++n;
          cells.emplace(int(std::floor(p.xy.x() / c.width * 8)),
                        int(std::floor(p.xy.y() / c.height * 8)));
        }
      rows[im.Name()] = {{"observations", n},
                         {"occupied_8x8_cells", cells.size()}};
      min_observations = std::min(min_observations, n);
      min_cells = std::min(min_cells, cells.size());
    }
  std::set<colmap::image_t> remaining;
  for (const auto& [id, neighbors] : adjacent) remaining.insert(id);
  while (!remaining.empty()) {
    std::set<colmap::image_t> reached{*remaining.begin()};
    std::vector<colmap::image_t> pending{*remaining.begin()};
    while (!pending.empty()) {
      const auto id = pending.back();
      pending.pop_back();
      for (auto n : adjacent[id])
        if (reached.insert(n).second) pending.push_back(n);
    }
    for (auto id : reached) remaining.erase(id);
    components.push_back(reached);
  }
  return {
      {"per_image", rows},
      {"image_components", components},
      {"min_observations", min_observations == SIZE_MAX ? 0 : min_observations},
      {"min_occupied_cells", min_cells == SIZE_MAX ? 0 : min_cells}};
}
Json FilterTracks(colmap::Reconstruction& rec, const Json& data,
                  const fs::path& directory) {
  std::map<Node, colmap::point3D_t> mapped;
  std::map<std::string, colmap::image_t> ids;
  std::map<Node, std::set<Node>> adjacent;
  for (const auto& [id, im] : rec.Images()) {
    ids[im.Name()] = id;
    for (colmap::point2D_t i = 0; i < im.NumPoints2D(); ++i)
      if (im.Point2D(i).HasPoint3D())
        mapped[{im.Name(), i}] = im.Point2D(i).point3D_id;
  }
  size_t edges = 0;
  for (const auto& entry : data.at("pairs")) {
    const auto raw = LoadMatches(entry);
    const std::string a = entry.at("a"), b = entry.at("b");
    for (size_t i = 0; i < raw.pairs.size(); ++i)
      if (raw.scores[i] >= .5f) {
        const Node x{a, raw.pairs[i].point2D_idx1},
            y{b, raw.pairs[i].point2D_idx2};
        if (mapped.count(x) && mapped.count(y) &&
            mapped.at(x) == mapped.at(y)) {
          adjacent[x].insert(y);
          adjacent[y].insert(x);
          ++edges;
        }
      }
  }
  const Json before = {{"points", rec.NumPoints3D()},
                       {"observations", rec.ComputeNumObservations()},
                       {"coverage", Coverage(rec)}};
  Json decisions = Json::array();
  size_t triangles = 0, split = 0;
  std::vector<colmap::point3D_t> points;
  for (const auto& [id, p] : rec.Points3D()) points.push_back(id);
  std::sort(points.begin(), points.end());
  for (auto pid : points) {
    const auto point = rec.Point3D(pid);
    std::set<Node> nodes;
    for (const auto& e : point.track.Elements())
      nodes.emplace(rec.Image(e.image_id).Name(), e.point2D_idx);
    std::map<Node, std::set<Node>> graph;
    size_t n = 0;
    for (const auto& a : nodes) {
      std::vector<Node> neighbors;
      for (const auto& b : adjacent[a])
        if (b > a && nodes.count(b)) neighbors.push_back(b);
      for (size_t i = 0; i < neighbors.size(); ++i)
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
          const auto& b = neighbors[i];
          const auto& c = neighbors[j];
          if (a.first == b.first || a.first == c.first || b.first == c.first ||
              !adjacent[b].count(c))
            continue;
          ++n;
          for (const auto& [x, y] :
               {std::pair<Node, Node>{a, b}, {a, c}, {b, c}}) {
            graph[x].insert(y);
            graph[y].insert(x);
          }
        }
    }
    triangles += n;
    std::set<Node> remaining;
    for (const auto& [node, neighbors] : graph) remaining.insert(node);
    std::vector<std::set<Node>> components;
    while (!remaining.empty()) {
      std::set<Node> reached{*remaining.begin()};
      std::vector<Node> pending{*remaining.begin()};
      while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        for (const auto& neighbor : graph[node])
          if (reached.insert(neighbor).second) pending.push_back(neighbor);
      }
      for (const auto& node : reached) remaining.erase(node);
      components.push_back(reached);
    }
    std::sort(components.begin(), components.end());
    if (components.size() == 1 && components[0] == nodes) continue;
    rec.DeletePoint3D(pid);
    Json replacements = Json::array();
    for (const auto& component : components)
      if (component.size() >= 3) {
        colmap::Track track;
        for (const auto& node : component)
          track.AddElement(ids.at(node.first), node.second);
        const auto id = rec.AddPoint3D(point.xyz, track, point.color);
        replacements.push_back({{"point_id", id}, {"observations", component}});
      }
    if (replacements.size() > 1) ++split;
    decisions.push_back({{"original_point_id", pid},
                         {"original_observations", nodes},
                         {"replacement_components", replacements},
                         {"triangles", n}});
  }
  const auto coverage = Coverage(rec);
  const bool admissible =
      coverage.at("image_components").size() == 1 &&
      coverage.at("min_observations").get<size_t>() >= 100 &&
      coverage.at("min_occupied_cells").get<size_t>() >= 12;
  Json report = {{"before", before},
                 {"after",
                  {{"points", rec.NumPoints3D()},
                   {"observations", rec.ComputeNumObservations()},
                   {"coverage", coverage}}},
                 {"changed_original_tracks", decisions.size()},
                 {"split_original_tracks", split},
                 {"supported_triangles", triangles},
                 {"eligible_matching_edges", edges},
                 {"coverage_admissible", admissible}};
  fs::create_directories(directory);
  WriteJson(directory / "track_filter.json", report);
  WriteJson(directory / "track_decisions.json", decisions);
  if (!admissible) throw std::runtime_error("Training track coverage rejected");
  return report;
}
Json Quality(const colmap::Reconstruction& rec, const Json& scene,
             const Json& settings = Json::object()) {
  if (!rec.IsValid() || rec.NumRegImages() != scene.at("images").size() ||
      rec.Cameras().size() != 2)
    throw std::runtime_error("Visual model is incomplete");
  std::map<std::string, Json> expected;
  for (const auto& row : scene.at("images"))
    expected[row.at("name").get<std::string>()] = row;
  const auto depth = Cheirality(rec), coverage = Coverage(rec);
  if (depth.at("invalid_depth") != 0 ||
      coverage.at("image_components").size() != 1 ||
      coverage.at("min_observations").get<size_t>() < 100 ||
      coverage.at("min_occupied_cells").get<size_t>() < 12)
    throw std::runtime_error("Visual depth or coverage rejected");
  std::vector<double> all;
  std::map<std::string, std::vector<double>> by_camera;
  Json health = Json::object();
  for (const auto& [id, c] : rec.Cameras()) {
    Camera cam;
    cam.label = id == 1 ? "L" : "R";
    cam.width = int(c.width);
    cam.height = int(c.height);
    if (c.ModelName() != "OPENCV_FISHEYE" || c.params.size() != 8)
      throw std::runtime_error("Visual camera model changed");
    std::copy(c.params.begin(), c.params.end(), cam.params.begin());
    cam.Radius();
    health[cam.label] = {{"params", cam.params}, {"monotonic_in_160deg", true}};
  }
  for (const auto& [id, image] : rec.Images())
    if (image.HasPose()) {
      const auto& row = expected.at(image.Name());
      if (id != row.at("image_id") || image.FrameId() != id ||
          image.CameraId() != (row.at("camera") == "L" ? 1 : 2))
        throw std::runtime_error("Visual exposure identity changed");
      const auto& c = rec.Camera(image.CameraId());
      Camera cam;
      cam.label = image.CameraId() == 1 ? "L" : "R";
      cam.width = int(c.width);
      cam.height = int(c.height);
      std::copy(c.params.begin(), c.params.end(), cam.params.begin());
      std::vector<cv::Point3d> xyz;
      std::vector<Eigen::Vector2d> observed;
      const auto pose = image.CamFromWorld();
      for (const auto& point : image.Points2D())
        if (point.HasPoint3D()) {
          const Eigen::Vector3d p = pose * rec.Point3D(point.point3D_id).xyz;
          xyz.emplace_back(p.x(), p.y(), p.z());
          observed.push_back(point.xy);
        }
      const auto projected = cam.Project(xyz);
      for (size_t i = 0; i < projected.size(); ++i) {
        const double error =
            (Eigen::Vector2d(projected[i].x, projected[i].y) - observed[i])
                .norm();
        all.push_back(error);
        by_camera[cam.label].push_back(error);
      }
    }
  Json per_camera = Json::object();
  for (const auto& [label, values] : by_camera)
    per_camera[label] = Statistics(values);
  Json report = {{"accuracy_scope", "fitted_observations"},
                 {"registered", rec.NumRegImages()},
                 {"points", rec.NumPoints3D()},
                 {"observations", rec.ComputeNumObservations()},
                 {"cheirality", depth},
                 {"coverage", coverage},
                 {"camera_health", health},
                 {"training_reprojection", Statistics(all)},
                 {"training_reprojection_by_camera", per_camera}};
  const double median = settings.value("quality_max_median_px", 2.),
               p90 = settings.value("quality_max_p90_px", 4.);
  Json checks = per_camera;
  checks["all"] = report.at("training_reprojection");
  for (const auto& [label, stats] : checks.items())
    if (stats.at("count") == 0 ||
        stats.at("count") != stats.at("finite_count") ||
        stats.at("median").get<double>() > median ||
        stats.at("p90").get<double>() > p90)
      throw VisualAcceptanceRejected(
          "Fitted reprojection acceptance rejected: " + label);
  report["acceptance"] = {{"status", "passed"},
                          {"limits", {{"median_px", median}, {"p90_px", p90}}}};
  return report;
}
void VisualStages(const Input& input, const fs::path& directory,
                  const fs::path& initial, bool calibration, int workers,
                  const Json& settings) {
  const auto geometry = directory / "mapping_inputs";
  BuildGeometry(input, geometry, initial, workers);
  const auto mapping = directory / "mapping";
  if (!fs::exists(mapping / "mapping.json"))
    RunMap(Arguments(
        {"geosunpre-native", "map", "--database",
         (geometry / "database.db").u8string(), "--seed",
         (geometry / "seed").u8string(), "--images",
         (fs::u8path(input.scene.at("input_root").get<std::string>()) / "Image")
             .u8string(),
         "--output", mapping.u8string(), "--workers",
         std::to_string(workers)}));
  colmap::Reconstruction rec;
  rec.Read(mapping / "sparse/0");
  if (rec.NumRegImages() != input.records.size())
    throw std::runtime_error("Mapping did not register all selected images");
  Json base = Json::object();
  for (const auto& [id, c] : rec.Cameras()) base[std::to_string(id)] = c.params;
  std::vector<std::string> stages =
      calibration ? std::vector<std::string>{"fixed", "focal_low_radial",
                                             "all_radial", "principal"}
                  : std::vector<std::string>{"fixed", "tracks", "principal"};
  fs::path previous = mapping / "sparse/0";
  for (const auto& stage : stages) {
    const auto output = directory / "stages" / stage;
    const auto input_files = ModelFiles(previous);
    const auto report_name = "ba.json";
    if (fs::exists(output / report_name)) {
      const auto report = ReadJson(output / report_name);
      if (report.at("input_model_files") != input_files ||
          report.at("model_files") != ModelFiles(output / "sparse/0"))
        throw std::runtime_error("Committed visual stage changed: " + stage);
      rec.Read(output / "sparse/0");
    } else {
      Json report;
      Json filtration;
      if (stage == "tracks")
        filtration = FilterTracks(rec, input.data, directory / "track_support");
      {
        Arguments options({"geosunpre-native", "ba", "--stage",
                           stage == "tracks" ? "fixed" : stage, "--workers",
                           std::to_string(workers)});
        if (!calibration) options.values["--protect-depth"] = "true";
        const auto archive = directory / "principal_iteration_limit";
        const bool resume =
            stage == "principal" && fs::exists(archive / "ba.json");
        if (resume) {
          report = ReadJson(archive / "ba.json");
          if (report.at("input_model_files") != input_files ||
              report.at("model_files") != ModelFiles(archive / "sparse/0"))
            throw std::runtime_error("Principal continuation inputs changed");
          rec.Read(archive / "sparse/0");
        } else
          report = SolveBA(rec, options, base);
        if (stage == "principal" &&
            report.at("termination_type") ==
                "BundleAdjustmentTerminationType.NO_CONVERGENCE") {
          if (!resume) {
            report["input_model_files"] = input_files;
            CommitModel(rec, archive, "ba.json", report);
          }
          if (!calibration) options.values["--iterations"] = "40";
          auto continuation = SolveBA(rec, options, base);
          continuation["continuation"] = {
              {"reason", "Continue from the solver iteration limit"},
              {"previous_solver", report.at("solver")},
              {"previous_elapsed_seconds", report.at("elapsed_seconds")}};
          report = continuation;
        }
        if (stage == "tracks") report["track_filter"] = filtration;
      }
      report["input_model_files"] = input_files;
      CommitModel(rec, output, report_name, report);
    }
    previous = output / "sparse/0";
  }
  if (!calibration) {
    const auto bounds =
        ReadJson(directory / "stages/principal/ba.json").at("cameras");
    for (const auto& [id, health] : bounds.items())
      if (!health.at("parameters_at_bounds").empty())
        throw std::runtime_error(
            "Camera parameters reached optimization bounds: " + id);
    auto quality = Quality(rec, input.scene, settings);
    quality["camera_parameter_bounds"] = bounds;
    WriteJson(directory / "quality.json", quality);
  }
  WriteJson(directory / "summary.json",
            {{"registered", rec.NumRegImages()},
             {"points", rec.NumPoints3D()},
             {"observations", rec.ComputeNumObservations()},
             {"runtime", "C++"},
             {"pose_prior_used", false}});
}
}  // namespace
Json VisualQuality(const colmap::Reconstruction& rec, const Json& scene,
                   const Json& settings) {
  return Quality(rec, scene, settings);
}
void RunGeometry(const Arguments& args) {
  args.Validate({"--manifest", "--frontend", "--output", "--initial-model",
                 "--workers", "--select", "--blas-threads"});
  const Input input(args.Path("--manifest"), args.Path("--frontend"));
  if (args.Get("--select") == "true") {
    const auto result = SelectCalibration(input);
    fs::create_directories(args.Path("--output"));
    WriteJson(args.Path("--output") / "selection.json", result);
    std::cout << result.dump() << '\n';
  } else
    BuildGeometry(input, args.Path("--output"),
                  args.Get("--initial-model").empty()
                      ? fs::path()
                      : args.Path("--initial-model"),
                  args.Integer("--workers", 4));
}
void RunVisualPipeline(const Arguments& args) {
  args.Validate({"--work", "--workers", "--blas-threads", "--stage"});
  const auto work = args.Path("--work");
  const Input global(work / "sfm/manifest.json", work / "sfm/frontend.json");
  const auto calibration = work / "calibration";
  fs::create_directories(calibration);
  const auto settings =
      fs::exists(work / "pipeline_config.json")
          ? ReadJson(work / "pipeline_config.json").at("settings")
          : Json::object();
  const auto selection = fs::exists(calibration / "selection.json")
                             ? ReadJson(calibration / "selection.json")
                             : SelectCalibration(global);
  FrozenJson(calibration / "selection.json", selection);
  const auto names = selection.at("image_names").get<std::set<std::string>>();
  Json scene = global.scene, data = global.data, images = Json::array(),
       pairs = Json::array(), features = Json::object();
  for (const auto& row : scene.at("images"))
    if (names.count(row.at("name").get<std::string>())) images.push_back(row);
  for (const auto& entry : data.at("pairs"))
    if (names.count(entry.at("a").get<std::string>()) &&
        names.count(entry.at("b").get<std::string>()))
      pairs.push_back(entry);
  for (const auto& name : names) features[name] = data.at("features").at(name);
  scene["images"] = images;
  data["pairs"] = pairs;
  data["features"] = features;
  FrozenJson(calibration / "manifest.json", scene);
  FrozenJson(calibration / "frontend.json", data);
  const Input local(calibration / "manifest.json",
                    calibration / "frontend.json");
  const int workers = args.Integer("--workers", 4);
  VisualStages(local, calibration, {}, true, workers, settings);
  if (args.Get("--stage") == "calibration") return;
  colmap::Reconstruction calibrated;
  calibrated.Read(calibration / "stages/principal/sparse/0");
  scene = global.scene;
  for (const auto& [id, c] : calibrated.Cameras()) {
    const std::string label = id == 1 ? "L" : "R";
    scene["cameras"][label]["params"] = c.params;
  }
  FrozenJson(work / "sfm/calibrated_manifest.json", scene);
  const Input production(work / "sfm/calibrated_manifest.json",
                         work / "sfm/frontend.json");
  VisualStages(production, work / "visual",
               calibration / "stages/principal/sparse/0", false, workers,
               settings);
}
void RunTrackFilter(const Arguments& args) {
  args.Validate({"--input", "--frontend", "--output", "--blas-threads"});
  colmap::Reconstruction rec;
  rec.Read(args.Path("--input"));
  const auto report = FilterTracks(rec, ReadJson(args.Path("--frontend")),
                                   fs::path(args.Path("--output").native() +
                                            fs::path(".diagnostics").native()));
  CommitModel(rec, args.Path("--output"), "track_filter.json", report);
  std::cout << report.dump() << '\n';
}
void RunQuality(const Arguments& args) {
  args.Validate({"--input", "--manifest", "--output", "--blas-threads"});
  colmap::Reconstruction rec;
  rec.Read(args.Path("--input"));
  const auto report = Quality(rec, ReadJson(args.Path("--manifest")));
  WriteJson(args.Path("--output"), report);
  std::cout << report.dump() << '\n';
}

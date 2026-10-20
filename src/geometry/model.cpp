#include "geometry/model.h"
#include "io/cache.h"
#include <set>
#include <cmath>

namespace geosun {
size_t Reconstruction::Registered() const {
  return std::count_if(images.begin(), images.end(), [](const auto& i) { return i.registered; });
}
size_t Reconstruction::Observations() const {
  size_t n = 0;
  for (const auto& p : points)
    n += p.track.size();
  return n;
}
Json Reconstruction::Registration(const Scene& scene) const {
  Json missing = Json::array();
  for (uint32_t id = 0; id < images.size(); ++id)
    if (!images[id].registered)
      missing.push_back({{"image_index", id},
                         {"name", scene.images[id].name},
                         {"camera", images[id].camera}});
  return {{"complete", missing.empty()},
          {"selected_images", images.size()},
          {"registered_images", Registered()},
          {"unregistered_images", missing}};
}
void Reconstruction::Apply(const Similarity& s) {
  for (auto& p : points)
    p.xyz = s.Apply(p.xyz);
  for (auto& image : images)
    if (image.registered) {
      const Eigen::Quaterniond q = image.pose.Rotation() * s.rotation.conjugate();
      const Eigen::Vector3d center = s.Apply(image.pose.Center());
      image.pose.Set(q, -(q * center));
    }
  gauge_baseline *= s.scale;
}
void Reconstruction::RebuildReferences() {
  for (auto& image : images)
    image.points.assign(image.pixels.size(), NoPoint);
  for (uint64_t id = 0; id < points.size(); ++id) {
    std::set<uint32_t> seen;
    for (const auto& o : points[id].track) {
      if (o.image >= images.size() || o.feature >= images[o.image].pixels.size() ||
          !images[o.image].registered || !seen.insert(o.image).second ||
          images[o.image].points[o.feature] != NoPoint)
        throw std::runtime_error("Invalid or conflicting reconstruction track");
      images[o.image].points[o.feature] = id;
    }
  }
}
void Reconstruction::Filter(double threshold, double min_angle) {
  const double cosine = std::cos(min_angle * 3.14159265358979323846 / 180);
  for (auto& p : points) {
    std::vector<Observation> track;
    std::vector<Eigen::Vector3d> directions;
    double error = 0;
    for (const auto& o : p.track) {
      const auto& image = images[o.image];
      const auto xyz = image.pose.Apply(p.xyz);
      const auto& c = cameras[image.camera];
      const double e = (c.Project(xyz) - image.pixels[o.feature]).norm();
      if (xyz.z() > 1e-9 && xyz.allFinite() && std::isfinite(e) && e <= threshold) {
        track.push_back(o);
        error += e;
        directions.push_back((p.xyz - image.pose.Center()).normalized());
      }
    }
    bool parallax = false;
    for (size_t i = 0; i < directions.size() && !parallax; ++i)
      for (size_t j = i + 1; j < directions.size(); ++j)
        if (directions[i].dot(directions[j]) <= cosine) {
          parallax = true;
          break;
        }
    if (track.size() < 2 || !parallax)
      p.track.clear();
    else {
      p.track = std::move(track);
      p.error = error / p.track.size();
    }
  }
  RebuildReferences();
}
Json Reconstruction::Quality(const Scene& scene, double median_limit, double p90_limit) const {
  Json rows = Json::array(), failures = Json::array();
  std::vector<double> errors;
  std::array<std::vector<double>, 2> by_camera;
  std::array<size_t, 2> camera_images{};
  std::array<std::array<std::vector<double>, 4>, 2> by_angle;
  uint64_t invalid = 0;
  size_t active = 0;
  std::vector<double> lengths;
  size_t long_tracks = 0, extended_tracks = 0;
  for (const auto& p : points)
    if (!p.track.empty()) {
      ++active;
      lengths.push_back(double(p.track.size()));
      long_tracks += p.track.size() >= 3;
      extended_tracks += p.track.size() >= 8;
    }
  for (uint32_t id = 0; id < images.size(); ++id) {
    const auto& image = images[id];
    const auto& c = cameras[image.camera];
    std::vector<double> e;
    std::set<int> cells;
    if (!image.registered) {
      rows.push_back({{"image_index", id},
                      {"name", scene.images[id].name},
                      {"registered", false},
                      {"observations", 0},
                      {"occupied_cells", 0},
                      {"reprojection_px", Statistics(e)}});
      continue;
    }
    ++camera_images[image.camera];
    for (size_t k = 0; k < image.points.size(); ++k)
      if (image.points[k] != NoPoint) {
        const auto xyz = image.pose.Apply(points.at(image.points[k]).xyz);
        const double r = (c.Project(xyz) - image.pixels[k]).norm();
        if (!xyz.allFinite() || xyz.z() <= 0 || !std::isfinite(r)) {
          ++invalid;
          continue;
        }
        e.push_back(r);
        errors.push_back(r);
        by_camera[image.camera].push_back(r);
        const auto& xy = image.pixels[k];
        const double angle =
            std::acos(std::clamp(scene.cameras[image.camera].Ray(xy).z(), -1., 1.));
        const int bin = std::clamp(int(angle * 180 / 3.14159265358979323846 / 20), 0, 3);
        by_angle[image.camera][bin].push_back(r);
        cells.insert(std::clamp(int(xy.x() / c.width * 8), 0, 7) +
                     8 * std::clamp(int(xy.y() / c.height * 8), 0, 7));
      }
    rows.push_back({{"image_index", id},
                    {"name", scene.images[id].name},
                    {"registered", true},
                    {"observations", e.size()},
                    {"occupied_cells", cells.size()},
                    {"reprojection_px", Statistics(e)}});
    if (e.size() < 32 || cells.size() < 8)
      failures.push_back("Insufficient image coverage: " + scene.images[id].name);
    if (!e.empty() && Quantile(e, .9) > 2 * p90_limit)
      failures.push_back("Image reprojection error exceeds limit: " + scene.images[id].name);
  }
  if (invalid)
    failures.push_back("Nonpositive depth or nonfinite projection");
  if (errors.empty() || Quantile(errors, .5) > median_limit || Quantile(errors, .9) > p90_limit)
    failures.push_back("Global reprojection error exceeds limit");
  Json calibration = Json::object();
  for (int cid = 0; cid < 2; ++cid) {
    cameras[cid].Radius();
    const auto& base = scene.cameras[cid].params;
    Json bins = Json::array();
    for (int bin = 0; bin < 4; ++bin)
      bins.push_back({{"angle_degrees", {bin * 20, (bin + 1) * 20}},
                      {"reprojection_px", Statistics(by_angle[cid][bin])}});
    calibration[cameras[cid].label] = {
        {"registered_images", camera_images[cid]},
        {"initial_params", base},
        {"params", cameras[cid].params},
        {"principal_shift_px",
         {cameras[cid].params[2] - base[2], cameras[cid].params[3] - base[3]}},
        {"reprojection_by_angle", bins}};
    const auto bounds = scene.cameras[cid].CalibrationBounds();
    for (int j = 0; j < 8; ++j)
      if (std::abs(cameras[cid].params[j] - base[j]) >= bounds[j] * (1 - 1e-4))
        failures.push_back("Camera parameter reached calibration bound: " + cameras[cid].label +
                           "/" + std::to_string(j));
    if (camera_images[cid] &&
        (by_camera[cid].empty() || Quantile(by_camera[cid], .5) > median_limit ||
         Quantile(by_camera[cid], .9) > p90_limit))
      failures.push_back("Camera reprojection error exceeds limit: " + cameras[cid].label);
  }
  return {{"passed", failures.empty()},
          {"registration", Registration(scene)},
          {"calibration", calibration},
          {"failures", failures},
          {"registered_images", Registered()},
          {"expected_images", scene.images.size()},
          {"points", active},
          {"track_lengths", Statistics(lengths)},
          {"tracks_at_least_3", long_tracks},
          {"tracks_at_least_8", extended_tracks},
          {"observations", errors.size()},
          {"invalid_depth", invalid},
          {"reprojection_px", Statistics(errors)},
          {"cameras", {{"L", Statistics(by_camera[0])}, {"R", Statistics(by_camera[1])}}},
          {"images", rows},
          {"limits", {{"median_px", median_limit}, {"p90_px", p90_limit}}}};
}
void WriteModel(const fs::path& path, const Reconstruction& model) {
  BinaryWriter out(path);
  out.String("GEOSUN_RECONSTRUCTION_1");
  out.String(Json({model.cameras[0].ToJson(), model.cameras[1].ToJson()}).dump());
  out.Value(model.gauge_image);
  out.Value(model.scale_image);
  out.Value(model.gauge_baseline);
  out.Value(uint64_t(model.images.size()));
  out.Value(uint64_t(model.points.size()));
  for (const auto& image : model.images) {
    out.Value(image.camera);
    out.Value(uint8_t(image.registered));
    out.Value(image.pose.params);
    out.Value(uint64_t(image.pixels.size()));
    for (const auto& xy : image.pixels) {
      out.Value(xy.x());
      out.Value(xy.y());
    }
  }
  for (const auto& p : model.points) {
    for (int i = 0; i < 3; ++i)
      out.Value(p.xyz[i]);
    out.Value(p.color);
    out.Value(p.error);
    out.Value(uint64_t(p.track.size()));
    for (const auto& o : p.track) {
      out.Value(o.image);
      out.Value(o.feature);
    }
  }
  out.Commit();
}
Reconstruction ReadModel(const fs::path& path) {
  BinaryReader in(path);
  if (in.String() != "GEOSUN_RECONSTRUCTION_1")
    throw std::runtime_error("Unsupported reconstruction cache");
  Reconstruction m;
  const auto c = Json::parse(in.String());
  for (int i = 0; i < 2; ++i)
    m.cameras[i] = Camera::FromJson(c[i]);
  m.gauge_image = in.Value<uint32_t>();
  m.scale_image = in.Value<uint32_t>();
  m.gauge_baseline = in.Value<double>();
  const auto ni = in.Value<uint64_t>(), np = in.Value<uint64_t>();
  if (ni > 1000000 || np > fs::file_size(path) / 43)
    throw std::runtime_error("Invalid reconstruction counts");
  m.images.resize(ni);
  m.points.resize(np);
  for (auto& image : m.images) {
    image.camera = in.Value<uint32_t>();
    image.registered = in.Value<uint8_t>() != 0;
    image.pose.params = in.Value<std::array<double, 7>>();
    const auto n = in.Value<uint64_t>();
    if (image.camera >= 2 || n > 12288)
      throw std::runtime_error("Invalid reconstruction image");
    image.pixels.resize(n);
    for (auto& xy : image.pixels) {
      const double x = in.Value<double>(), y = in.Value<double>();
      xy = {x, y};
    }
  }
  for (auto& p : m.points) {
    for (int i = 0; i < 3; ++i)
      p.xyz[i] = in.Value<double>();
    p.color = in.Value<std::array<uint8_t, 3>>();
    p.error = in.Value<double>();
    const auto n = in.Value<uint64_t>();
    if (n > ni || !p.xyz.allFinite() || !std::isfinite(p.error))
      throw std::runtime_error("Invalid reconstruction point");
    p.track.resize(n);
    for (auto& o : p.track) {
      o.image = in.Value<uint32_t>();
      o.feature = in.Value<uint32_t>();
    }
  }
  in.End();
  m.RebuildReferences();
  return m;
}
}  // namespace geosun

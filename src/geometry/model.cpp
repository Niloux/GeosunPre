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
  constexpr double to_degrees = 180 / 3.14159265358979323846;
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
    double maximum = 0;
    for (size_t i = 0; i < directions.size(); ++i)
      for (size_t j = i + 1; j < directions.size(); ++j)
        maximum = std::max(
            maximum, std::acos(std::clamp(directions[i].dot(directions[j]), -1., 1.)) * to_degrees);
    if (track.size() < 2 || maximum < min_angle)
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
  uint64_t invalid = 0;
  size_t active = 0;
  for (const auto& p : points)
    if (!p.track.empty())
      ++active;
  for (uint32_t id = 0; id < images.size(); ++id) {
    const auto& image = images[id];
    const auto& c = cameras[image.camera];
    std::vector<double> e;
    std::set<int> cells;
    if (!image.registered) {
      failures.push_back("Unregistered image: " + scene.images[id].name);
      continue;
    }
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
        cells.insert(std::clamp(int(xy.x() / c.width * 8), 0, 7) +
                     8 * std::clamp(int(xy.y() / c.height * 8), 0, 7));
      }
    rows.push_back({{"name", scene.images[id].name},
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
  for (int cid = 0; cid < 2; ++cid) {
    cameras[cid].Radius();
    const auto& base = scene.cameras[cid].params;
    const std::array<double, 8> bounds = {base[0] * .1, base[1] * .1, 50, 50, .05, .02, .01, .005};
    for (int j = 0; j < 8; ++j)
      if (std::abs(cameras[cid].params[j] - base[j]) >= bounds[j] * (1 - 1e-4))
        failures.push_back("Camera parameter reached calibration bound: " + cameras[cid].label +
                           "/" + std::to_string(j));
    if (by_camera[cid].empty() || Quantile(by_camera[cid], .5) > median_limit ||
        Quantile(by_camera[cid], .9) > p90_limit)
      failures.push_back("Camera reprojection error exceeds limit: " + cameras[cid].label);
  }
  return {{"passed", failures.empty()},
          {"failures", failures},
          {"registered_images", Registered()},
          {"expected_images", scene.images.size()},
          {"points", active},
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

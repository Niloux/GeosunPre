#include "io/colmap.h"
#include "io/cache.h"
#include "io/las.h"
#include "frontend/frontend.h"
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <unordered_map>

namespace geosun {
namespace {
void PoseOut(BinaryWriter& out, const Pose& p) {
  out.Value(p.params[3]);
  for (int i = 0; i < 3; ++i)
    out.Value(p.params[i]);
  for (int i = 4; i < 7; ++i)
    out.Value(p.params[i]);
}
Pose PoseIn(BinaryReader& in) {
  Pose p;
  p.params[3] = in.Value<double>();
  for (int i = 0; i < 3; ++i)
    p.params[i] = in.Value<double>();
  for (int i = 4; i < 7; ++i)
    p.params[i] = in.Value<double>();
  for (double x : p.params)
    if (!std::isfinite(x))
      throw std::runtime_error("Nonfinite COLMAP pose");
  if (std::abs(p.Rotation().norm() - 1) > 1e-6)
    throw std::runtime_error("Nonunit COLMAP quaternion");
  return p;
}
void WriteColmap(const fs::path& path, const Scene& scene, const Reconstruction& model) {
  BinaryWriter cameras(path / "cameras.bin");
  cameras.Value(uint64_t(2));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto& c = model.cameras[i];
    cameras.Value(i + 1);
    cameras.Value(int32_t(5));
    cameras.Value(uint64_t(c.width));
    cameras.Value(uint64_t(c.height));
    for (double x : c.params)
      cameras.Value(x);
  }
  cameras.Commit();
  BinaryWriter rigs(path / "rigs.bin");
  rigs.Value(uint64_t(2));
  for (uint32_t i = 1; i <= 2; ++i) {
    rigs.Value(i);
    rigs.Value(uint32_t(1));
    rigs.Value(int32_t(0));
    rigs.Value(i);
  }
  rigs.Commit();
  BinaryWriter frames(path / "frames.bin"), images(path / "images.bin");
  frames.Value(uint64_t(model.Registered()));
  images.Value(uint64_t(model.Registered()));
  for (uint32_t i = 0; i < model.images.size(); ++i) {
    const auto& image = model.images[i];
    if (!image.registered)
      continue;
    frames.Value(i + 1);
    frames.Value(image.camera + 1);
    PoseOut(frames, image.pose);
    frames.Value(uint32_t(1));
    frames.Value(int32_t(0));
    frames.Value(image.camera + 1);
    frames.Value(uint64_t(i + 1));
    images.Value(i + 1);
    PoseOut(images, image.pose);
    images.Value(image.camera + 1);
    const auto& name = scene.images[i].name;
    images.Bytes(name.c_str(), name.size() + 1);
    images.Value(uint64_t(image.pixels.size()));
    for (size_t k = 0; k < image.pixels.size(); ++k) {
      images.Value(image.pixels[k].x());
      images.Value(image.pixels[k].y());
      images.Value(image.points[k] == NoPoint ? NoPoint : image.points[k] + 1);
    }
  }
  frames.Commit();
  images.Commit();
  size_t count = 0;
  for (const auto& p : model.points)
    if (p.track.size() >= 2)
      ++count;
  BinaryWriter points(path / "points3D.bin");
  points.Value(uint64_t(count));
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& p = model.points[id];
    if (p.track.size() < 2)
      continue;
    points.Value(id + 1);
    for (int j = 0; j < 3; ++j)
      points.Value(p.xyz[j]);
    points.Value(p.color);
    points.Value(p.error);
    points.Value(uint64_t(p.track.size()));
    for (const auto& o : p.track) {
      points.Value(o.image + 1);
      points.Value(o.feature);
    }
  }
  points.Commit();
}
Reconstruction ReadColmap(const fs::path& path, Scene& scene) {
  Reconstruction m;
  BinaryReader cameras(path / "cameras.bin");
  if (cameras.Value<uint64_t>() != 2)
    throw std::runtime_error("Expected two shared fisheye cameras");
  for (int i = 0; i < 2; ++i) {
    const auto id = cameras.Value<uint32_t>();
    if (id != uint32_t(i + 1) || cameras.Value<int32_t>() != 5)
      throw std::runtime_error("Unexpected COLMAP camera id/model");
    auto& c = m.cameras[i];
    c.label = i ? "R" : "L";
    const auto w = cameras.Value<uint64_t>(), h = cameras.Value<uint64_t>();
    if (w > 100000 || h > 100000)
      throw std::runtime_error("Invalid camera dimensions");
    c.width = int(w);
    c.height = int(h);
    for (double& x : c.params)
      x = cameras.Value<double>();
    c.Radius();
  }
  cameras.End();
  scene.cameras = m.cameras;
  BinaryReader rigs(path / "rigs.bin");
  if (rigs.Value<uint64_t>() != 2)
    throw std::runtime_error("Expected one rig per camera");
  for (uint32_t i = 1; i <= 2; ++i)
    if (rigs.Value<uint32_t>() != i || rigs.Value<uint32_t>() != 1 || rigs.Value<int32_t>() != 0 ||
        rigs.Value<uint32_t>() != i)
      throw std::runtime_error("Unexpected rig sensor");
  rigs.End();
  BinaryReader images(path / "images.bin");
  const auto n = images.Value<uint64_t>();
  if (n > 1000000)
    throw std::runtime_error("Invalid image count");
  m.images.resize(n);
  scene.images.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (images.Value<uint32_t>() != i + 1)
      throw std::runtime_error("Unexpected image identity");
    auto& image = m.images[i];
    image.pose = PoseIn(images);
    image.camera = images.Value<uint32_t>() - 1;
    if (image.camera >= 2)
      throw std::runtime_error("Unknown image camera");
    std::string name;
    for (size_t k = 0;; ++k) {
      const char c = images.Value<char>();
      if (!c)
        break;
      if (k > 4096)
        throw std::runtime_error("Invalid image name");
      name += c;
    }
    const auto relative = fs::u8path(name);
    if (relative.is_absolute())
      throw std::runtime_error("Image name must be relative");
    for (const auto& part : relative)
      if (part == "..")
        throw std::runtime_error("Image name escapes bundle");
    scene.images[i].id = i;
    scene.images[i].camera = image.camera;
    scene.images[i].name = name;
    const auto count = images.Value<uint64_t>();
    if (count > 12288)
      throw std::runtime_error("Invalid feature count");
    image.pixels.resize(count);
    image.points.resize(count);
    image.registered = true;
    for (size_t k = 0; k < count; ++k) {
      const double x = images.Value<double>(), y = images.Value<double>();
      image.pixels[k] = {x, y};
      image.points[k] = images.Value<uint64_t>();
      if (!std::isfinite(x) || !std::isfinite(y))
        throw std::runtime_error("Nonfinite feature coordinate");
    }
  }
  images.End();
  BinaryReader frames(path / "frames.bin");
  if (frames.Value<uint64_t>() != n)
    throw std::runtime_error("Frame count differs");
  for (uint32_t i = 0; i < n; ++i) {
    if (frames.Value<uint32_t>() != i + 1 || frames.Value<uint32_t>() != m.images[i].camera + 1)
      throw std::runtime_error("Frame identity differs");
    const auto p = PoseIn(frames);
    for (int j = 0; j < 7; ++j)
      if (p.params[j] != m.images[i].pose.params[j])
        throw std::runtime_error("Frame and image pose differ");
    if (frames.Value<uint32_t>() != 1 || frames.Value<int32_t>() != 0 ||
        frames.Value<uint32_t>() != m.images[i].camera + 1 || frames.Value<uint64_t>() != i + 1)
      throw std::runtime_error("Frame exposure differs");
  }
  frames.End();
  BinaryReader points(path / "points3D.bin");
  const auto count = points.Value<uint64_t>();
  if (count > fs::file_size(path / "points3D.bin") / 43)
    throw std::runtime_error("Invalid point count");
  m.points.resize(count);
  std::unordered_map<uint64_t, uint64_t> ids;
  for (uint64_t i = 0; i < count; ++i) {
    auto& p = m.points[i];
    const auto id = points.Value<uint64_t>();
    if (id == NoPoint || !ids.emplace(id, i).second)
      throw std::runtime_error("Duplicate point identity");
    for (int j = 0; j < 3; ++j)
      p.xyz[j] = points.Value<double>();
    p.color = points.Value<std::array<uint8_t, 3>>();
    p.error = points.Value<double>();
    const auto length = points.Value<uint64_t>();
    if (length < 2 || length > n || !p.xyz.allFinite() || !std::isfinite(p.error) || p.error < 0)
      throw std::runtime_error("Invalid point/track");
    std::set<uint32_t> seen;
    for (size_t j = 0; j < length; ++j) {
      const auto image = points.Value<uint32_t>() - 1, feature = points.Value<uint32_t>();
      if (image >= n || feature >= m.images[image].points.size() || !seen.insert(image).second ||
          m.images[image].points[feature] != id)
        throw std::runtime_error("Point/observation references differ");
      p.track.push_back({image, feature});
    }
  }
  points.End();
  for (auto& image : m.images)
    for (auto& id : image.points)
      if (id != NoPoint) {
        const auto it = ids.find(id);
        if (it == ids.end())
          throw std::runtime_error("Image refers to missing point");
        id = it->second;
      }
  const auto references = m.images;
  m.RebuildReferences();
  for (size_t i = 0; i < m.images.size(); ++i)
    if (m.images[i].points != references[i].points)
      throw std::runtime_error("Track references are not reciprocal");
  for (const auto& point : m.points) {
    double error = 0;
    for (const auto& observation : point.track) {
      const auto& image = m.images[observation.image];
      error += (m.cameras[image.camera].Project(image.pose.Apply(point.xyz)) -
                image.pixels[observation.feature])
                   .norm();
    }
    if (std::abs(error / point.track.size() - point.error) > 1e-6)
      throw std::runtime_error("COLMAP point error differs from its observations");
  }
  return m;
}
}  // namespace
void ExportBundle(const Scene& scene, const Reconstruction& model, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report) {
  const Json request = {{"scene_file", FileMetadata(work / "scene.json")},
                        {"model_file", FileMetadata(work / "cache/aligned.bin")},
                        {"include_ply", include_ply},
                        {"voxel", voxel}};
  if (fs::exists(output)) {
    if (!fs::is_regular_file(output / "manifest.json") ||
        ReadJson(output / "manifest.json").at("request") != request)
      throw std::runtime_error("Output exists with different inputs; choose a new output");
    CheckBundle(output);
    return;
  }
  auto pending = output;
  pending += ".pending";
  if (fs::exists(pending)) {
    auto archived = pending;
    archived += ".interrupted-" +
                std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    fs::rename(pending, archived);
  }
  fs::create_directories(pending / "sparse/0");
  WriteColmap(pending / "sparse/0", scene, model);
  Json photos = Json::array();
  for (const auto& e : scene.images) {
    const auto destination = pending / "images" / fs::u8path(e.name);
    fs::create_directories(destination.parent_path());
    fs::copy_file(e.source, destination);
    const auto mask = pending / "masks" / fs::u8path(e.name + ".png");
    fs::create_directories(mask.parent_path());
    fs::copy_file(MaskPath(work, e.id), mask);
    photos.push_back({{"name", e.name},
                      {"camera", e.camera},
                      {"image_size", fs::file_size(destination)},
                      {"mask_size", fs::file_size(mask)}});
  }
  Json initialization = nullptr;
  if (include_ply)
    initialization = ExportLasPly(scene.las, pending / "sparse/0/points3D.ply", voxel);
  WriteJson(pending / "quality.json", report.at("quality"));
  if (include_ply)
    WriteJson(pending / "initialization.json", initialization);
  Json sizes = Json::object();
  for (const auto* file : {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"})
    sizes[file] = fs::file_size(pending / "sparse/0" / file);
  WriteJson(pending / "manifest.json", {{"version", 1},
                                        {"request", request},
                                        {"images", photos},
                                        {"model_files", sizes},
                                        {"initialization", initialization}});
  CheckBundle(pending);
  fs::rename(pending, output);
}
Json CheckBundle(const fs::path& output) {
  const auto manifest = ReadJson(output / "manifest.json");
  if (manifest.at("version") != 1)
    throw std::runtime_error("Unsupported bundle manifest");
  const auto sparse = output / "sparse/0";
  for (const auto& [name, size] : manifest.at("model_files").items())
    if (fs::file_size(sparse / fs::u8path(name)) != size.get<uint64_t>())
      throw std::runtime_error("Model file size differs");
  Scene scene;
  const auto model = ReadColmap(sparse, scene);
  if (manifest.at("images").size() != scene.images.size())
    throw std::runtime_error("Bundle exposure count differs");
  for (uint32_t i = 0; i < scene.images.size(); ++i) {
    const auto& e = scene.images[i];
    const auto& saved = manifest.at("images")[i];
    if (saved.at("name") != e.name || saved.at("camera") != e.camera)
      throw std::runtime_error("Bundle image identity differs");
    const auto image = output / "images" / fs::u8path(e.name),
               mask = output / "masks" / fs::u8path(e.name + ".png");
    if (fs::file_size(image) != saved.at("image_size").get<uint64_t>() ||
        fs::file_size(mask) != saved.at("mask_size").get<uint64_t>())
      throw std::runtime_error("Image/mask size differs");
    const auto ih = ImageHeader(image), mh = ImageHeader(mask);
    const auto& camera = model.cameras[e.camera];
    if (ih.at("width") != camera.width || ih.at("height") != camera.height ||
        ih.at("orientation") != 1 || mh.at("width") != camera.width ||
        mh.at("height") != camera.height || mh.at("mode") != "L")
      throw std::runtime_error("Image/mask geometry differs");
  }
  if (!manifest.at("initialization").is_null())
    VerifyInitializationPly(sparse / "points3D.ply", manifest.at("initialization").at("points"));
  const auto saved = ReadJson(output / "quality.json");
  const auto quality =
      model.Quality(scene, saved.at("limits").at("median_px"), saved.at("limits").at("p90_px"));
  if (!saved.at("passed").get<bool>() || !quality.at("passed").get<bool>())
    throw std::runtime_error("Bundle geometry failed acceptance");
  return {{"passed", true},
          {"images", model.Registered()},
          {"points", quality.at("points")},
          {"observations", quality.at("observations")}};
}
}  // namespace geosun

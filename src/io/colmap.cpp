#include "io/colmap.h"
#include "io/cache.h"
#include "io/crop.h"
#include "io/las.h"
#include "io/roi.h"
#include "io/trainer_config.h"
#include "geometry/static_masks.h"
#include "sfm/shutter.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <memory>
#include <set>
#include <thread>
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
  std::vector<uint32_t> image_ids(model.images.size());
  uint32_t next_image_id = 1;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered)
      image_ids[i] = next_image_id++;
  for (uint32_t i = 0; i < model.images.size(); ++i) {
    const auto& image = model.images[i];
    if (!image.registered)
      continue;
    frames.Value(image_ids[i]);
    frames.Value(image.camera + 1);
    PoseOut(frames, image.pose);
    frames.Value(uint32_t(1));
    frames.Value(int32_t(0));
    frames.Value(image.camera + 1);
    frames.Value(uint64_t(image_ids[i]));
    images.Value(image_ids[i]);
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
      points.Value(image_ids[o.image]);
      points.Value(o.feature);
    }
  }
  points.Commit();
}
// shutter is the delivery's rolling_shutter.json when its photos are the original rolling-shutter
// images; point errors are then verified with that projection, returned in motion.
Reconstruction ReadColmap(const fs::path& path, Scene& scene, const Json* shutter,
                          ShutterMotion& motion) {
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
  if (shutter)
    motion = DeliveryShutterFromJson(*shutter, scene, m);
  for (const auto& point : m.points) {
    const double error = shutter ? ShutterPointError(m, motion, point) : m.PointError(point);
    if (std::abs(error - point.error) > 1e-6)
      throw std::runtime_error("COLMAP point error differs from its observations");
  }
  return m;
}
// Masks with the people that are persistent in the scene made static again (see StaticMasks),
// cached in the work directory beside the masks they come from. When keep_cache is non-null and
// a fresh cache had to be built, transfers ownership to keep_cache for supervision projection.
fs::path BuildStaticMasks(const Scene& scene, const Reconstruction& selected,
                          const ShutterMotion* motion, const fs::path& work, const Json& report,
                          int cache_mb, std::unique_ptr<LasSurfelCache>* keep_cache = nullptr) {
  const auto source = work / "cache/masks", target = work / "cache/static_masks";
  Json inputs = Json::object();
  for (const auto& image : scene.images)
    if (selected.images[image.id].registered)
      inputs[image.name] = FileMetadata(source / (std::to_string(image.id) + ".png"));
  Json request = {{"las_file", FileMetadata(scene.las)},
                  {"model_file", FileMetadata(SelectedModelPath(work, report))},
                  {"rules", StaticMaskRules()},
                  {"masks", inputs}};
  if (motion)
    request["motion_file"] = FileMetadata(work / "cache/shutter.json");
  if (fs::is_regular_file(target / "static_masks.json") &&
      ReadJson(target / "static_masks.json").at("request") == request)
    return target;
  if (FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed since reconstruction; choose a new work directory");
  auto pending = target;
  pending += ".pending";
  ArchiveInterrupted(pending);
  fs::create_directories(pending);
  const auto header = ReadSceneLas(scene);
  auto surfel_cache =
      std::make_unique<LasSurfelCache>(scene.las, header, work / "cache", cache_mb, &selected);
  const auto result =
      StaticMasks(scene, selected, motion, header, *surfel_cache,
                  [&](uint32_t id) { return source / (std::to_string(id) + ".png"); },
                  [&](uint32_t id) { return pending / (std::to_string(id) + ".png"); });
  if (keep_cache)
    *keep_cache = std::move(surfel_cache);
  else
    surfel_cache->Finish();
  WriteJson(pending / "static_masks.json", {{"request", request}, {"result", result}});
  RemoveAll(target);
  PublishDirectory(pending, target);
  return target;
}
}  // namespace
Json ExportBundle(const Scene& scene, const Reconstruction& selected, const fs::path& work,
                  const fs::path& output, bool include_ply, double voxel, const Json& report,
                  const SupervisionOptions& supervision_options, bool crop, bool include_roi,
                  double vram_gb, const fs::path& trainer_template) {
  const bool generate_roi = include_ply && include_roi;
  const double resolved_vram_gb = vram_gb > 0.0 ? vram_gb : QueryGpuVramGiB();
  // A rolling-shutter corrected model describes photos at its reference poses; they are delivered
  // original, with the readout motion beside the model.
  const bool rolling = report.contains("shutter") && report.at("shutter").value("used", false);
  auto masks = work / "cache/masks";
  auto step = Clock::now();
  const auto timed = [&](const char* name) {
    Log(std::string("Export ") + name + ": " + std::to_string(Seconds(step)) + " s");
    step = Clock::now();
  };
  ShutterMotion motion;
  if (rolling)
    motion =
        MotionFromJson(ReadJson(work / "cache/shutter.json").at("motion"), selected.images.size());
  // A cropped delivery keeps each lens's valid domain: photos are cut losslessly on the JPEG MCU
  // grid and the readout motion is rescaled to the cropped readout axis.
  auto options = supervision_options;
  std::unique_ptr<LasSurfelCache> surfel_cache;
  // Static masks use the LAS nearest-depth surfel projection; keep the cache to also project
  // supervision LAS depths before MoGe-2 loads on the GPU.
  Json static_request = nullptr;
  Json static_summary;
  struct CalibratedPhoto {
    double ev = 0.0;
    double pixel_ev_shift = 0.0;
    double residual_ev = 0.0;
    std::array<double, 3> wb_gains{1.0, 1.0, 1.0};
  };
  bool photometric_calibrated = false;
  bool wb_calibrated = false;
  double exposure_ref_ev = 0.0;
  std::array<std::array<double, 3>, 2> vig_coeffs{};
  std::unordered_map<uint32_t, CalibratedPhoto> calibrated_photos;
  if (options.enabled && options.static_masks) {
    masks = BuildStaticMasks(scene, selected, rolling ? &motion : nullptr, work, report,
                             options.cache_mb, &surfel_cache);
    static_request = {{"rules", StaticMaskRules()},
                      {"file", FileMetadata(masks / "static_masks.json")}};
    static_summary = ReadJson(masks / "static_masks.json").at("result");
    if (static_summary.contains("photometric") &&
        static_summary.at("photometric").contains("exposure") &&
        static_summary.at("photometric").at("exposure").value("calibrated", false) &&
        static_summary.contains("exposure_ev")) {
      const auto& pm = static_summary.at("photometric");
      photometric_calibrated = true;
      exposure_ref_ev = pm.at("exposure").value("reference_ev", 0.0);
      if (pm.contains("white_balance"))
        wb_calibrated = pm.at("white_balance").value("calibrated", false);
      if (pm.contains("vignetting_coeffs") && pm.at("vignetting_coeffs").size() == 2) {
        for (size_t c = 0; c < 2; ++c)
          for (size_t k = 0; k < 3; ++k)
            vig_coeffs[c][k] = pm.at("vignetting_coeffs")[c][k].get<double>();
      }
      for (const auto& item : static_summary.at("exposure_ev")) {
        CalibratedPhoto cp;
        cp.ev = item.at("ev").get<double>();
        cp.pixel_ev_shift = item.value("pixel_ev_shift", 0.0);
        cp.residual_ev = item.value("residual_ev", cp.ev);
        if (item.contains("wb_gains") && item.at("wb_gains").size() == 3) {
          for (int ch = 0; ch < 3; ++ch)
            cp.wb_gains[size_t(ch)] = item.at("wb_gains")[size_t(ch)].get<double>();
        }
        calibrated_photos.emplace(item.at("image_index").get<uint32_t>(), cp);
      }
    }
    static_summary.erase("component_rows");
    static_summary.erase("exposure_ev");
    timed("static masks");
  }
  // Photos and masks are independent files; their decoding and writing run on all cores.
  const int workers = std::clamp(int(std::thread::hardware_concurrency()), 1, 16);
  std::vector<const Exposure*> registered;
  for (const auto& e : scene.images)
    if (selected.images[e.id].registered)
      registered.push_back(&e);
  std::optional<std::array<CropWindow, 2>> windows;
  if (crop) {
    std::vector<int> mcus(registered.size());
    ParallelFor(registered.size(), workers, 8,
                [&](size_t i) { mcus[i] = JpegMcuSize(registered[i]->source); });
    int mcu = 0;
    for (const int size : mcus)
      mcu = std::max(mcu, size);
    windows = ValidDomainCrop(scene, selected, masks, rolling ? motion.axis : 1, mcu, workers);
    options.crop = windows;
    timed("crop windows");
  }
  Json request = {{"static_masks", static_request},
                  {"scene_file", FileMetadata(work / "scene.json")},
                  {"model_file", FileMetadata(SelectedModelPath(work, report))},
                  {"include_ply", include_ply},
                  {"voxel", voxel}};
  if (rolling)
    request["rolling_shutter"] = {{"motion_file", FileMetadata(work / "cache/shutter.json")},
                                  {"photos", "original"}};
  if (windows)
    request["crop"] = {{"mode", "valid_domain"},
                       {"windows", {(*windows)[0].ToJson(), (*windows)[1].ToJson()}}};
  if (photometric_calibrated)
    request["photometric_normalization"] = {{"enabled", true},
                                            {"jpeg_quality", 97},
                                            {"vignetting_corrected", true},
                                            {"white_balance_corrected", wb_calibrated},
                                            {"reference_ev", exposure_ref_ev}};
  if (include_ply)
    request["ply_filter"] = {
        {"max_range_m", 50.0}, {"max_points", 2500000}, {"max_visual_supplement_points", 100000}};
  request["trainer_config"] = {
      {"vram_gb", resolved_vram_gb},
      {"template",
       trainer_template.empty() ? Json("embedded") : FileMetadata(trainer_template)}};
  request["roi"] =
      generate_roi
          ? Json{{"enabled", true}, {"source", "las_camera_seeded_occupancy"}, {"version", 1}}
          : Json{{"enabled", false}};
  if (options.enabled)
    request["supervision"] = SupervisionRequest(scene, selected, masks, options, rolling);
  timed("request");
  if (fs::exists(output)) {
    if (!fs::is_regular_file(output / "manifest.json") ||
        ReadJson(output / "manifest.json").at("request") != request)
      throw std::runtime_error("Output exists with different inputs; choose a new output");
    return CheckBundle(output);
  }
  // Delivery copies the original files, so they must still be the ones reconstructed.
  for (const auto& e : scene.images)
    if (selected.images[e.id].registered && FileMetadata(e.source) != e.source_file)
      throw std::runtime_error("Source image changed since reconstruction: " + e.name);
  if ((include_ply || options.enabled) && FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed since reconstruction; choose a new work directory");
  auto pending = output;
  pending += ".pending";
  ArchiveInterrupted(pending);
  fs::create_directories(pending / "sparse/0");
  timed("source checks");
  const auto original = rolling ? OriginalPhotoObservations(selected, motion, 8) : selected;
  Reconstruction uncropped_visual;
  if (include_ply) {
    const auto pre_cleanup_path =
        rolling ? (work / "cache/shutter.bin") : (work / "cache/aligned.bin");
    auto pre_cleanup = fs::exists(pre_cleanup_path) ? ReadModel(pre_cleanup_path) : selected;
    bool stripped = false;
    for (size_t id = 0; id < pre_cleanup.images.size(); ++id)
      if (!selected.images[id].registered && pre_cleanup.images[id].registered) {
        pre_cleanup.images[id].registered = false;
        stripped = true;
      }
    if (stripped) {
      for (auto& pt : pre_cleanup.points) {
        pt.track.erase(std::remove_if(pt.track.begin(), pt.track.end(),
                                      [&](const Observation& o) {
                                        return !pre_cleanup.images[o.image].registered;
                                      }),
                       pt.track.end());
        if (pt.track.size() < 2)
          pt.track.clear();
      }
      pre_cleanup.RebuildReferences();
    }
    uncropped_visual = rolling ? OriginalPhotoObservations(pre_cleanup, motion, 8)
                               : std::move(pre_cleanup);
  }
  if (windows && rolling)
    motion = CropMotion(motion, original, *windows);
  const auto model = windows ? CropModel(original, *windows, rolling ? &motion : nullptr) : original;
  const auto visual_model =
      include_ply
          ? (windows ? CropModel(uncropped_visual, *windows, rolling ? &motion : nullptr)
                     : std::move(uncropped_visual))
          : Reconstruction{};
  WriteColmap(pending / "sparse/0", scene, model);
  timed("model");
  for (const auto* e : registered) {
    fs::create_directories((pending / "images" / fs::u8path(e->name)).parent_path());
    fs::create_directories((pending / "masks" / fs::u8path(e->name + ".png")).parent_path());
  }
  std::vector<Json> rows(registered.size());
  ParallelFor(registered.size(), workers, 1, [&](size_t i) {
    const auto& e = *registered[i];
    const auto destination = pending / "images" / fs::u8path(e.name);
    const auto mask = pending / "masks" / fs::u8path(e.name + ".png");
    const auto source_mask = masks / (std::to_string(e.id) + ".png");
    const auto cal_it = calibrated_photos.find(e.id);
    if (cal_it != calibrated_photos.end()) {
      const auto& cp = cal_it->second;
      PhotometricCorrection corr;
      corr.source_camera = selected.cameras[e.camera];
      corr.vig_coeffs = vig_coeffs[e.camera];
      corr.wb_gains = cp.wb_gains;
      corr.pixel_ev_shift = cp.pixel_ev_shift;
      corr.residual_ev = cp.residual_ev;
      const std::optional<CropWindow> win =
          windows ? std::optional<CropWindow>((*windows)[e.camera]) : std::nullopt;
      ExportNormalizedJpeg(e.source, destination, win, corr, 97);
      if (win)
        CropMask(source_mask, mask, *win);
      else
        fs::copy_file(source_mask, mask);
      if (FileMetadata(e.source) != e.source_file)
        throw std::runtime_error("Source image changed while normalizing: " + e.name);
    } else if (windows) {
      CropJpeg(e.source, destination, (*windows)[e.camera]);
      CropMask(source_mask, mask, (*windows)[e.camera]);
      if (FileMetadata(e.source) != e.source_file)
        throw std::runtime_error("Source image changed while cropping: " + e.name);
    } else {
      fs::copy_file(e.source, destination);
      if (fs::file_size(destination) != e.source_file.at("size_bytes").get<uint64_t>())
        throw std::runtime_error("Source image changed while copying: " + e.name);
      fs::copy_file(source_mask, mask);
    }
    rows[i] = {{"image_index", e.id},
               {"name", e.name},
               {"camera", e.camera},
               {"mask_size", fs::file_size(mask)},
               {"image_size", fs::file_size(destination)}};
    if (cal_it != calibrated_photos.end()) {
      const auto& cp = cal_it->second;
      rows[i]["calibrated_ev"] = cp.residual_ev;
      rows[i]["raw_relative_ev"] = cp.ev;
      rows[i]["pixel_ev_shift"] = cp.pixel_ev_shift;
      rows[i]["wb_gains"] = {cp.wb_gains[0], cp.wb_gains[1], cp.wb_gains[2]};
    }
  });
  Json photos = Json::array();
  for (auto& row : rows)
    photos.push_back(std::move(row));
  timed("photos and masks");
  // The PLY voxelizes the LAS on the CPU (supplemented with un-covered static visual points) while
  // the supervision projects it on the GPU. Wait for the job before locals unwind on exception.
  std::future<Json> initialization_job;
  struct JoinInitialization {
    std::future<Json>& job;
    ~JoinInitialization() {
      if (job.valid())
        job.wait();
    }
  } join_initialization{initialization_job};
  const size_t registered_count = registered.size();
  const auto cap_resolver = [registered_count, resolved_vram_gb, trainer_template](
                                uint64_t /*voxel_total*/, uint64_t geom_points) -> uint64_t {
    return BuildAdaptiveTrainerConfig(
               {registered_count, geom_points, resolved_vram_gb, trainer_template})
        .ply_cap;
  };
  Json roi_json;
  if (include_ply)
    initialization_job = std::async(std::launch::async, [&] {
      return ExportLasPly(
          scene.las, ReadSceneLas(scene), pending / "sparse/0/points3D.ply", voxel, &model,
          &visual_model,
          [&](uint32_t id) { return pending / "masks" / fs::u8path(scene.images[id].name + ".png"); },
          generate_roi ? &roi_json : nullptr, cap_resolver);
    });
  Json supervision = nullptr;
  if (options.enabled) {
    supervision = ExportSupervision(scene, model, masks, pending, options, rolling ? &motion : nullptr,
                                    surfel_cache.get());
    surfel_cache.reset();
    WriteJson(pending / "supervision.json", supervision);
  }
  surfel_cache.reset();
  const Json initialization = include_ply ? initialization_job.get() : Json(nullptr);
  if (generate_roi)
    WriteJson(pending / "roi.json", roi_json);
  uint64_t geom_points = 0;
  if (!initialization.is_null()) {
    if (initialization.contains("roi_voxel_points_before_cap"))
      geom_points = initialization.at("roi_voxel_points_before_cap").get<uint64_t>();
    else if (initialization.contains("voxel_points_before_cap"))
      geom_points = initialization.at("voxel_points_before_cap").get<uint64_t>();
    else if (initialization.contains("points"))
      geom_points = initialization.at("points").get<uint64_t>();
  }
  const auto trainer_result = BuildAdaptiveTrainerConfig(
      {registered_count, geom_points, resolved_vram_gb, trainer_template});
  WriteJson(pending / "trainer.json", trainer_result.config);
  timed("PLY and supervision");
  const auto registration = model.Registration(scene);
  auto quality = report.at("quality");
  if (rolling) {
    // The delivered observations are original pixels; report their rolling-shutter fit.
    const double median = quality.at("limits").at("median_px"),
                 p90 = quality.at("limits").at("p90_px");
    quality = model.Quality(windows ? CropScene(scene, *windows) : scene, median, p90, &motion);
    WriteJson(pending / "rolling_shutter.json", DeliveryShutterJson(scene, model, motion));
  }
  quality["registration"] = registration;
  WriteJson(pending / "quality.json", quality);
  // Stage reports travel with the delivery; production runs remove the work directory.
  WriteJson(pending / "result.json", report);
  if (include_ply)
    WriteJson(pending / "initialization.json", initialization);
  Json sizes = Json::object();
  for (const auto* file : {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"})
    sizes[file] = fs::file_size(pending / "sparse/0" / file);
  Json manifest = {{"project", scene.project.filename().u8string()},
                   {"request", request},
                   {"registration", registration},
                   {"images", photos},
                   {"model_files", sizes},
                   {"initialization", initialization}};
  if (options.enabled) {
    manifest["supervision"] = supervision;
    if (!static_summary.is_null())
      manifest["static_masks"] = static_summary;
  }
  if (rolling)
    manifest["rolling_shutter"] = {{"file", "rolling_shutter.json"},
                                   {"size_bytes", fs::file_size(pending / "rolling_shutter.json")}};
  if (generate_roi)
    manifest["roi"] = {{"file", "roi.json"},
                       {"size_bytes", fs::file_size(pending / "roi.json")}};
  manifest["trainer_config"] = {{"file", "trainer.json"},
                                {"size_bytes", fs::file_size(pending / "trainer.json")},
                                {"summary", trainer_result.summary}};
  WriteJson(pending / "manifest.json", manifest);
  timed("reports");
  auto checked = CheckBundle(pending);
  timed("readback");
  PublishDirectory(pending, output);
  return checked;
}
Json CheckBundle(const fs::path& output, const std::string& project) {
  const auto manifest = ReadJson(output / "manifest.json");
  if (!project.empty()) {
    const auto recorded = manifest.value("project", "");
    if (recorded.empty())
      throw std::runtime_error("Existing delivery has no recorded PCC project. Choose a new output.");
    const auto name = fs::u8path(project).filename();
    if (_wcsicmp(fs::u8path(recorded).c_str(), name.c_str()) != 0)
      throw std::runtime_error("Existing delivery was made from " + recorded +
                               ". Choose a new output for " + name.u8string() + ".");
  }
  const auto sparse = output / "sparse/0";
  for (const auto& [name, size] : manifest.at("model_files").items())
    if (fs::file_size(sparse / fs::u8path(name)) != size.get<uint64_t>())
      throw std::runtime_error("Model file size differs");
  Scene scene;
  const bool rolling = manifest.contains("rolling_shutter");
  const auto& request = manifest.at("request");
  if (rolling != (request.contains("rolling_shutter") &&
                  request.at("rolling_shutter").value("photos", "") == "original") ||
      rolling != fs::exists(output / "rolling_shutter.json"))
    throw std::runtime_error("Rolling-shutter delivery files differ from the request");
  Json shutter;
  if (rolling) {
    if (fs::file_size(output / "rolling_shutter.json") !=
        manifest.at("rolling_shutter").at("size_bytes").get<uint64_t>())
      throw std::runtime_error("Rolling-shutter file size differs");
    shutter = ReadJson(output / "rolling_shutter.json");
  }
  const bool has_roi = manifest.contains("roi");
  const bool roi_requested = request.contains("roi") && request.at("roi").value("enabled", false);
  if (has_roi != roi_requested || has_roi != fs::exists(output / "roi.json"))
    throw std::runtime_error("ROI delivery files differ from the request");
  ShutterMotion motion;
  const auto model = ReadColmap(sparse, scene, rolling ? &shutter : nullptr, motion);
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
    if (saved.contains("calibrated_ev")) {
      const auto ev = ReadJpegExifEv(image);
      const double expected_ev = saved.at("calibrated_ev").get<double>();
      if (!ev || !std::isfinite(*ev) || std::abs(*ev - expected_ev) > 0.01)
        throw std::runtime_error("Calibrated EXIF EV differs for " + e.name);
    }
  }
  if (!manifest.at("initialization").is_null())
    VerifyInitializationPly(sparse / "points3D.ply", manifest.at("initialization").at("points"));
  std::optional<RoiPolygon> checked_roi;
  if (has_roi) {
    if (manifest.at("initialization").is_null())
      throw std::runtime_error("ROI delivery requires initialization PLY metadata");
    if (fs::file_size(output / "roi.json") !=
        manifest.at("roi").at("size_bytes").get<uint64_t>())
      throw std::runtime_error("ROI file size differs");
    checked_roi = ValidateRoiJson(
        ReadJson(output / "roi.json"), model,
        manifest.at("initialization").at("coordinate_frame").get<std::string>());
  }
  if (manifest.contains("supervision")) {
    if (ReadJson(output / "supervision.json") != manifest.at("supervision") ||
        manifest.at("supervision").at("request") != manifest.at("request").at("supervision"))
      throw std::runtime_error("Supervision manifest differs");
    CheckSupervision(output, model, manifest.at("images"), manifest.at("supervision"));
  }
  if (manifest.contains("trainer_config")) {
    const auto& tc = manifest.at("trainer_config");
    const auto trainer_path = output / fs::u8path(tc.at("file").get<std::string>());
    if (!fs::is_regular_file(trainer_path) ||
        fs::file_size(trainer_path) != tc.at("size_bytes").get<uint64_t>())
      throw std::runtime_error("Trainer config file is missing or size differs");
    const auto parsed = ReadJson(trainer_path);
    if (!parsed.is_object() || !parsed.contains("optimization") ||
        !parsed.at("optimization").is_object() ||
        parsed.at("optimization").value("iterations", uint64_t(0)) == 0 ||
        parsed.at("optimization").value("max_cap", uint64_t(0)) == 0)
      throw std::runtime_error("Trainer config JSON is invalid");
  }
  const auto saved = ReadJson(output / "quality.json");
  const auto registration = manifest.at("registration");
  if (registration != saved.at("registration") ||
      registration.at("registered_images") != model.Registered() ||
      registration.at("selected_images").get<size_t>() !=
          model.Registered() + registration.at("unregistered_images").size() ||
      registration.at("complete") != registration.at("unregistered_images").empty())
    throw std::runtime_error("Bundle registration summary differs");
  const auto quality =
      model.Quality(scene, saved.at("limits").at("median_px"), saved.at("limits").at("p90_px"),
                    rolling ? &motion : nullptr);
  if (!saved.at("passed").get<bool>() || !quality.at("passed").get<bool>())
    throw std::runtime_error("Bundle geometry failed acceptance");
  Json checked = {{"passed", true},
          {"registration", registration},
          {"images", model.Registered()},
          {"points", quality.at("points")},
          {"observations", quality.at("observations")}};
  if (manifest.contains("supervision")) {
    const auto& supervision = manifest.at("supervision");
    checked["supervision"] = {{"images", supervision.at("images").size()},
                              {"depth_pixels", supervision.at("depth_pixels")},
                              {"normal_pixels", supervision.at("normal_pixels")}};
  }
  if (checked_roi) {
    checked["roi"] = {{"vertices", checked_roi->vertices_xy.size()},
                      {"area_m2", checked_roi->area_m2},
                      {"z_min", checked_roi->z_min},
                      {"z_max", checked_roi->z_max},
                      {"cameras_inside", checked_roi->cameras_inside}};
  }
  if (manifest.contains("trainer_config"))
    checked["trainer_config"] = manifest.at("trainer_config").at("summary");
  return checked;
}
}  // namespace geosun

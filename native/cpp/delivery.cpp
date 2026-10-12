#include "delivery.h"

#include <colmap/scene/reconstruction.h>

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <set>

#include "las_io.h"
#include "scene.h"

namespace {
const std::vector<std::string> ModelNames = {
    "cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"};
Json ModelMetadata(const fs::path& path) {
  Json result = Json::object();
  for (const auto& name : ModelNames) result[name] = FileMetadata(path / name);
  return result;
}
fs::path Member(const fs::path& root, const std::string& name) {
  const auto relative = fs::u8path(name);
  if (name.empty() || relative.is_absolute() ||
      name.find('\\') != std::string::npos ||
      relative.generic_u8string() != name)
    throw std::runtime_error("Expected canonical relative dataset path: " +
                             name);
  for (const auto& part : relative)
    if (part == "..")
      throw std::runtime_error("Expected relative dataset member");
  const auto file = root / relative;
  if (!fs::is_regular_file(file))
    throw std::runtime_error("Missing dataset member: " + name);
  return file;
}
colmap::Reconstruction ReadModel(const fs::path& path) {
  colmap::Reconstruction rec;
  rec.Read(path);
  if (!rec.IsValid() || !rec.NumRegImages() ||
      rec.NumImages() != rec.NumRegImages())
    throw std::runtime_error("Expected consistent registered COLMAP model");
  std::set<std::string> names;
  for (const auto& [id, c] : rec.Cameras())
    if (!c.width || !c.height ||
        !std::all_of(c.params.begin(), c.params.end(),
                     [](double v) { return std::isfinite(v); }))
      throw std::runtime_error("Invalid export camera");
  for (const auto& [id, image] : rec.Images()) {
    if (!names.insert(image.Name()).second)
      throw std::runtime_error("Duplicate export image name");
    if (!image.CamFromWorld().params.allFinite())
      throw std::runtime_error("Nonfinite image pose");
    for (const auto& point : image.Points2D())
      if (!point.xy.allFinite())
        throw std::runtime_error("Nonfinite image observation");
  }
  for (const auto& [id, p] : rec.Points3D())
    if (!p.xyz.allFinite()) throw std::runtime_error("Nonfinite sparse point");
  return rec;
}
bool Contains(const fs::path& parent, const fs::path& child) {
  const auto p = fs::weakly_canonical(parent), c = fs::weakly_canonical(child);
  auto a = p.begin(), b = c.begin();
  for (; a != p.end(); ++a, ++b) {
    if (b == c.end()) return false;
#ifdef _WIN32
    if (_wcsicmp(a->c_str(), b->c_str())) return false;
#else
    if (*a != *b) return false;
#endif
  }
  return true;
}
Json Recipe(double voxel) {
  return {{"version", 1},
          {"voxel_size_m", voxel},
          {"sampling",
           "all non-withheld, non-class-7 LAS records; one centroid per voxel"},
          {"color",
           "voxel mean of LAS RGB; auto-detect 8/16-bit range; gray if absent"},
          {"normals", "zero"},
          {"coordinate_frame", "input_las_xyz"}};
}
Json Request(const fs::path& model, const Json& scene, const fs::path& las,
             const fs::path& quality, const Json& provenance,
             const std::string& image_mode, bool masks, bool ply,
             double voxel) {
  Json initialization = nullptr;
  if (ply) {
    initialization = Recipe(voxel);
    initialization["source_las_file"] = FileMetadata(las);
  }
  std::map<std::string, Json> sorted;
  for (const auto& row : scene.at("images"))
    sorted[row.at("name").get<std::string>()] = row;
  Json images = Json::array();
  for (const auto& [name, row] : sorted)
    images.push_back(
        {{"name", name},
         {"source_file",
          FileMetadata(fs::u8path(row.at("source").get<std::string>()))},
         {"source", image_mode == "link" ? row.at("source") : Json(nullptr)},
         {"mask_file", masks && !row.at("semantic_mask").is_null()
                           ? FileMetadata(fs::u8path(
                                 row.at("semantic_mask").get<std::string>()))
                           : Json(nullptr)}});
  return {
      {"source_model_files", ModelMetadata(model)},
      {"provenance", provenance},
      {"image_mode", image_mode},
      {"include_masks", masks},
      {"include_ply", ply},
      {"initialization", initialization},
      {"quality_file", quality.empty() ? Json(nullptr) : FileMetadata(quality)},
      {"mask_cameras", masks ? scene.at("cameras") : Json(nullptr)},
      {"mask_fov", masks ? scene.at("fov_degrees") : Json(nullptr)},
      {"images", images}};
}
}  // namespace
Json CheckExport(const fs::path& output) {
  const auto identity = ReadJson(Member(output, "colmap_export.json"));
  if (identity.at("schema_version") != 4 || identity.at("format") != "COLMAP" ||
      !identity.at("request").is_object() ||
      (identity.at("image_mode") != "copy" &&
       identity.at("image_mode") != "link") ||
      identity.at("model_files").size() != ModelNames.size())
    throw std::runtime_error("Unsupported COLMAP export contract");
  for (const auto& name : ModelNames)
    if (fs::file_size(Member(output, "sparse/0/" + name)) !=
        identity.at("model_files").at(name).get<uint64_t>())
      throw std::runtime_error("Exported COLMAP file size changed: " + name);
  const auto rec = ReadModel(output / "sparse/0");
  std::map<std::string, colmap::image_t> images;
  for (const auto& [id, im] : rec.Images()) images[im.Name()] = id;
  if (identity.at("images").size() != images.size() ||
      rec.NumPoints3D() != identity.at("points").get<size_t>())
    throw std::runtime_error("Exported model membership differs");
  std::set<std::string> names;
  for (const auto& row : identity.at("images")) {
    const auto name = row.at("name").get<std::string>();
    if (!names.insert(name).second)
      throw std::runtime_error("Duplicate manifest image");
    const auto& image = rec.Image(images.at(name));
    const auto& camera = rec.Camera(image.CameraId());
    if (row.at("camera_id") != image.CameraId() ||
        row.at("size") != Json::array({camera.width, camera.height}))
      throw std::runtime_error("Exported camera/image identity changed");
    const auto file = Member(output, "images/" + name);
    const auto header = ImageHeader(file);
    if (fs::file_size(file) != row.at("size_bytes").get<uint64_t>() ||
        header.at("width") != camera.width ||
        header.at("height") != camera.height || header.at("orientation") != 1)
      throw std::runtime_error(
          "Exported image dimensions, size or orientation changed");
    if (identity.at("include_masks").get<bool>()) {
      const auto mask = Member(output, "masks/" + name + ".png");
      const auto h = ImageHeader(mask);
      if (fs::file_size(mask) != row.at("mask_size_bytes").get<uint64_t>() ||
          h.at("width") != camera.width || h.at("height") != camera.height ||
          h.at("mode") != "L")
        throw std::runtime_error("Invalid exported mask");
    }
  }
  for (const auto& [name, size] : identity.at("supplemental_files").items())
    if (fs::file_size(Member(output, name)) != size.get<uint64_t>())
      throw std::runtime_error("Exported supplementary file size changed");
  const auto& initialization = identity.at("initialization");
  if (!initialization.is_null()) {
    if (initialization.at("version") != 1 ||
        initialization.at("source") != "las" ||
        initialization.at("coordinate_frame") != "input_las_xyz" ||
        initialization.at("ply_size_bytes") !=
            identity.at("supplemental_files").at("sparse/0/points3D.ply") ||
        ReadJson(Member(output, "initialization.json")) != initialization)
      throw std::runtime_error("LAS initialization identity differs");
    if (initialization.at("voxel_size_m").get<double>() <= 0)
      throw std::runtime_error("Invalid initialization voxel size");
    VerifyInitializationPly(Member(output, "sparse/0/points3D.ply"),
                            initialization.at("points"));
  } else if (identity.at("supplemental_files")
                 .contains("sparse/0/points3D.ply"))
    throw std::runtime_error("Initialization PLY lacks provenance");
  return {{"output", output.u8string()},
          {"images", images.size()},
          {"points", rec.NumPoints3D()},
          {"image_mode", identity.at("image_mode")},
          {"include_masks", identity.at("include_masks")},
          {"initialization",
           initialization.is_null()
               ? Json(nullptr)
               : Json{{"source", "las"},
                      {"points", initialization.at("points")},
                      {"voxel_size_m", initialization.at("voxel_size_m")}}},
          {"check_scope", "model_structure_and_file_sizes"},
          {"runtime", "C++"}};
}
void RunExport(const Arguments& args) {
  args.Validate({"--work", "--input", "--manifest", "--output", "--image-mode",
                 "--las", "--quality", "--provenance", "--ply-voxel-size",
                 "--include-ply", "--no-include-ply", "--include-masks",
                 "--no-include-masks", "--blas-threads"});
  const auto work =
      args.Get("--work").empty() ? fs::path() : args.Path("--work");
  const auto model = args.Get("--input").empty()
                         ? work / "registration/native/sparse/0"
                         : args.Path("--input");
  const auto scene =
      ReadJson(args.Get("--manifest").empty() ? work / "sfm/manifest.json"
                                              : args.Path("--manifest"));
  const auto output =
      args.Get("--output").empty() ? work / "colmap" : args.Path("--output");
  if (Contains(output, model) || Contains(model, output))
    throw std::runtime_error("Export must be separate from source model");
  const bool masks = args.Get("--no-include-masks") != "true",
             ply = args.Get("--no-include-ply") != "true";
  const auto image_mode = args.Get("--image-mode", "copy");
  if (image_mode != "copy" && image_mode != "link")
    throw std::runtime_error("image-mode must be copy or link");
  const double voxel = args.Number("--ply-voxel-size", .05);
  fs::path las;
  if (ply) {
    if (!args.Get("--las").empty())
      las = args.Path("--las");
    else if (!work.empty())
      las = fs::u8path(
          ReadJson(work / "native_contract.json").at("las").get<std::string>());
    else
      throw std::runtime_error("LAS input is required for initialization");
    if (Contains(output, las))
      throw std::runtime_error("Export contains source LAS");
  }
  const auto quality =
      args.Get("--quality").empty()
          ? (work.empty() ? fs::path()
                          : work / "registration/native/quality.json")
          : args.Path("--quality");
  const auto provenance = args.Get("--provenance").empty()
                              ? Json::object()
                              : ReadJson(args.Path("--provenance"));
  const auto request = Request(model, scene, las, quality, provenance,
                               image_mode, masks, ply, voxel);
  if (fs::exists(output)) {
    const auto saved = ReadJson(output / "colmap_export.json");
    if (saved.at("request") != request)
      throw std::runtime_error(
          "Existing COLMAP export differs; choose new output");
    std::cout << CheckExport(output).dump() << '\n';
    return;
  }
  auto pending = output;
  pending += ".pending";
  if (fs::exists(pending)) throw std::runtime_error("Unfinished export exists");
  fs::create_directories(pending / "sparse/0");
  const auto rec = ReadModel(model);
  std::map<std::string, colmap::image_t> ids;
  for (const auto& [id, image] : rec.Images()) ids[image.Name()] = id;
  if (ids.size() != scene.at("images").size())
    throw std::runtime_error("Scene images must match registered model");
  Json files = Json::object(), images = Json::array(),
       supplements = Json::object();
  for (const auto& name : ModelNames) {
    fs::copy_file(model / name, pending / "sparse/0" / name);
    files[name] = fs::file_size(pending / "sparse/0" / name);
  }
  std::map<std::string, cv::Mat> fov;
  size_t done = 0;
  for (const auto& record : scene.at("images")) {
    const auto name = record.at("name").get<std::string>();
    const auto source = fs::u8path(record.at("source").get<std::string>()),
               destination = pending / "images" / fs::u8path(name);
    if (Contains(output, source))
      throw std::runtime_error("Export contains source image");
    fs::create_directories(destination.parent_path());
    if (image_mode == "copy")
      fs::copy_file(source, destination);
    else
      fs::create_symlink(fs::relative(source, destination.parent_path()),
                         destination);
    const auto& image = rec.Image(ids.at(name));
    const auto& camera = rec.Camera(image.CameraId());
    Json row = {{"name", name},
                {"camera_id", image.CameraId()},
                {"size", {camera.width, camera.height}},
                {"size_bytes", fs::file_size(destination)}};
    if (masks) {
      const auto label = record.at("camera").get<std::string>();
      if (!fov.count(label))
        fov[label] = Camera::FromJson(scene.at("cameras").at(label))
                         .Mask(scene.at("fov_degrees"));
      cv::Mat mask = fov.at(label).clone();
      if (!record.at("semantic_mask").is_null()) {
        mask =
            ReadImage(fs::u8path(record.at("semantic_mask").get<std::string>()),
                      cv::IMREAD_GRAYSCALE);
        if (mask.size() != fov.at(label).size())
          throw std::runtime_error("Mask/image dimensions differ");
        mask.setTo(0, fov.at(label) == 0);
      }
      const auto mask_path = pending / "masks" / fs::u8path(name + ".png");
      WriteImage(mask_path, mask);
      row["mask_size_bytes"] = fs::file_size(mask_path);
    }
    images.push_back(row);
    if (++done % 80 == 0 || done == scene.at("images").size())
      std::cerr << "Native export images " << done << '/'
                << scene.at("images").size() << '\n';
  }
  Json initialization = nullptr;
  if (ply) {
    initialization =
        ExportLasPly(las, pending / "sparse/0/points3D.ply", voxel);
    WriteJson(pending / "initialization.json", initialization);
    supplements["sparse/0/points3D.ply"] = initialization.at("ply_size_bytes");
    supplements["initialization.json"] =
        fs::file_size(pending / "initialization.json");
  }
  if (!quality.empty()) {
    fs::copy_file(quality, pending / "quality.json");
    supplements["quality.json"] = fs::file_size(pending / "quality.json");
  }
  if (Request(model, scene, las, quality, provenance, image_mode, masks, ply,
              voxel) != request)
    throw std::runtime_error("Export sources changed during copying");
  WriteJson(pending / "colmap_export.json",
            {{"schema_version", 4},
             {"format", "COLMAP"},
             {"request", request},
             {"image_mode", image_mode},
             {"include_masks", masks},
             {"model_files", files},
             {"images", images},
             {"points", rec.NumPoints3D()},
             {"supplemental_files", supplements},
             {"provenance", provenance},
             {"initialization", initialization}});
  CheckExport(pending);
  fs::rename(pending, output);
  std::cout << CheckExport(output).dump() << '\n';
}
void RunCheck(const Arguments& args) {
  args.Validate({"--output", "--work", "--blas-threads"});
  fs::path output;
  if (!args.Get("--output").empty())
    output = args.Path("--output");
  else {
    const auto work = args.Path("--work");
    output = fs::u8path(ReadJson(work / "native_contract.json")
                            .at("export")
                            .at("output")
                            .get<std::string>());
  }
  std::cout << CheckExport(output).dump() << '\n';
}

#include "app/pipeline.h"
#include "io/scene.h"
#include "io/las.h"
#include "io/colmap.h"
#include "io/cache.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "sfm/reconstruction.h"
#include "sfm/structure.h"
#include "sfm/shutter.h"
#include "alignment/alignment.h"
#include <set>
#include <iostream>
#include <iterator>
#include <optional>
#include <Windows.h>

namespace geosun {
// Compiled from configs/default.json.
extern const char DefaultSettings[];
namespace {
bool Contains(const fs::path& parent, const fs::path& child) {
  const auto p = fs::weakly_canonical(parent), c = fs::weakly_canonical(child);
  auto a = p.begin(), b = c.begin();
  for (; a != p.end() && !a->empty(); ++a, ++b) {
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
fs::path NormalizeDirectory(const fs::path& path) {
  auto normalized = path.lexically_normal();
  if (!normalized.has_filename() && normalized.has_relative_path())
    normalized = normalized.parent_path();
  return normalized;
}
// Work and delivery folders stay apart from every source photo directory and input file.
void ValidateSourcePaths(const SceneInputs& inputs, const fs::path& folder, const char* what) {
  for (const auto& photos : inputs.photos)
    if (Contains(folder, photos) || Contains(photos, folder))
      throw std::runtime_error(std::string("Use a separate ") + what +
                               " directory outside the photo directory " + photos.u8string());
  for (const auto& file : {inputs.project, inputs.camera_positions, inputs.navigation, inputs.enh,
                           inputs.las})
    if (Contains(folder, file))
      throw std::runtime_error(std::string("The ") + what + " directory must not contain input " +
                               file.u8string());
}
void ValidateDeliveryPaths(const SceneInputs& inputs, const fs::path& work, const fs::path& output) {
  auto pending = output;
  pending += ".pending";
  ValidateSourcePaths(inputs, work, "work");
  for (const auto& folder : {output, pending}) {
    ValidateSourcePaths(inputs, folder, "output");
    if (Contains(work, folder) || Contains(folder, work))
      throw std::runtime_error("Output and work directories must be separate");
  }
}
// All source paths, or none to reuse those recorded in an existing work directory.
std::optional<SceneInputs> ParseInputs(const Arguments& args) {
  static const char* const options[] = {"--project", "--images", "--camera-pos",
                                        "--pos",     "--pos-enh", "--las"};
  std::string missing;
  int given = 0;
  for (const auto* option : options) {
    if (args.Get(option).empty())
      missing += std::string(" ") + option;
    else
      ++given;
  }
  if (!given)
    return std::nullopt;
  if (!missing.empty())
    throw std::invalid_argument("Missing input option(s):" + missing);
  SceneInputs inputs;
  inputs.project = args.Path("--project");
  inputs.photos = args.Paths("--images");
  inputs.camera_positions = args.Path("--camera-pos");
  inputs.navigation = args.Path("--pos");
  inputs.enh = args.Path("--pos-enh");
  inputs.las = args.Path("--las");
  return inputs;
}
class Lock {
  HANDLE handle;

 public:
  explicit Lock(const fs::path& work)
      : handle(CreateFileW((work / ".lock").c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)) {
    if (handle == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Work directory is in use or cannot be locked");
  }
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;
  ~Lock() { CloseHandle(handle); }
};
Json Settings(const fs::path& file) {
  Json result = Json::parse(DefaultSettings);
  if (!file.empty()) {
    const auto configured = ReadJson(file);
    if (!configured.is_object())
      throw std::runtime_error("Configuration must be a JSON object");
    for (const auto& [key, value] : configured.items()) {
      if (!result.contains(key))
        throw std::runtime_error("Unknown setting: " + key);
      result[key] = value;
    }
  }
  enum class Kind { Integer, Boolean, Positive, Choice };
  static constexpr struct {
    const char* key;
    Kind kind;
    int min = 0, max = 0;
    const char* first = nullptr;
    const char* second = nullptr;
  } schema[] = {
      {"workers", Kind::Integer, 1, 32},
      {"ba_workers", Kind::Integer, 1, 32},
      {"sfm_repair_rounds", Kind::Integer, 0, 2},
      {"rolling_shutter", Kind::Boolean},
      {"quality_median_px", Kind::Positive},
      {"quality_p90_px", Kind::Positive},
      {"las_support_fraction", Kind::Positive},
      {"las_median_m", Kind::Positive},
      {"las_p90_m", Kind::Positive},
      {"calibration_tracks", Kind::Choice, 0, 0, "all", "stable"},
  };
  if (result.size() != std::size(schema))
    throw std::logic_error("Configuration defaults do not match the settings schema");
  for (const auto& rule : schema) {
    const auto& value = result.at(rule.key);
    const std::string key = rule.key;
    switch (rule.kind) {
      case Kind::Integer:
        if (!value.is_number_integer() || value.get<double>() < rule.min ||
            value.get<double>() > rule.max)
          throw std::runtime_error(key + " must be an integer from " +
                                   std::to_string(rule.min) + " to " + std::to_string(rule.max));
        break;
      case Kind::Boolean:
        if (!value.is_boolean())
          throw std::runtime_error(key + " must be a boolean");
        break;
      case Kind::Positive:
        if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() <= 0)
          throw std::runtime_error("Expected positive numeric setting: " + key);
        break;
      case Kind::Choice:
        if (value != rule.first && value != rule.second)
          throw std::runtime_error(key + " must be " + rule.first + " or " + rule.second);
        break;
    }
  }
  if (result.at("las_support_fraction").get<double>() > 1 ||
      result.at("quality_median_px").get<double>() > result.at("quality_p90_px").get<double>() ||
      result.at("las_median_m").get<double>() > result.at("las_p90_m").get<double>())
    throw std::runtime_error("Invalid quality limit ordering");
  return result;
}
AlignmentLimits Limits(const Json& settings) {
  return {settings.at("las_support_fraction"), settings.at("las_median_m"),
          settings.at("las_p90_m"), settings.at("quality_median_px"),
          settings.at("quality_p90_px")};
}
// Delivery options shared by run and export.
struct Delivery {
  bool ply;
  bool roi;
  double voxel;
  SupervisionOptions supervision;
  bool crop;
};
// The options ParseDelivery reads, accepted by every command that writes a delivery.
std::vector<std::string> WithDeliveryOptions(std::vector<std::string> options) {
  for (const auto* option :
       {"--no-ply", "--no-roi", "--ply-voxel", "--no-las-supervision", "--supervision-max-size",
        "--supervision-boundary", "--supervision-mask-dir", "--supervision-cache-mb",
        "--supervision-source", "--supervision-models", "--no-crop-valid-domain",
        "--no-static-masks"})
    options.push_back(option);
  return options;
}
Delivery ParseDelivery(const Arguments& args) {
  return {args.Get("--no-ply") != "true", args.Get("--no-roi") != "true",
          args.Number("--ply-voxel", .05), ParseSupervisionOptions(args),
          args.Get("--no-crop-valid-domain") != "true"};
}
// Key of a product: relative to the work directory, or absolute when no relative path exists
// (another drive). Verify resolves both, since work / absolute is the absolute path.
std::string InventoryKey(const fs::path& work, const fs::path& path) {
  const auto relative = fs::relative(path, work);
  return (relative.empty() ? path : relative).generic_u8string();
}
Json Inventory(const fs::path& work, const std::vector<fs::path>& products) {
  Json result = Json::object();
  for (const auto& path : products) {
    if (fs::is_directory(path)) {
      for (const auto& row : fs::recursive_directory_iterator(path))
        if (row.is_regular_file())
          result[InventoryKey(work, row.path())] = FileMetadata(row.path());
    } else
      result[InventoryKey(work, path)] = FileMetadata(path);
  }
  return result;
}
void Verify(const fs::path& work, const Json& files) {
  for (const auto& [name, metadata] : files.items())
    if (FileMetadata(work / fs::u8path(name)) != metadata)
      throw std::runtime_error("Completed stage artifact changed: " + name);
}
Json PostprocessResult(const Scene& scene, const Reconstruction& selected, const Json& original,
                       const std::string& selected_model) {
  std::vector<double> navigation_errors;
  const auto navigation = NavigationPositions(scene);
  for (uint32_t image = 0; image < selected.images.size(); ++image)
    if (selected.images[image].registered)
      navigation_errors.push_back(
          (selected.images[image].pose.Center() - navigation[image]).norm());
  Json result = original;
  result["quality"] = selected.Quality(scene, original.at("quality").at("limits").at("median_px"),
                                       original.at("quality").at("limits").at("p90_px"));
  result["selected_model"] = selected_model;
  result["navigation_error_m"] = Statistics(navigation_errors);
  result["build"] = BuildInfo();
  return result;
}
// state.json of a work directory. It is saved after every change so an interruption keeps the
// completed stages.
struct WorkState {
  fs::path path;
  Json value;
  void Save() const { WriteJson(path, value); }
};
// Bind the work directory to one model package; a different package needs a new work directory.
void RecordModels(WorkState& state, const fs::path& root) {
  Json identity = {{"manifest", FileMetadata(root / "manifest.json")}};
  const auto manifest = ReadJson(root / "manifest.json");
  for (const auto& [key, value] : manifest.at("models").items())
    identity[key] = FileMetadata(root / "engines" / (key + ".engine"));
  if (state.value.contains("model_identity") && state.value.at("model_identity") != identity)
    throw std::runtime_error("Models changed; choose a new work directory");
  state.value["model_identity"] = identity;
  state.value["models"] = root.u8string();
  state.Save();
}
Json ExportSelected(const Scene& scene, const fs::path& work, const fs::path& output,
                    const Delivery& delivery) {
  const auto result = ReadJson(work / "result.json");
  return ExportBundle(scene, ReadModel(SelectedModelPath(work, result)), work, output, delivery.ply,
                      delivery.voxel, result, delivery.supervision, delivery.crop, delivery.roi);
}
std::vector<fs::path> ReconstructVisual(const Scene& scene, const fs::path& work,
                                        const fs::path& root, const Json& settings,
                                        WorkState& state) {
  Json report;
  Json quality;
  Reconstruction model;
  const auto profile_path = work / "cache/matches_profile.json";
  const int completed_repairs =
      int(ReadJson(profile_path).value("repairs", Json::array()).size());
  const int max_repairs = settings.at("sfm_repair_rounds");
  for (int repair = completed_repairs;; ++repair) {
    std::set<uint32_t> deficient;
    try {
      model = Reconstruct(scene, work, settings.at("workers"), report,
                          settings.at("calibration_tracks") == "stable", repair >= max_repairs,
                          settings.at("ba_workers"));
      quality =
          model.Quality(scene, settings.at("quality_median_px"), settings.at("quality_p90_px"));
      if (quality.at("passed").get<bool>())
        break;
      for (uint32_t id = 0; id < model.images.size(); ++id) {
        if (!model.images[id].registered)
          continue;
        const auto& row = quality.at("images")[id];
        if (row.at("observations").get<size_t>() < MinimumImageObservations ||
            row.at("occupied_cells").get<size_t>() < MinimumImageCells)
          deficient.insert(id);
      }
      if (deficient.empty() || repair >= max_repairs)
        throw std::runtime_error("Visual acceptance rejected: " + quality.at("failures").dump());
      WriteModel(work / "cache/incremental.bin", model);
    } catch (const RegistrationIncomplete& missing) {
      if (repair >= max_repairs)
        throw;
      deficient.insert(missing.images.begin(), missing.images.end());
    }
    Log("Repair matching support for " + std::to_string(deficient.size()) + " images");
    auto profile = ReadJson(profile_path);
    profile["repairs"].push_back(MakeMatches(scene, work, root, deficient));
    WriteJson(profile_path, profile);
    state.value["stages"]["matches"]["files"] =
        Inventory(work, {MatchesPath(work), work / "cache/pairs.json", profile_path});
    state.Save();
  }
  WriteModel(work / "cache/visual.bin", model);
  report["quality"] = quality;
  WriteJson(work / "cache/sfm.json", report);
  state.value["registration"] = quality.at("registration");
  if (model.Registered() != model.images.size())
    Log("Partial reconstruction accepted: " + std::to_string(model.Registered()) + "/" +
        std::to_string(model.images.size()) + "; see quality.registration for missing images");
  return {work / "cache/visual.bin", work / "cache/sfm.json"};
}
std::vector<fs::path> AlignVisual(const Scene& scene, const fs::path& work, const Json& settings) {
  auto model = ReadModel(work / "cache/visual.bin");
  auto result =
      Align(scene, model, settings.at("workers"), Limits(settings), settings.at("ba_workers"));
  result["sfm"] = ReadJson(work / "cache/sfm.json");
  for (const auto& name : {"masks", "features", "matches"})
    result["frontend"][name] = ReadJson(work / "cache" / (std::string(name) + "_profile.json"));
  result["runtime"] = "C++/CUDA/TensorRT";
  result["build"] = BuildInfo();
  result["inputs"] = scene.SceneInputs::ToJson();
  WriteModel(work / "cache/aligned.bin", model);
  WriteJson(work / "cache/alignment.json", result);
  return {work / "cache/aligned.bin", work / "cache/alignment.json"};
}
std::vector<fs::path> CorrectShutter(const Scene& scene, const fs::path& work,
                                     const Json& settings) {
  const auto original = ReadJson(work / "cache/alignment.json");
  const auto model = ReadModel(SelectedModelPath(work, original));
  auto correction = CorrectRollingShutter(scene, model, work, settings.at("workers"),
                                          Limits(settings), settings.at("ba_workers"));
  auto& report = correction.report;
  report["enabled"] = true;
  report["source_model"] = original.value("selected_model", std::string("aligned"));
  std::vector<fs::path> products{work / "cache/shutter.bin", work / "cache/shutter.json",
                                 work / "cache/shutter_result.json"};
  const auto& selected =
      (correction.accepted || correction.rig_refined) ? correction.candidate : model;
  auto result = PostprocessResult(scene, selected, original, "shutter");
  result["shutter"] = report;
  auto saved = report;
  saved["motion"] = MotionToJson(correction.motion, correction.candidate);
  WriteModel(work / "cache/shutter.bin", selected);
  WriteJson(work / "cache/shutter.json", saved);
  WriteJson(work / "cache/shutter_result.json", result);
  Log("Rolling-shutter correction " + report.at("reason").get<std::string>());
  return products;
}
std::vector<fs::path> CleanSelected(const Scene& scene, const fs::path& work, const Json& settings,
                                    WorkState& state) {
  auto original = settings.at("rolling_shutter").get<bool>()
                      ? ReadJson(work / "cache/shutter_result.json")
                      : ReadJson(work / "cache/alignment.json");
  if (!original.contains("shutter"))
    original["shutter"] = {{"enabled", false}, {"used", false},
                           {"reason", "disabled by configuration"}};
  const auto model = ReadModel(SelectedModelPath(work, original));
  auto cleanup = CleanSparseStructure(scene, model, settings.at("workers"), Limits(settings));
  cleanup.report["source_model"] = original.value("selected_model", std::string("aligned"));
  const auto& selected = cleanup.accepted ? cleanup.candidate : model;
  auto result = PostprocessResult(scene, selected, original, "cleaned");
  result["cleanup"] = cleanup.report;
  WriteModel(work / "cache/cleaned.bin", selected);
  WriteJson(work / "cache/cleanup.json", cleanup.report);
  WriteJson(work / "result.json", result);
  state.value["registration"] = result.at("quality").at("registration");
  Log("Sparse structure cleanup " + cleanup.report.at("reason").get<std::string>());
  return {work / "cache/cleaned.bin", work / "cache/cleanup.json", work / "result.json"};
}
// Runs one stage and returns the files that make up its result.
std::vector<fs::path> RunStage(const std::string& stage, const Scene& scene, const fs::path& work,
                               const fs::path& root, const Json& settings, const fs::path& output,
                               const Delivery& delivery, WorkState& state) {
  if (stage == "prepare") {
    NavigationPositions(scene);
    return {work / "scene.json"};
  }
  if (stage == "masks") {
    WriteJson(work / "cache/masks_profile.json", MakeMasks(scene, work, root));
    return {work / "cache/masks", work / "cache/masks_profile.json"};
  }
  if (stage == "features") {
    WriteJson(work / "cache/features_profile.json", MakeFeatures(scene, work, root));
    return {work / "cache/features", work / "cache/features_profile.json"};
  }
  if (stage == "matches") {
    WriteJson(work / "cache/matches_profile.json", MakeMatches(scene, work, root));
    return {MatchesPath(work), work / "cache/pairs.json", work / "cache/matches_profile.json"};
  }
  if (stage == "sfm")
    return ReconstructVisual(scene, work, root, settings, state);
  if (stage == "align")
    return AlignVisual(scene, work, settings);
  if (stage == "shutter")
    return CorrectShutter(scene, work, settings);
  if (stage == "cleanup")
    return CleanSelected(scene, work, settings, state);
  ExportSelected(scene, work, output, delivery);
  return {output / "manifest.json"};
}
std::vector<std::string> Stages(const Json& settings, const std::string& requested) {
  std::vector<std::string> stages = {"prepare", "masks", "features", "matches", "sfm", "align"};
  if (settings.at("rolling_shutter").get<bool>())
    stages.push_back("shutter");
  else if (requested == "shutter")
    throw std::runtime_error("Rolling-shutter correction is disabled by configuration");
  stages.push_back("cleanup");
  stages.push_back("export");
  if (requested != "all" && std::find(stages.begin(), stages.end(), requested) == stages.end())
    throw std::runtime_error("Unknown stage: " + requested);
  return stages;
}
// The model package a work directory was reconstructed with, else the default package.
fs::path RecordedModels(const Json& state) {
  return state.contains("models") ? fs::u8path(state.at("models").get<std::string>())
                                  : DefaultModels();
}
// Production runs keep only the delivery: their work directory sits beside it and is removed
// once the delivery is published. --debug keeps the work directory and its cache.
fs::path ProductionWork(const fs::path& output) {
  auto work = output;
  work += ".work";
  return work;
}
}  // namespace
void Run(const Arguments& args) {
  args.Validate(WithDeliveryOptions({"--work", "--models", "--config", "--stage", "--output",
                                     "--project", "--images", "--camera-pos", "--pos", "--pos-enh",
                                     "--las", "--debug"}));
  auto delivery = ParseDelivery(args);
  auto inputs = ParseInputs(args);
  // Without inputs, an existing work directory supplies the inputs and output.
  auto output = args.Get("--output").empty() ? fs::path()
                                             : NormalizeDirectory(args.Path("--output"));
  fs::path work;
  if (!args.Get("--work").empty())
    work = args.Path("--work");
  else {
    if (output.empty())
      throw std::invalid_argument("Missing --output");
    work = args.Get("--debug") == "true" ? fs::absolute(fs::path("work") / output.filename())
                                         : ProductionWork(output);
  }
  const auto state_path = work / "state.json";
  const bool fresh = !fs::exists(state_path);
  if (fresh && !inputs)
    throw std::invalid_argument(
        "A new run needs --project, --images, --camera-pos, --pos, --pos-enh and --las");
  if (fresh && output.empty())
    throw std::invalid_argument("Missing --output");
  const auto saved = fresh ? Json() : ReadJson(state_path);
  if (!inputs)
    inputs = SceneInputs::FromJson(saved.at("scene").at("inputs"));
  if (output.empty()) {
    if (!saved.contains("output"))
      throw std::runtime_error("Work directory has no recorded output; specify --output");
    output = NormalizeDirectory(fs::u8path(saved.at("output").get<std::string>()));
  }
  ValidateDeliveryPaths(*inputs, work, output);
  // A finished production run leaves no work directory; do not repeat it over its delivery.
  if (fresh && fs::exists(output))
    throw std::runtime_error("Output already exists; choose a new output directory");
  // A production run will remove its work directory. Do not adopt existing files as fresh work.
  if (fresh && fs::is_directory(work)) {
    for (const auto& entry : fs::directory_iterator(work)) {
      const auto name = entry.path().filename();
      if (name != ".lock" && name != "state.json.writing")
        throw std::runtime_error("Work directory contains existing files. Choose a new directory.");
    }
  }
  bool debug = false, finished = false;
  Json summary;
  const auto started = Clock::now();
  {
    fs::create_directories(work);
    Lock lock(work);
    WorkState state{state_path, Json::object()};
    if (fs::exists(state.path))
      state.value = ReadJson(state.path);
    auto settings = args.Get("--config").empty()
                        ? (state.value.empty() ? Settings({}) : state.value.at("settings"))
                        : Settings(args.Path("--config"));
    const Scene scene = ReadScene(*inputs);
    ReadSceneLas(scene);
    const auto prepared = scene.ToJson();
    if (state.value.empty()) {
      state.value = {{"settings", settings}, {"scene", prepared}, {"stages", Json::object()}};
    } else if (state.value.at("scene") != prepared || state.value.at("settings") != settings)
      throw std::runtime_error("Inputs/settings changed; choose a new work directory");
    const auto root = args.Get("--models").empty()
                          ? (state.value.contains("models")
                                 ? fs::u8path(state.value.at("models").get<std::string>())
                                 : DefaultModels())
                          : args.Path("--models");
    const auto requested = args.Get("--stage", args.command == "prepare" ? "prepare" : "all");
    const auto stages = Stages(settings, requested);
    // Only runs that reach the export need the geometry engine; fail before reconstruction.
    if (requested == "all" || requested == "export")
      ResolveSupervisionModels(delivery.supervision, root);
    debug = args.Get("--debug") == "true" || state.value.value("debug", false);
    state.value["debug"] = debug;
    state.value["output"] = output.u8string();
    state.value["status"] = "running";
    state.value["build"] = BuildInfo();
    state.value["error"] = nullptr;
    state.Save();
    try {
      if (!state.value.at("stages").contains("prepare"))
        WriteJson(work / "scene.json", prepared);
      for (const auto& stage : stages) {
        if (requested != "all" && stage != requested)
          continue;
        auto& completed = state.value.at("stages");
        if (stage != "prepare") {
          const auto& previous = *(std::find(stages.begin(), stages.end(), stage) - 1);
          if (!completed.contains(previous))
            throw std::runtime_error("Run prerequisite stage: " + previous);
          if (stage == "align" || stage == "shutter" || stage == "cleanup" || stage == "export")
            Verify(work, completed.at(previous).at("files"));
          // SfM repair rounds run the matcher again with the recorded models.
          if (stage == "masks" || stage == "features" || stage == "matches" ||
              (stage == "sfm" && settings.at("sfm_repair_rounds").get<int>() > 0))
            RecordModels(state, root);
        }
        Log("Stage: " + stage);
        state.value["active_stage"] = stage;
        state.Save();
        const auto begin = Clock::now();
        if (completed.contains(stage)) {
          Verify(work, completed.at(stage).at("files"));
          if (stage == "export")
            ExportSelected(scene, work, output, delivery);
          Log("Reused " + stage);
          continue;
        }
        const auto products = RunStage(stage, scene, work, root, settings, output, delivery, state);
        state.value["stages"][stage] = {{"seconds", Seconds(begin)},
                                        {"files", Inventory(work, products)}};
        state.Save();
        Log(stage + ": completed in " + std::to_string(Seconds(begin)) + " s");
      }
      state.value["status"] = state.value.at("stages").contains("export") ? "completed" : "ready";
      state.value["active_stage"] = nullptr;
      state.value["last_invocation_seconds"] = Seconds(started);
      state.value["resources"] = Resources();
      state.Save();
    } catch (const std::exception& e) {
      state.value["status"] = "failed";
      state.value["error"] = e.what();
      state.Save();
      throw;
    }
    finished = state.value.at("status") == "completed";
    summary = {{"status", state.value.at("status")},
               {"registration", state.value.value("registration", Json(nullptr))}};
  }
  // The lock is released; a published production delivery no longer needs its work directory.
  if (finished && !debug) {
    RemoveAll(work);
    Log("Removed work directory " + work.u8string());
  }
  summary["output"] = output.u8string();
  summary["work"] = debug || !finished ? Json(work.u8string()) : Json(nullptr);
  summary["seconds"] = Seconds(started);
  std::cout << summary.dump() << '\n';
}
void Export(const Arguments& args) {
  args.Validate(WithDeliveryOptions({"--work", "--output"}));
  auto delivery = ParseDelivery(args);
  const auto work = args.Path("--work");
  if (!fs::is_regular_file(work / "scene.json"))
    throw std::runtime_error("Not a work directory (no scene.json): " + work.u8string());
  Lock lock(work);
  const auto state = ReadJson(work / "state.json");
  ResolveSupervisionModels(delivery.supervision, RecordedModels(state));
  if (!state.at("stages").contains("cleanup"))
    throw std::runtime_error("Run prerequisite stage: cleanup");
  Verify(work, state.at("stages").at("cleanup").at("files"));
  const auto scene = Scene::FromJson(ReadJson(work / "scene.json"));
  const auto output = NormalizeDirectory(args.Path("--output"));
  ValidateDeliveryPaths(scene, work, output);
  // The ECEF frame conversion is refitted from the navigation file: it must be the one reconstructed.
  if (ReadScene(scene).ToJson() != scene.ToJson())
    throw std::runtime_error("Source inputs changed; reconstruct in a new work directory");
  std::cout << ExportSelected(scene, work, output, delivery).dump() << '\n';
}
}  // namespace geosun

#include "app/pipeline.h"
#include "io/scene.h"
#include "io/las.h"
#include "io/colmap.h"
#include "io/cache.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "sfm/reconstruction.h"
#include "sfm/photometric.h"
#include "sfm/structure.h"
#include "sfm/shutter.h"
#include "alignment/alignment.h"
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace geosun {
// Compiled from configs/default.json.
extern const char DefaultSettings[];
namespace {
bool Contains(const fs::path& parent, const fs::path& child) {
  const auto p = fs::weakly_canonical(parent), c = fs::weakly_canonical(child);
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
class Lock {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
#endif
 public:
  explicit Lock(const fs::path& work) {
#ifdef _WIN32
    handle = CreateFileW((work / ".lock").c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Work directory is in use or cannot be locked");
#endif
  }
  ~Lock() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE)
      CloseHandle(handle);
#endif
  }
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
  for (const auto* key : {"workers", "ba_workers"})
    if (!result.at(key).is_number_integer() || result.at(key).get<int>() < 1 ||
        result.at(key).get<int>() > 32)
      throw std::runtime_error(std::string(key) + " must be 1-32");
  if (!result.at("sfm_repair_rounds").is_number_integer() || result.at("sfm_repair_rounds") < 0 ||
      result.at("sfm_repair_rounds") > 2)
    throw std::runtime_error("sfm_repair_rounds must be an integer from 0 to 2");
  for (const auto* key : {"refine", "rolling_shutter", "enhanced_photos"})
    if (!result.at(key).is_boolean())
      throw std::runtime_error(std::string(key) + " must be a boolean");
  for (const auto& [key, value] : result.items())
    if (key != "workers" && key != "calibration_tracks" && key != "sfm_repair_rounds" &&
        key != "refine" && key != "rolling_shutter" && key != "enhanced_photos" &&
        (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() <= 0))
      throw std::runtime_error("Expected positive numeric setting: " + key);
  if (result.at("calibration_tracks") != "all" && result.at("calibration_tracks") != "stable")
    throw std::runtime_error("calibration_tracks must be all or stable");
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
// Delivery options shared by run, export, refine and cleanup.
struct Delivery {
  bool ply;
  double voxel;
  SupervisionOptions supervision;
  bool crop;
};
// The options ParseDelivery reads, accepted by every command that writes a delivery.
std::vector<std::string> WithDeliveryOptions(std::vector<std::string> options) {
  for (const auto* option :
       {"--no-ply", "--ply-voxel", "--las-supervision", "--no-las-supervision",
        "--supervision-max-size", "--supervision-boundary", "--supervision-mask-dir",
        "--supervision-cache-mb", "--crop-valid-domain", "--no-crop-valid-domain"})
    options.push_back(option);
  return options;
}
Delivery ParseDelivery(const Arguments& args) {
  if (args.values.count("--crop-valid-domain") && args.values.count("--no-crop-valid-domain"))
    throw std::runtime_error("--crop-valid-domain conflicts with --no-crop-valid-domain");
  return {args.Get("--no-ply") != "true", args.Number("--ply-voxel", .05),
          ParseSupervisionOptions(args), args.Get("--no-crop-valid-domain") != "true"};
}
Json Inventory(const fs::path& work, const std::vector<fs::path>& products) {
  Json result = Json::object();
  for (const auto& path : products) {
    if (fs::is_directory(path)) {
      for (const auto& row : fs::recursive_directory_iterator(path))
        if (row.is_regular_file())
          result[fs::relative(row.path(), work).generic_u8string()] = FileMetadata(row.path());
    } else
      result[fs::relative(path, work).generic_u8string()] = FileMetadata(path);
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
void ExportSelected(const Scene& scene, const fs::path& work, const fs::path& output,
                    const Delivery& delivery) {
  const auto result = ReadJson(work / "result.json");
  ExportBundle(scene, ReadModel(SelectedModelPath(work, result)), work, output, delivery.ply,
               delivery.voxel, result, delivery.supervision, delivery.crop);
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
        Inventory(work, {work / "cache/matches", work / "cache/pairs.json", profile_path});
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
  result["input"] = scene.input.u8string();
  WriteModel(work / "cache/aligned.bin", model);
  WriteJson(work / "cache/alignment.json", result);
  return {work / "cache/aligned.bin", work / "cache/alignment.json"};
}
std::vector<fs::path> RefineAligned(const Scene& scene, const fs::path& work,
                                    const Json& settings) {
  const auto model = ReadModel(work / "cache/aligned.bin");
  const auto original = ReadJson(work / "cache/alignment.json");
  auto refinement = RefinePhotometric(scene, model, settings.at("workers"), Limits(settings),
                                      settings.at("ba_workers"));
  refinement.report["enabled"] = true;
  const auto& selected = refinement.accepted ? refinement.candidate : model;
  auto result = PostprocessResult(scene, selected, original, "refined");
  result["refinement"] = refinement.report;
  WriteModel(work / "cache/refinement_candidate.bin", refinement.candidate);
  WriteModel(work / "cache/refined.bin", selected);
  WriteJson(work / "cache/refinement.json", refinement.report);
  WriteJson(work / "cache/refinement_result.json", result);
  Log("Photometric refinement " + refinement.report.at("reason").get<std::string>());
  return {work / "cache/refinement_candidate.bin", work / "cache/refined.bin",
          work / "cache/refinement.json", work / "cache/refinement_result.json"};
}
// Result of the camera stages before rolling-shutter correction: alignment or refinement.
Json CameraResult(const fs::path& work, const Json& settings) {
  const bool photo = settings.at("refine");
  auto original =
      ReadJson(work / (photo ? "cache/refinement_result.json" : "cache/alignment.json"));
  if (!photo)
    original["refinement"] = {
        {"enabled", false}, {"accepted", false}, {"reason", "disabled by configuration"}};
  return original;
}
std::vector<fs::path> CorrectShutter(const Scene& scene, const fs::path& work,
                                     const Json& settings) {
  const auto original = CameraResult(work, settings);
  const auto model = ReadModel(SelectedModelPath(work, original));
  auto correction = CorrectRollingShutter(scene, model, settings.at("workers"), Limits(settings),
                                          settings.at("ba_workers"));
  auto& report = correction.report;
  report["enabled"] = true;
  report["source_model"] = original.value("selected_model", std::string("aligned"));
  std::vector<fs::path> products{work / "cache/shutter_candidate.bin", work / "cache/shutter.bin",
                                 work / "cache/shutter.json", work / "cache/shutter_result.json"};
  const auto& selected = correction.accepted ? correction.candidate : model;
  auto result = PostprocessResult(scene, selected, original, "shutter");
  result["shutter"] = report;
  auto saved = report;
  saved["motion"] = MotionToJson(correction.motion, correction.candidate);
  WriteModel(work / "cache/shutter_candidate.bin", correction.candidate);
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
                      : CameraResult(work, settings);
  if (!original.contains("shutter"))
    original["shutter"] = {{"enabled", false}, {"used", false},
                           {"reason", "disabled by configuration"}};
  const auto model = ReadModel(SelectedModelPath(work, original));
  auto cleanup = CleanSparseStructure(scene, model, settings.at("workers"), Limits(settings));
  cleanup.report["source_model"] = original.value("selected_model", std::string("aligned"));
  const auto& selected = cleanup.accepted ? cleanup.candidate : model;
  auto result = PostprocessResult(scene, selected, original, "cleaned");
  result["cleanup"] = cleanup.report;
  WriteModel(work / "cache/cleanup_candidate.bin", cleanup.candidate);
  WriteModel(work / "cache/cleaned.bin", selected);
  WriteJson(work / "cache/cleanup.json", cleanup.report);
  WriteJson(work / "result.json", result);
  state.value["registration"] = result.at("quality").at("registration");
  Log("Sparse structure cleanup " + cleanup.report.at("reason").get<std::string>());
  return {work / "cache/cleanup_candidate.bin", work / "cache/cleaned.bin",
          work / "cache/cleanup.json", work / "result.json"};
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
    return {work / "cache/matches", work / "cache/pairs.json", work / "cache/matches_profile.json"};
  }
  if (stage == "sfm")
    return ReconstructVisual(scene, work, root, settings, state);
  if (stage == "align")
    return AlignVisual(scene, work, settings);
  if (stage == "refine")
    return RefineAligned(scene, work, settings);
  if (stage == "shutter")
    return CorrectShutter(scene, work, settings);
  if (stage == "cleanup")
    return CleanSelected(scene, work, settings, state);
  ExportSelected(scene, work, output, delivery);
  return {output / "manifest.json"};
}
std::vector<std::string> Stages(const Json& settings, const std::string& requested) {
  std::vector<std::string> stages = {"prepare", "masks", "features", "matches", "sfm", "align"};
  if (settings.at("refine").get<bool>())
    stages.push_back("refine");
  else if (requested == "refine")
    throw std::runtime_error("Refinement is disabled; enable --refine in a new work directory");
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
// Production runs keep only the delivery: their work directory sits beside it and is removed
// once the delivery is published. --debug keeps the work directory and its cache.
fs::path ProductionWork(const fs::path& output) {
  auto work = output;
  work += ".work";
  return work;
}
}  // namespace
void Run(const Arguments& args) {
  args.Validate(WithDeliveryOptions({"--input", "--work", "--models", "--config", "--stage",
                                     "--output", "--las", "--refine", "--debug"}));
  const auto delivery = ParseDelivery(args);
  // Without --input, an existing work directory supplies the input and output.
  auto output = args.Get("--output").empty() ? fs::path() : args.Path("--output");
  fs::path work;
  if (!args.Get("--work").empty())
    work = args.Path("--work");
  else {
    const auto name = args.Path("--input").filename();
    if (output.empty())
      output = fs::absolute(fs::path("output") / name);
    work = args.Get("--debug") == "true" ? fs::absolute(fs::path("work") / name)
                                         : ProductionWork(output);
  }
  const bool fresh = !fs::exists(work / "state.json");
  if (output.empty() && !fresh) {
    const auto saved = ReadJson(work / "state.json");
    if (!saved.contains("output"))
      throw std::runtime_error("Work directory has no recorded output; specify --output");
    output = fs::u8path(saved.at("output").get<std::string>());
  }
  if (output.empty())
    output = fs::absolute(fs::path("output") / args.Path("--input").filename());
  if (Contains(work, output) || Contains(output, work))
    throw std::runtime_error("Output and work directories must be separate");
  // A finished production run leaves no work directory; do not repeat it over its delivery.
  if (fresh && fs::exists(output))
    throw std::runtime_error("Output already exists; choose a new output directory");
  bool debug = false, finished = false;
  Json summary;
  const auto started = Clock::now();
  {
    fs::create_directories(work);
    Lock lock(work);
    WorkState state{work / "state.json", Json::object()};
    if (fs::exists(state.path))
      state.value = ReadJson(state.path);
    if (!state.value.empty() && state.value.at("version") != 1)
      throw std::runtime_error("Unsupported work format; choose a new work directory");
    if (!state.value.empty() && state.value.value("pipeline_version", 0) != PipelineVersion)
      throw std::runtime_error(
          "Reconstruction implementation changed; choose a new work directory");
    const auto input = args.Get("--input").empty()
                           ? fs::u8path(state.value.at("input").get<std::string>())
                           : args.Path("--input");
    if (Contains(input, work) || Contains(work, input))
      throw std::runtime_error("Input and work directories must be separate");
    auto settings = args.Get("--config").empty()
                        ? (state.value.empty() ? Settings({}) : state.value.at("settings"))
                        : Settings(args.Path("--config"));
    if (args.Get("--refine") == "true")
      settings["refine"] = true;
    const auto las = args.Get("--las").empty()
                         ? (state.value.empty()
                                ? fs::path()
                                : fs::u8path(state.value.at("scene").at("las").get<std::string>()))
                         : args.Path("--las");
    // Work directories frozen before this setting existed read as original photos and
    // fail the scene comparison below if they had used enhanced copies.
    const Scene scene = ReadScene(input, las, settings.value("enhanced_photos", false));
    ReadLasHeader(scene.las);
    const auto prepared = scene.ToJson();
    if (state.value.empty()) {
      state.value = {{"version", 1},
                     {"pipeline_version", PipelineVersion},
                     {"input", scene.input.u8string()},
                     {"settings", settings},
                     {"scene", prepared},
                     {"stages", Json::object()}};
      WriteJson(work / "scene.json", prepared);
    } else if (state.value.at("scene") != prepared || state.value.at("settings") != settings)
      throw std::runtime_error("Inputs/settings changed; choose a new work directory");
    const auto root = args.Get("--models").empty()
                          ? (state.value.contains("models")
                                 ? fs::u8path(state.value.at("models").get<std::string>())
                                 : DefaultModels())
                          : args.Path("--models");
    const auto requested = args.Get("--stage", args.command == "prepare" ? "prepare" : "all");
    const auto stages = Stages(settings, requested);
    if (Contains(input, output) || Contains(output, input))
      throw std::runtime_error("Output must be separate from input");
    // Work directories from before this option kept their cache; they stay debug runs.
    debug = args.Get("--debug") == "true" || (!fresh && state.value.value("debug", true));
    state.value["debug"] = debug;
    state.value["output"] = output.u8string();
    state.value["status"] = "running";
    state.value["build"] = BuildInfo();
    state.value["error"] = nullptr;
    state.Save();
    try {
      for (const auto& stage : stages) {
        if (requested != "all" && stage != requested)
          continue;
        auto& completed = state.value.at("stages");
        if (stage != "prepare") {
          const auto& previous = *(std::find(stages.begin(), stages.end(), stage) - 1);
          if (!completed.contains(previous))
            throw std::runtime_error("Run prerequisite stage: " + previous);
          if (stage == "refine" || stage == "shutter" || stage == "cleanup" || stage == "export")
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
    fs::remove_all(work);
    Log("Removed work directory " + work.u8string());
  }
  summary["output"] = output.u8string();
  summary["work"] = debug || !finished ? Json(work.u8string()) : Json(nullptr);
  summary["seconds"] = Seconds(started);
  std::cout << summary.dump() << '\n';
}
void Export(const Arguments& args) {
  args.Validate(WithDeliveryOptions({"--work", "--output"}));
  const auto delivery = ParseDelivery(args);
  const auto work = args.Path("--work");
  Lock lock(work);
  if (fs::exists(work / "state.json")) {
    const auto state = ReadJson(work / "state.json");
    if (state.value("pipeline_version", 0) == PipelineVersion) {
      if (!state.at("stages").contains("cleanup"))
        throw std::runtime_error("Run prerequisite stage: cleanup");
      Verify(work, state.at("stages").at("cleanup").at("files"));
    }
  }
  const auto scene = Scene::FromJson(ReadJson(work / "scene.json"));
  const auto output = args.Path("--output");
  if (Contains(scene.input, output) || Contains(output, scene.input))
    throw std::runtime_error("Output must be separate from input");
  if (Contains(work, output) || Contains(output, work))
    throw std::runtime_error("Output and work directories must be separate");
  ExportSelected(scene, work, output, delivery);
  std::cout << CheckBundle(output).dump() << '\n';
}
void Postprocess(const Arguments& args) {
  args.Validate(
      WithDeliveryOptions({"--work", "--output", "--workers", "--new-work", "--debug"}));
  const auto delivery = ParseDelivery(args);
  const bool photo = args.command == "refine";
  const bool debug = args.Get("--debug") == "true";
  const auto source = args.Path("--work"), delivery_output = args.Path("--output");
  // Like run: the new work directory is kept only with --debug.
  const auto final_output =
      args.Get("--new-work").empty() ? ProductionWork(delivery_output) : args.Path("--new-work");
  if (fs::exists(delivery_output))
    throw std::runtime_error("Output already exists; choose a new output directory");
  if (Contains(final_output, delivery_output) || Contains(delivery_output, final_output) ||
      Contains(source, delivery_output) || Contains(delivery_output, source))
    throw std::runtime_error("Output must be outside the source and new work directories");
  const int workers = args.Integer("--workers", 4);
  if (workers > 32)
    throw std::runtime_error("workers must be 1-32");
  if (fs::exists(final_output))
    throw std::runtime_error("Postprocessing output already exists; choose a new work directory");
  Lock lock(source);
  const auto scene = Scene::FromJson(ReadJson(source / "scene.json"));
  if (Contains(source, final_output) || Contains(scene.input, final_output))
    throw std::runtime_error("Postprocessing output must be outside source work and input");
  // Build the whole work directory beside its destination and publish it only when complete.
  auto output = final_output;
  output += ".pending";
  if (Contains(source, output) || Contains(output, source) || Contains(scene.input, output) ||
      Contains(output, scene.input))
    throw std::runtime_error("Postprocessing .pending directory overlaps source work or input");
  ArchiveInterrupted(output);
  if (ReadScene(scene.input, scene.las, scene.enhanced).ToJson() != scene.ToJson())
    throw std::runtime_error("Source inputs changed; reconstruct in a new work directory");
  const auto original = ReadJson(source / "result.json");
  const bool corrected =
      original.contains("shutter") && original.at("shutter").value("used", false);
  if (photo && corrected)
    throw std::runtime_error(
        "Photometric refinement needs the original photo geometry; the source model is "
        "rolling-shutter corrected");
  const auto model = ReadModel(SelectedModelPath(source, original));
  AlignmentLimits limits;
  if (original.contains("alignment")) {
    const auto& saved = original.at("alignment").at("limits");
    limits.support_fraction = saved.at("support_fraction");
    limits.median_m = saved.at("median_m");
    limits.p90_m = saved.at("p90_m");
  } else if (original.contains("refinement") && original.at("refinement").contains("limits")) {
    const auto& saved = original.at("refinement").at("limits");
    limits.support_fraction = saved.at("las_support_fraction");
    limits.median_m = saved.at("las_median_m");
    limits.p90_m = saved.at("las_p90_m");
  }
  limits.median_px = original.at("quality").at("limits").at("median_px");
  limits.p90_px = original.at("quality").at("limits").at("p90_px");
  fs::create_directories(output / "cache");
  Reconstruction selected;
  Json report;
  const std::string selected_name = photo ? "refined" : "cleaned";
  const std::string report_name = photo ? "refinement" : "cleanup";
  bool accepted;
  if (photo) {
    auto refinement = RefinePhotometric(scene, model, workers, limits);
    accepted = refinement.accepted;
    report = std::move(refinement.report);
    report["enabled"] = true;
    selected = accepted ? refinement.candidate : model;
    WriteModel(output / "cache/refinement_candidate.bin", refinement.candidate);
  } else {
    auto cleanup = CleanSparseStructure(scene, model, workers, limits);
    accepted = cleanup.accepted;
    report = std::move(cleanup.report);
    report["source_model"] = original.value("selected_model", std::string("aligned"));
    selected = accepted ? cleanup.candidate : model;
    WriteModel(output / "cache/cleanup_candidate.bin", cleanup.candidate);
  }
  report["source_work"] = source.u8string();
  auto result = PostprocessResult(scene, selected, original, selected_name);
  if (photo)
    result.erase("cleanup");
  result[report_name] = report;
  WriteModel(output / "cache/aligned.bin", model);
  WriteJson(output / "cache/alignment.json", original);
  WriteModel(output / "cache" / (selected_name + ".bin"), selected);
  WriteJson(output / "cache" / (report_name + ".json"), report);
  WriteJson(output / "scene.json", scene.ToJson());
  WriteJson(output / "result.json", result);
  // Rolling-shutter corrected models carry their readout motion.
  if (corrected)
    fs::copy_file(source / "cache/shutter.json", output / "cache/shutter.json");
  const std::array<cv::Mat, 2> domains{MaskDomain(scene.cameras[0]), MaskDomain(scene.cameras[1])};
  for (const auto& image : scene.images) {
    if (!selected.images[image.id].registered)
      continue;
    const auto name = std::to_string(image.id) + ".png";
    const auto mask = output / "cache/masks" / name;
    fs::create_directories(mask.parent_path());
    auto pixels = ReadImage(source / "cache/masks" / name, cv::IMREAD_GRAYSCALE);
    pixels.setTo(0, domains[image.camera] == 0);
    WriteImage(mask, pixels);
  }
  if (Contains(scene.input, delivery_output) || Contains(delivery_output, scene.input))
    throw std::runtime_error("Output must be separate from input");
  ExportBundle(scene, selected, output, delivery_output, delivery.ply, delivery.voxel, result,
               delivery.supervision, delivery.crop);
  if (debug)
    fs::rename(output, final_output);
  else
    fs::remove_all(output);
  Log(args.command + ": " + report.at("reason").get<std::string>());
  std::cout << Json({{"output", delivery_output.u8string()},
                     {"work", debug ? Json(final_output.u8string()) : Json(nullptr)},
                     {"accepted", accepted},
                     {"reason", report.at("reason")}})
                   .dump(2)
            << '\n';
}
}  // namespace geosun

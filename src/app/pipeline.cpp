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
#include "alignment/alignment.h"
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace geosun {
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
  Json result = {{"workers", 4},         {"sfm_repair_rounds", 0},      {"quality_median_px", 2.},
                 {"quality_p90_px", 4.}, {"las_support_fraction", .25}, {"las_median_m", .1},
                 {"las_p90_m", .3},      {"calibration_tracks", "all"}, {"refine", false}};
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
  if (!result.at("workers").is_number_integer() || result.at("workers").get<int>() < 1 ||
      result.at("workers").get<int>() > 32)
    throw std::runtime_error("workers must be 1-32");
  if (!result.at("sfm_repair_rounds").is_number_integer() || result.at("sfm_repair_rounds") < 0 ||
      result.at("sfm_repair_rounds") > 2)
    throw std::runtime_error("sfm_repair_rounds must be an integer from 0 to 2");
  if (!result.at("refine").is_boolean())
    throw std::runtime_error("refine must be a boolean");
  for (const auto& [key, value] : result.items())
    if (key != "workers" && key != "calibration_tracks" && key != "sfm_repair_rounds" &&
        key != "refine" &&
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
}  // namespace
void Run(const Arguments& args) {
  args.Validate({"--input", "--work", "--models", "--config", "--stage", "--output", "--las",
                 "--no-ply", "--ply-voxel", "--las-supervision", "--no-las-supervision",
                 "--supervision-max-size", "--supervision-boundary", "--supervision-mask-dir",
                 "--refine"});
  const auto supervision = ParseSupervisionOptions(args);
  const auto work = args.Get("--work").empty()
                        ? fs::absolute(fs::path("work") / args.Path("--input").filename())
                        : args.Path("--work");
  fs::create_directories(work);
  Lock lock(work);
  const auto state_path = work / "state.json";
  Json state = fs::exists(state_path) ? ReadJson(state_path) : Json::object();
  if (!state.empty() && state.at("version") != 1)
    throw std::runtime_error("Unsupported work format; choose a new work directory");
  if (!state.empty() && state.value("pipeline_version", 0) != PipelineVersion)
    throw std::runtime_error("Reconstruction implementation changed; choose a new work directory");
  const auto input = args.Get("--input").empty() ? fs::u8path(state.at("input").get<std::string>())
                                                 : args.Path("--input");
  if (Contains(input, work) || Contains(work, input))
    throw std::runtime_error("Input and work directories must be separate");
  auto settings = args.Get("--config").empty()
                      ? (state.empty() ? Settings({}) : state.at("settings"))
                      : Settings(args.Path("--config"));
  if (args.Get("--refine") == "true")
    settings["refine"] = true;
  const auto las =
      args.Get("--las").empty()
          ? (state.empty() ? fs::path()
                           : fs::u8path(state.at("scene").at("las").get<std::string>()))
          : args.Path("--las");
  const Scene scene = ReadScene(input, las);
  ReadLasHeader(scene.las);
  const auto prepared = scene.ToJson();
  if (state.empty()) {
    state = {{"version", 1},
             {"pipeline_version", PipelineVersion},
             {"input", scene.input.u8string()},
             {"settings", settings},
             {"scene", prepared},
             {"stages", Json::object()}};
    WriteJson(work / "scene.json", prepared);
  } else if (state.at("scene") != prepared || state.at("settings") != settings)
    throw std::runtime_error("Inputs/settings changed; choose a new work directory");
  const auto root =
      args.Get("--models").empty()
          ? (state.contains("models") ? fs::u8path(state.at("models").get<std::string>())
                                      : DefaultModels())
          : args.Path("--models");
  const auto requested = args.Get("--stage", args.command == "prepare" ? "prepare" : "all");
  std::vector<std::string> stages = {"prepare", "masks", "features", "matches", "sfm", "align"};
  if (settings.at("refine").get<bool>())
    stages.push_back("refine");
  else if (requested == "refine")
    throw std::runtime_error("Refinement is disabled; enable --refine in a new work directory");
  stages.push_back("cleanup");
  stages.push_back("export");
  if (requested != "all" && std::find(stages.begin(), stages.end(), requested) == stages.end())
    throw std::runtime_error("Unknown stage: " + requested);
  const auto output = args.Get("--output").empty() ? work / "colmap" : args.Path("--output");
  if (Contains(input, output) || Contains(output, input))
    throw std::runtime_error("Output must be separate from input");
  const bool ply = args.Get("--no-ply") != "true";
  const double voxel = args.Number("--ply-voxel", .05);
  state["status"] = "running";
  state["build"] = BuildInfo();
  state["error"] = nullptr;
  WriteJson(state_path, state);
  const auto started = Clock::now();
  const int workers = settings.at("workers");
  const AlignmentLimits limits{settings.at("las_support_fraction"), settings.at("las_median_m"),
                               settings.at("las_p90_m"), settings.at("quality_median_px"),
                               settings.at("quality_p90_px")};
  try {
    for (const auto& stage : stages) {
      if (requested != "all" && stage != requested)
        continue;
      if (stage != "prepare") {
        const size_t index = std::find(stages.begin(), stages.end(), stage) - stages.begin();
        if (!state.at("stages").contains(stages[index - 1]))
          throw std::runtime_error("Run prerequisite stage: " + stages[index - 1]);
        if (stage == "refine" || stage == "cleanup" || stage == "export")
          Verify(work, state.at("stages").at(stages[index - 1]).at("files"));
        if (stage == "masks" || stage == "features" || stage == "matches") {
          Json identity = {{"manifest", FileMetadata(root / "manifest.json")}};
          const auto manifest = ReadJson(root / "manifest.json");
          for (const auto& [key, value] : manifest.at("models").items())
            identity[key] = FileMetadata(root / "engines" / (key + ".engine"));
          if (state.contains("model_identity") && state.at("model_identity") != identity)
            throw std::runtime_error("Models changed; choose a new work directory");
          state["model_identity"] = identity;
          state["models"] = root.u8string();
          WriteJson(state_path, state);
        }
      }
      Log("Stage: " + stage);
      state["active_stage"] = stage;
      WriteJson(state_path, state);
      const auto begin = Clock::now();
      if (state.at("stages").contains(stage)) {
        Verify(work, state.at("stages").at(stage).at("files"));
        if (stage == "export") {
          const auto result = ReadJson(work / "result.json");
          ExportBundle(scene, ReadModel(SelectedModelPath(work, result)), work, output, ply, voxel,
                       result, supervision);
        }
        Log("Reused " + stage);
        continue;
      }
      std::vector<fs::path> products;
      if (stage == "prepare") {
        NavigationPositions(scene);
        products = {work / "scene.json"};
      } else if (stage == "masks") {
        WriteJson(work / "cache/masks_profile.json", MakeMasks(scene, work, root));
        products = {work / "cache/masks", work / "cache/masks_profile.json"};
      } else if (stage == "features") {
        WriteJson(work / "cache/features_profile.json", MakeFeatures(scene, work, root));
        products = {work / "cache/features", work / "cache/features_profile.json"};
      } else if (stage == "matches") {
        WriteJson(work / "cache/matches_profile.json", MakeMatches(scene, work, root));
        products = {work / "cache/matches", work / "cache/pairs.json",
                    work / "cache/matches_profile.json"};
      } else if (stage == "sfm") {
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
            model =
                Reconstruct(scene, work, workers, report,
                            settings.at("calibration_tracks") == "stable", repair >= max_repairs);
            quality = model.Quality(scene, settings.at("quality_median_px"),
                                    settings.at("quality_p90_px"));
            if (quality.at("passed").get<bool>())
              break;
            for (uint32_t id = 0; id < model.images.size(); ++id) {
              if (!model.images[id].registered)
                continue;
              const auto& row = quality.at("images")[id];
              if (row.at("observations").get<size_t>() < 32 ||
                  row.at("occupied_cells").get<size_t>() < 8)
                deficient.insert(id);
            }
            if (deficient.empty() || repair >= max_repairs)
              throw std::runtime_error("Visual acceptance rejected: " +
                                       quality.at("failures").dump());
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
          state["stages"]["matches"]["files"] =
              Inventory(work, {work / "cache/matches", work / "cache/pairs.json", profile_path});
          WriteJson(state_path, state);
        }
        WriteModel(work / "cache/visual.bin", model);
        report["quality"] = quality;
        WriteJson(work / "cache/sfm.json", report);
        state["registration"] = quality.at("registration");
        if (model.Registered() != model.images.size())
          Log("Partial reconstruction accepted: " + std::to_string(model.Registered()) + "/" +
              std::to_string(model.images.size()) +
              "; see quality.registration for missing images");
        products = {work / "cache/visual.bin", work / "cache/sfm.json"};
      } else if (stage == "align") {
        auto model = ReadModel(work / "cache/visual.bin");
        auto result = Align(scene, model, workers, limits);
        result["sfm"] = ReadJson(work / "cache/sfm.json");
        for (const auto& name : {"masks", "features", "matches"})
          result["frontend"][name] =
              ReadJson(work / "cache" / (std::string(name) + "_profile.json"));
        result["runtime"] = "C++/CUDA/TensorRT";
        result["build"] = BuildInfo();
        result["input"] = scene.input.u8string();
        WriteModel(work / "cache/aligned.bin", model);
        WriteJson(work / "cache/alignment.json", result);
        products = {work / "cache/aligned.bin", work / "cache/alignment.json"};
      } else if (stage == "refine") {
        const auto model = ReadModel(work / "cache/aligned.bin");
        const auto original = ReadJson(work / "cache/alignment.json");
        auto refinement = RefinePhotometric(scene, model, workers, limits);
        refinement.report["enabled"] = true;
        const auto& selected = refinement.accepted ? refinement.candidate : model;
        auto result = PostprocessResult(scene, selected, original, "refined");
        result["refinement"] = refinement.report;
        WriteModel(work / "cache/refinement_candidate.bin", refinement.candidate);
        WriteModel(work / "cache/refined.bin", selected);
        WriteJson(work / "cache/refinement.json", refinement.report);
        WriteJson(work / "cache/refinement_result.json", result);
        Log("Photometric refinement " + refinement.report.at("reason").get<std::string>());
        products = {work / "cache/refinement_candidate.bin", work / "cache/refined.bin",
                    work / "cache/refinement.json", work / "cache/refinement_result.json"};
      } else if (stage == "cleanup") {
        const bool photo = settings.at("refine");
        auto original =
            ReadJson(work / (photo ? "cache/refinement_result.json" : "cache/alignment.json"));
        if (!photo)
          original["refinement"] = {
              {"enabled", false}, {"accepted", false}, {"reason", "disabled by configuration"}};
        const auto model = ReadModel(SelectedModelPath(work, original));
        auto cleanup = CleanSparseStructure(scene, model, workers, limits);
        cleanup.report["source_model"] = original.value("selected_model", std::string("aligned"));
        const auto& selected = cleanup.accepted ? cleanup.candidate : model;
        auto result = PostprocessResult(scene, selected, original, "cleaned");
        result["cleanup"] = cleanup.report;
        WriteModel(work / "cache/cleanup_candidate.bin", cleanup.candidate);
        WriteModel(work / "cache/cleaned.bin", selected);
        WriteJson(work / "cache/cleanup.json", cleanup.report);
        WriteJson(work / "result.json", result);
        state["registration"] = result.at("quality").at("registration");
        Log("Sparse structure cleanup " + cleanup.report.at("reason").get<std::string>());
        products = {work / "cache/cleanup_candidate.bin", work / "cache/cleaned.bin",
                    work / "cache/cleanup.json", work / "result.json"};
      } else {
        const auto result = ReadJson(work / "result.json");
        ExportBundle(scene, ReadModel(SelectedModelPath(work, result)), work, output, ply, voxel,
                     result, supervision);
        products = {output / "manifest.json"};
      }
      state["stages"][stage] = {{"seconds", Seconds(begin)}, {"files", Inventory(work, products)}};
      WriteJson(state_path, state);
      Log(stage + ": completed in " + std::to_string(Seconds(begin)) + " s");
    }
    state["status"] = state.at("stages").contains("export") ? "completed" : "ready";
    state["active_stage"] = nullptr;
    state["last_invocation_seconds"] = Seconds(started);
    state["resources"] = Resources();
    WriteJson(state_path, state);
  } catch (const std::exception& e) {
    state["status"] = "failed";
    state["error"] = e.what();
    WriteJson(state_path, state);
    throw;
  }
  std::cout << Json({{"work", work.u8string()},
                     {"status", state.at("status")},
                     {"registration", state.value("registration", Json(nullptr))},
                     {"seconds", Seconds(started)}})
                   .dump()
            << '\n';
}
void Export(const Arguments& args) {
  args.Validate({"--work", "--output", "--no-ply", "--ply-voxel", "--las-supervision",
                 "--no-las-supervision", "--supervision-max-size", "--supervision-boundary",
                 "--supervision-mask-dir"});
  const auto supervision = ParseSupervisionOptions(args);
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
  const auto result = ReadJson(work / "result.json");
  const auto model = ReadModel(SelectedModelPath(work, result));
  const auto output = args.Get("--output").empty() ? work / "colmap" : args.Path("--output");
  if (Contains(scene.input, output) || Contains(output, scene.input))
    throw std::runtime_error("Output must be separate from input");
  ExportBundle(scene, model, work, output, args.Get("--no-ply") != "true",
               args.Number("--ply-voxel", .05), result, supervision);
  std::cout << CheckBundle(output).dump() << '\n';
}
void Postprocess(const Arguments& args) {
  args.Validate({"--work", "--output", "--workers", "--las-supervision", "--no-las-supervision",
                 "--supervision-max-size", "--supervision-boundary", "--supervision-mask-dir"});
  const auto supervision = ParseSupervisionOptions(args);
  const bool photo = args.command == "refine";
  const auto source = args.Path("--work"), output = args.Path("--output");
  const int workers = args.Integer("--workers", 4);
  if (workers > 32)
    throw std::runtime_error("workers must be 1-32");
  if (fs::exists(output))
    throw std::runtime_error("Postprocessing output already exists; choose a new work directory");
  const auto scene = Scene::FromJson(ReadJson(source / "scene.json"));
  if (Contains(source, output) || Contains(scene.input, output))
    throw std::runtime_error("Postprocessing output must be outside source work and input");
  if (ReadScene(scene.input, scene.las).ToJson() != scene.ToJson())
    throw std::runtime_error("Source inputs changed; reconstruct in a new work directory");
  const auto original = ReadJson(source / "result.json");
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
  const std::array<cv::Mat, 2> domains{MaskDomain(scene.cameras[0]), MaskDomain(scene.cameras[1])};
  for (const auto& image : scene.images) {
    if (!selected.images[image.id].registered)
      continue;
    const auto mask = MaskPath(output, image.id);
    fs::create_directories(mask.parent_path());
    auto pixels = ReadImage(MaskPath(source, image.id), cv::IMREAD_GRAYSCALE);
    pixels.setTo(0, domains[image.camera] == 0);
    WriteImage(mask, pixels);
  }
  ExportBundle(scene, selected, output, output / "colmap", true, .05, result, supervision);
  Log(args.command + ": " + report.at("reason").get<std::string>());
  std::cout << Json({{"work", output.u8string()},
                     {"accepted", accepted},
                     {"reason", report.at("reason")}})
                   .dump(2)
            << '\n';
}
}  // namespace geosun

#include "app/pipeline.h"
#include "io/scene.h"
#include "io/las.h"
#include "io/colmap.h"
#include "io/cache.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "sfm/reconstruction.h"
#include "alignment/alignment.h"
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
  Json result = {{"workers", 4},         {"quality_median_px", 2.},
                 {"quality_p90_px", 4.}, {"las_support_fraction", .25},
                 {"las_median_m", .1},   {"las_p90_m", .3}};
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
  for (const auto& [key, value] : result.items())
    if (key != "workers" &&
        (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() <= 0))
      throw std::runtime_error("Expected positive numeric setting: " + key);
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
}  // namespace
void Run(const Arguments& args) {
  args.Validate({"--input", "--work", "--models", "--config", "--stage", "--output", "--las",
                 "--no-ply", "--ply-voxel"});
  const auto work = args.Get("--work").empty()
                        ? fs::absolute(fs::path("work") / args.Path("--input").filename())
                        : args.Path("--work");
  fs::create_directories(work);
  Lock lock(work);
  const auto state_path = work / "state.json";
  Json state = fs::exists(state_path) ? ReadJson(state_path) : Json::object();
  if (!state.empty() && state.at("version") != 1)
    throw std::runtime_error("Unsupported work format; choose a new work directory");
  if (!state.empty() && state.value("pipeline_version", 0) != 2)
    throw std::runtime_error("Reconstruction implementation changed; choose a new work directory");
  const auto input = args.Get("--input").empty() ? fs::u8path(state.at("input").get<std::string>())
                                                 : args.Path("--input");
  if (Contains(input, work) || Contains(work, input))
    throw std::runtime_error("Input and work directories must be separate");
  const auto settings = args.Get("--config").empty()
                            ? (state.empty() ? Settings({}) : state.at("settings"))
                            : Settings(args.Path("--config"));
  const auto las =
      args.Get("--las").empty()
          ? (state.empty() ? fs::path()
                           : fs::u8path(state.at("scene").at("las").get<std::string>()))
          : args.Path("--las");
  const Scene scene = ReadScene(input, las);
  ReadLasHeader(scene.las);
  const auto prepared = scene.ToJson();
  if (state.empty()) {
    state = {{"version", 1},         {"pipeline_version", 2}, {"input", scene.input.u8string()},
             {"settings", settings}, {"scene", prepared},     {"stages", Json::object()}};
    WriteJson(work / "scene.json", prepared);
  } else if (state.at("scene") != prepared || state.at("settings") != settings)
    throw std::runtime_error("Inputs/settings changed; choose a new work directory");
  const auto root =
      args.Get("--models").empty()
          ? (state.contains("models") ? fs::u8path(state.at("models").get<std::string>())
                                      : fs::absolute("models"))
          : args.Path("--models");
  const auto requested = args.Get("--stage", args.command == "prepare" ? "prepare" : "all");
  const std::vector<std::string> stages = {"prepare", "masks", "features", "matches",
                                           "sfm",     "align", "export"};
  if (requested != "all" && std::find(stages.begin(), stages.end(), requested) == stages.end())
    throw std::runtime_error("Unknown stage: " + requested);
  const auto output = args.Get("--output").empty() ? work / "colmap" : args.Path("--output");
  if (Contains(input, output) || Contains(output, input))
    throw std::runtime_error("Output must be separate from input");
  const bool ply = args.Get("--no-ply") != "true";
  const double voxel = args.Number("--ply-voxel", .05);
  state["status"] = "running";
  state["error"] = nullptr;
  WriteJson(state_path, state);
  const auto started = Clock::now();
  const int workers = settings.at("workers");
  try {
    for (const auto& stage : stages) {
      if (requested != "all" && stage != requested)
        continue;
      if (stage != "prepare") {
        const size_t index = std::find(stages.begin(), stages.end(), stage) - stages.begin();
        if (!state.at("stages").contains(stages[index - 1]))
          throw std::runtime_error("Run prerequisite stage: " + stages[index - 1]);
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
          ExportBundle(scene, ReadModel(work / "cache/aligned.bin"), work, output, ply, voxel,
                       ReadJson(work / "result.json"));
        }
        Log("Reused " + stage);
        continue;
      }
      std::vector<fs::path> products;
      if (stage == "prepare") {
        NavigationPositions(scene);
        products = {work / "scene.json"};
      } else if (stage == "masks") {
        MakeMasks(scene, work, root);
        products = {work / "cache/masks"};
      } else if (stage == "features") {
        MakeFeatures(scene, work, root);
        products = {work / "cache/features"};
      } else if (stage == "matches") {
        MakeMatches(scene, work, root);
        products = {work / "cache/matches", work / "cache/pairs.json"};
      } else if (stage == "sfm") {
        Json report;
        Reconstruction model;
        for (int repair = 0;; ++repair) {
          try {
            model = Reconstruct(scene, work, workers, report);
            break;
          } catch (const RegistrationIncomplete& missing) {
            if (repair >= 2)
              throw;
            Log("Repair matching support for " + std::to_string(missing.images.size()) +
                " unregistered images");
            MakeMatches(scene, work, root,
                        std::set<uint32_t>(missing.images.begin(), missing.images.end()));
            state["stages"]["matches"]["files"] =
                Inventory(work, {work / "cache/matches", work / "cache/pairs.json"});
            WriteJson(state_path, state);
          }
        }
        const auto quality =
            model.Quality(scene, settings.at("quality_median_px"), settings.at("quality_p90_px"));
        if (!quality.at("passed").get<bool>())
          throw std::runtime_error("Visual acceptance rejected: " + quality.at("failures").dump());
        WriteModel(work / "cache/visual.bin", model);
        report["quality"] = quality;
        WriteJson(work / "cache/sfm.json", report);
        products = {work / "cache/visual.bin", work / "cache/sfm.json"};
      } else if (stage == "align") {
        auto model = ReadModel(work / "cache/visual.bin");
        const AlignmentLimits limits{settings.at("las_support_fraction"),
                                     settings.at("las_median_m"), settings.at("las_p90_m"),
                                     settings.at("quality_median_px"),
                                     settings.at("quality_p90_px")};
        auto result = Align(scene, model, workers, limits);
        result["sfm"] = ReadJson(work / "cache/sfm.json");
        result["runtime"] = "C++/CUDA/TensorRT";
        result["input"] = scene.input.u8string();
        WriteModel(work / "cache/aligned.bin", model);
        WriteJson(work / "result.json", result);
        products = {work / "cache/aligned.bin", work / "result.json"};
      } else {
        ExportBundle(scene, ReadModel(work / "cache/aligned.bin"), work, output, ply, voxel,
                     ReadJson(work / "result.json"));
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
                     {"seconds", Seconds(started)}})
                   .dump()
            << '\n';
}
void Export(const Arguments& args) {
  args.Validate({"--work", "--output", "--no-ply", "--ply-voxel"});
  const auto work = args.Path("--work");
  Lock lock(work);
  const auto scene = Scene::FromJson(ReadJson(work / "scene.json"));
  const auto model = ReadModel(work / "cache/aligned.bin");
  const auto output = args.Get("--output").empty() ? work / "colmap" : args.Path("--output");
  if (Contains(scene.input, output) || Contains(output, scene.input))
    throw std::runtime_error("Output must be separate from input");
  ExportBundle(scene, model, work, output, args.Get("--no-ply") != "true",
               args.Number("--ply-voxel", .05), ReadJson(work / "result.json"));
  std::cout << CheckBundle(output).dump() << '\n';
}
}  // namespace geosun

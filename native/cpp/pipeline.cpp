#include "pipeline.h"

#include <iostream>
#include <set>
#include <toml++/toml.hpp>

#include "array_io.h"
#include "delivery.h"
#include "registration.h"
#include "scene.h"
#include "sfm_pipeline.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <process.h>
#endif

namespace {
const std::vector<std::string> Stages = {
    "prepare",     "masks", "features", "matches",  "observations",
    "calibration", "sfm",   "finalize", "register", "export"};
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
void Frozen(const fs::path& path, const Json& value) {
  if (fs::exists(path)) {
    if (ReadJson(path) != value)
      throw std::runtime_error("Frozen inputs changed: " + path.u8string());
  } else {
    fs::create_directories(path.parent_path());
    WriteJson(path, value);
  }
}
fs::path Executable() {
#ifdef _WIN32
  std::vector<wchar_t> path(32768);
  const auto size =
      GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
  if (!size || size == path.size())
    throw std::runtime_error("Cannot locate native executable");
  return fs::path(std::wstring(path.data(), size));
#else
  return fs::read_symlink("/proc/self/exe");
#endif
}
class WorkLock {
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#endif
 public:
  explicit WorkLock(const fs::path& work) {
#ifdef _WIN32
    const auto file = work / ".pipeline.lock";
    handle_ = CreateFileW(file.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                          nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE)
      throw std::runtime_error(
          "Another process owns the work directory, or the lock cannot be "
          "opened");
#else
    throw std::runtime_error(
        "This standalone production entry currently targets Windows");
#endif
  }
  ~WorkLock() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#endif
  }
};
void Frontend(const fs::path& work, const fs::path& models,
              const std::string& stage) {
  const auto program =
      Executable().parent_path() / "inference/geosunpre-inference.exe";
  if (!fs::is_regular_file(program))
    throw std::runtime_error("Missing native CUDA frontend: " +
                             program.u8string());
  const auto manifest =
      stage == "masks" ? work / "manifest.json" : work / "sfm/manifest.json";
#ifdef _WIN32
  const std::vector<std::wstring> values = {
      program.native(), L"frontend",
      L"--manifest",    manifest.native(),
      L"--work",        work.native(),
      L"--models",      models.native(),
      L"--stage",       fs::u8path(stage).native()};
  std::vector<const wchar_t*> arguments;
  for (const auto& value : values) arguments.push_back(value.c_str());
  arguments.push_back(nullptr);
  const auto result = _wspawnv(_P_WAIT, program.c_str(), arguments.data());
  if (result < 0) throw std::runtime_error("Cannot start native CUDA frontend");
  if (result)
    throw std::runtime_error("Native CUDA frontend failed with exit code " +
                             std::to_string(result));
#else
  throw std::runtime_error(
      "Native production process launch currently targets Windows");
#endif
}
Json Settings(const fs::path& file) {
  Json result = {{"workers", 4},
                 {"registration_las", nullptr},
                 {"quality_max_median_px", 2.},
                 {"quality_max_p90_px", 4.},
                 {"registration_min_support_fraction", .25},
                 {"registration_max_median_m", .1},
                 {"registration_max_p90_m", .3}};
  if (!file.empty()) {
    const auto table = toml::parse_file(file.u8string());
    for (const auto& [key, node] : table) {
      const auto name = std::string(key.str());
      if (!result.contains(name))
        throw std::runtime_error("Unknown pipeline setting: " + name);
      if (name == "workers") {
        const auto v = node.value<int64_t>();
        if (!v) throw std::runtime_error("workers must be an integer");
        result[name] = *v;
      } else if (name == "registration_las") {
        const auto v = node.value<std::string>();
        if (!v || v->empty())
          throw std::runtime_error("registration_las must be a path");
        result[name] = *v;
      } else {
        const auto v = node.value<double>();
        if (!v)
          throw std::runtime_error("Expected numeric pipeline setting: " +
                                   name);
        result[name] = *v;
      }
    }
  }
  if (result.at("workers").get<int64_t>() < 1 ||
      result.at("workers").get<int64_t>() > 16)
    throw std::runtime_error("workers must be 1-16");
  for (const auto& [name, value] : result.items())
    if (name != "workers" && name != "registration_las" &&
        (!std::isfinite(value.get<double>()) || value.get<double>() <= 0))
      throw std::runtime_error("Pipeline limits must be finite and positive");
  if (result.at("registration_min_support_fraction").get<double>() > 1 ||
      result.at("quality_max_median_px").get<double>() >
          result.at("quality_max_p90_px").get<double>() ||
      result.at("registration_max_median_m").get<double>() >
          result.at("registration_max_p90_m").get<double>())
    throw std::runtime_error("Invalid pipeline limit ordering");
  return result;
}
fs::path LasInput(const fs::path& input, const Json& settings) {
  if (!settings.at("registration_las").is_null()) {
    const auto path =
        fs::u8path(settings.at("registration_las").get<std::string>());
    return fs::weakly_canonical(path.is_absolute() ? path : input / path);
  }
  fs::path found;
  for (const auto& row : fs::directory_iterator(input / "ResultOut")) {
    auto extension = row.path().extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (extension == ".las") {
      if (!found.empty())
        throw std::runtime_error(
            "Expected one ResultOut/*.las; configure registration_las");
      found = row.path();
    }
  }
  if (found.empty()) throw std::runtime_error("Missing ResultOut/*.las");
  return fs::absolute(found);
}
Json Inventory(const fs::path& work, const std::vector<fs::path>& paths) {
  Json result = Json::object();
  auto key = [&](const fs::path& path) {
    const auto relative = fs::relative(path, work);
    return relative.empty() ? fs::absolute(path).generic_u8string()
                            : relative.generic_u8string();
  };
  for (const auto& file : paths) {
    if (fs::is_directory(file)) {
      for (const auto& row : fs::recursive_directory_iterator(file))
        if (row.is_regular_file())
          result[key(row.path())] = FileMetadata(row.path());
    } else
      result[key(file)] = FileMetadata(file);
  }
  return result;
}
void VerifyInventory(const fs::path& work, const Json& files) {
  for (const auto& [name, metadata] : files.items())
    if (FileMetadata(fs::u8path(name).is_absolute()
                         ? fs::u8path(name)
                         : work / fs::u8path(name)) != metadata)
      throw std::runtime_error("Committed pipeline artifact changed: " + name);
}
void ArchivePending(const fs::path& work, const fs::path& output) {
  std::vector<fs::path> pending;
  for (const auto& row : fs::recursive_directory_iterator(work))
    if (row.is_directory() && row.path().extension() == ".pending")
      pending.push_back(row.path());
  auto external = output;
  external += ".pending";
  if (fs::exists(external) && !Contains(work, external))
    pending.push_back(external);
  std::sort(pending.begin(), pending.end());
  const auto stamp =
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());
  for (const auto& source : pending)
    if (fs::exists(source)) {
      if (!Contains(work, source) && source != external)
        throw std::runtime_error("Unexpected pending output location");
      auto target = source;
      target += ".interrupted-" + stamp;
      fs::rename(source, target);
      std::cerr << "Archived interrupted output: " << target.u8string() << '\n';
    }
}
Json ObservationIdentity(const fs::path& work) {
  const auto root = work / "sfm";
  const auto data = ReadJson(root / "frontend.json");
  Json features = Json::object(), matches = Json::object();
  for (const auto& [name, row] : data.at("features").items())
    features[name] =
        FileMetadata(fs::u8path(row.at("path").get<std::string>()));
  for (const auto& row : data.at("pairs"))
    matches[row.at("path").get<std::string>()] =
        FileMetadata(fs::u8path(row.at("path").get<std::string>()));
  return {{"version", 2},
          {"scope", "all_eligible_observations"},
          {"pipeline_config_file", FileMetadata(work / "pipeline_config.json")},
          {"inputs",
           {{"scene_manifest_file", FileMetadata(root / "manifest.json")},
            {"frontend_file", FileMetadata(root / "frontend.json")},
            {"pair_protocol_file", FileMetadata(root / "pair_protocol.json")},
            {"feature_cache_files", features},
            {"match_cache_files", matches}}}};
}
void FreezeObservations(const fs::path& work) {
  const auto identity = ObservationIdentity(work),
             data = ReadJson(work / "sfm/frontend.json");
  const auto graph =
      ReadJson(work / "sfm/pair_protocol.json").at("graph_after_repair");
  if (graph.at("components") != 1 ||
      graph.at("images_below_three_supported_pairs") != 0)
    throw std::runtime_error(
        "Matching support graph remains disconnected or too weak");
  if (fs::exists(work / "sfm/observation_protocol.json")) {
    Frozen(work / "sfm/observation_protocol.json", identity);
    return;
  }
  std::map<std::string, size_t> sizes;
  for (const auto& [name, row] : data.at("features").items()) {
    const auto arrays =
        geosunpre::ReadArrays(fs::u8path(row.at("path").get<std::string>()),
                              {"keypoints", "descriptors",
                               "normalized_keypoints", "scores", "token"});
    const auto count = arrays.at("keypoints").shape.at(0);
    if (geosunpre::ArrayString(arrays.at("token")) !=
            row.at("token").get<std::string>() ||
        arrays.at("keypoints").shape != std::vector<size_t>{count, 2} ||
        arrays.at("normalized_keypoints").shape !=
            std::vector<size_t>{count, 2} ||
        arrays.at("descriptors").shape != std::vector<size_t>{count, 256} ||
        arrays.at("scores").shape != std::vector<size_t>{count})
      throw std::runtime_error("Invalid feature cache: " + name);
    for (const char* key :
         {"keypoints", "descriptors", "normalized_keypoints", "scores"}) {
      const auto& a = arrays.at(key);
      if (a.dtype != "<f4")
        throw std::runtime_error("Expected float32 features");
      const auto* values = reinterpret_cast<const float*>(a.bytes.data());
      for (size_t i = 0; i < a.Count(); ++i)
        if (!std::isfinite(values[i]))
          throw std::runtime_error("Nonfinite feature cache");
    }
    sizes[name] = count;
  }
  for (const auto& row : data.at("pairs")) {
    const auto arrays =
        geosunpre::ReadArrays(fs::u8path(row.at("path").get<std::string>()),
                              {"matches", "scores", "token"});
    const auto &m = arrays.at("matches"), s = arrays.at("scores");
    if (m.dtype != "<u4" || s.dtype != "<f4" || m.shape.size() != 2 ||
        m.shape[1] != 2 || s.shape != std::vector<size_t>{m.shape[0]} ||
        geosunpre::ArrayString(arrays.at("token")) !=
            row.at("token").get<std::string>())
      throw std::runtime_error("Invalid match cache");
    const auto* ids = reinterpret_cast<const uint32_t*>(m.bytes.data());
    const auto* scores = reinterpret_cast<const float*>(s.bytes.data());
    for (size_t i = 0; i < m.shape[0]; ++i)
      if (!std::isfinite(scores[i]) ||
          ids[2 * i] >= sizes.at(row.at("a").get<std::string>()) ||
          ids[2 * i + 1] >= sizes.at(row.at("b").get<std::string>()))
        throw std::runtime_error("Invalid matched feature index");
  }
  if (identity != ObservationIdentity(work))
    throw std::runtime_error("Frontend changed while freezing observations");
  Frozen(work / "sfm/observation_protocol.json", identity);
}
}  // namespace
void RunProduction(const Arguments& args) {
  args.Validate({"--input", "--work", "--models", "--config", "--stage",
                 "--output", "--image-mode", "--ply-voxel-size",
                 "--include-ply", "--no-include-ply", "--include-masks",
                 "--no-include-masks", "--blas-threads"});
  const auto work =
      args.Get("--work").empty()
          ? (args.Get("--input").empty()
                 ? throw std::runtime_error(
                       "Use --input for a new run, or --work to resume")
                 : fs::absolute(fs::path("work") /
                                args.Path("--input").filename()))
          : args.Path("--work");
  if (fs::exists(work / "task.json"))
    throw std::runtime_error(
        "This directory belongs to the reference pipeline; choose a new native "
        "work directory");
  Json saved = fs::exists(work / "native_contract.json")
                   ? ReadJson(work / "native_contract.json")
                   : Json::object();
  const auto input =
      args.Get("--input").empty()
          ? (saved.empty()
                 ? throw std::runtime_error("New runs require --input")
                 : fs::u8path(saved.at("input").get<std::string>()))
          : args.Path("--input");
  if (Contains(input, work) || Contains(work, input))
    throw std::runtime_error("Work and raw input must be separate directories");
  fs::create_directories(work);
  WorkLock lock(work);
  const auto settings =
      args.Get("--config").empty()
          ? (saved.empty() ? Settings({}) : saved.at("settings"))
          : Settings(args.Path("--config"));
  auto models =
      args.Get("--models").empty()
          ? (saved.empty() ? Executable().parent_path() / "models"
                           : fs::u8path(saved.at("models").get<std::string>()))
          : args.Path("--models");
  if (saved.empty() && args.Get("--models").empty() && !fs::exists(models))
    models = Executable().parent_path() / "../../models";
  models = fs::weakly_canonical(models);
  const auto manifest = ReadJson(models / "manifest.json");
  Json model_files = Json::object();
  model_files["manifest.json"] = FileMetadata(models / "manifest.json");
  for (const auto& [name, value] : manifest.at("models").items())
    model_files[name + ".pt"] = FileMetadata(models / fs::u8path(name + ".pt"));
  const auto las = LasInput(input, settings);
  ReadLasHeader(las);
  const auto output =
      args.Get("--output").empty()
          ? (saved.empty()
                 ? work / "colmap"
                 : fs::u8path(
                       saved.at("export").at("output").get<std::string>()))
          : args.Path("--output");
  const auto image_mode = args.Get(
      "--image-mode",
      saved.empty() ? "copy"
                    : saved.at("export").at("image_mode").get<std::string>());
  const bool masks = args.Get("--no-include-masks") == "true" ? false
                     : args.Get("--include-masks") == "true"  ? true
                     : saved.empty()
                         ? true
                         : saved.at("export").at("include_masks").get<bool>();
  const bool ply = args.Get("--no-include-ply") == "true" ? false
                   : args.Get("--include-ply") == "true"  ? true
                   : saved.empty()
                       ? true
                       : saved.at("export").at("include_ply").get<bool>();
  const double voxel = args.Number(
      "--ply-voxel-size",
      saved.empty() ? .05
                    : saved.at("export").at("ply_voxel_size").get<double>());
  const auto scene = PrepareScene(input, work);
  Json contract = {{"version", 1},
                   {"runtime", "C++/CUDA"},
                   {"input", fs::absolute(input).u8string()},
                   {"settings", settings},
                   {"models", models.u8string()},
                   {"model_files", model_files},
                   {"las", las.u8string()},
                   {"las_file", FileMetadata(las)},
                   {"prepared_scene", scene},
                   {"export",
                    {{"output", output.u8string()},
                     {"image_mode", image_mode},
                     {"include_masks", masks},
                     {"include_ply", ply},
                     {"ply_voxel_size", voxel}}}};
  Frozen(work / "native_contract.json", contract);
  Frozen(work / "pipeline_config.json",
         {{"version", 8}, {"settings", settings}});
  ArchivePending(work, output);
  fs::create_directories(work / "native_receipts");
  const auto requested = args.Get("--stage", "all");
  if (requested != "all" &&
      std::find(Stages.begin(), Stages.end(), requested) == Stages.end())
    throw std::runtime_error("Unknown production stage: " + requested);
  Json status = {{"version", 2},
                 {"runtime", "C++/CUDA"},
                 {"status", "running"},
                 {"requested_stage", requested},
                 {"stages", Json::array()}};
  WriteJson(work / "pipeline_status.json", status);
  const auto started = Clock::now();
  try {
    for (const auto& stage : Stages) {
      if (requested != "all" && requested != stage) continue;
      const auto checkpoint =
          work / "native_receipts" / fs::u8path(stage + ".json");
      Json row = {{"name", stage}, {"status", "running"}};
      status["active_stage"] = stage;
      status["stages"].push_back(row);
      WriteJson(work / "pipeline_status.json", status);
      const auto begin = Clock::now();
      std::cerr << "Pipeline stage: " << stage << '\n';
      if (fs::exists(checkpoint)) {
        const auto receipt = ReadJson(checkpoint);
        if (receipt.at("contract_file") !=
            FileMetadata(work / "native_contract.json"))
          throw std::runtime_error("Stage belongs to another native run");
        VerifyInventory(work, receipt.at("files"));
        if (stage == "export") CheckExport(output);
        status["stages"].back()["reused"] = true;
      } else {
        std::vector<fs::path> products;
        if (stage == "prepare")
          products = {work / "manifest.json", work / "pipeline_config.json"};
        else if (stage == "masks" || stage == "features" ||
                 stage == "matches") {
          Frontend(work, models, stage);
          const auto frontend = ReadJson(work / "frontend_runtime.json");
          status["stages"].back()["resources"] = frontend.at("resources");
          status["stages"].back()["environment"] = frontend.at("environment");
          if (stage == "masks")
            products = {work / "sfm/manifest.json", work / "sfm/masks"};
          else if (stage == "features")
            products = {work / "features", work / "sfm/feature_catalog.json",
                        work / "sfm/retrieval.npy"};
          else
            products = {work / "matches", work / "sfm/frontend.json",
                        work / "sfm/pair_protocol.json"};
        } else if (stage == "observations") {
          FreezeObservations(work);
          products = {work / "sfm/observation_protocol.json"};
        } else if (stage == "calibration" || stage == "sfm") {
          Frozen(work / "sfm/observation_protocol.json",
                 ObservationIdentity(work));
          RunVisualPipeline(
              Arguments({"geosunpre-native", "sfm", "--work", work.u8string(),
                         "--stage", stage, "--workers", "4"}));
          if (stage == "calibration")
            products = {work / "calibration/selection.json",
                        work / "calibration/stages",
                        work / "calibration/mapping_inputs/completion.json"};
          else
            products = {work / "visual/stages", work / "visual/quality.json",
                        work / "visual/mapping_inputs/completion.json"};
        } else if (stage == "finalize") {
          const auto quality = ReadJson(work / "visual/quality.json");
          if (quality.at("acceptance").at("status") != "passed")
            throw std::runtime_error("Visual acceptance missing");
          products = {work / "visual/stages/principal/sparse/0",
                      work / "visual/quality.json"};
        } else if (stage == "register") {
          RunRegistration(Arguments(
              {"geosunpre-native",
               "register",
               "--input",
               (work / "visual/stages/principal/sparse/0").u8string(),
               "--manifest",
               (work / "sfm/manifest.json").u8string(),
               "--las",
               las.u8string(),
               "--output",
               (work / "registration/native").u8string(),
               "--workers",
               std::to_string(settings.at("workers").get<int>()),
               "--minimum-support",
               settings.at("registration_min_support_fraction").dump(),
               "--maximum-median",
               settings.at("registration_max_median_m").dump(),
               "--maximum-p90",
               settings.at("registration_max_p90_m").dump(),
               "--maximum-median-px",
               settings.at("quality_max_median_px").dump(),
               "--maximum-p90-px",
               settings.at("quality_max_p90_px").dump(),
               "--joint",
               "true"}));
          const auto report =
              ReadJson(work / "registration/native/report.json");
          WriteJson(
              work / "registration/provenance.json",
              {{"runtime", "C++/CUDA"},
               {"input_root", input.u8string()},
               {"las", las.u8string()},
               {"las_file", FileMetadata(las)},
               {"selected_stage", report.at("selection").at("selected_stage")},
               {"coordinate_frame", "input_las_xyz"},
               {"absolute_accuracy_validated", false}});
          products = {work / "registration/native",
                      work / "registration/provenance.json"};
        } else if (stage == "export") {
          std::vector<std::string> options = {
              "geosunpre-native",
              "export",
              "--work",
              work.u8string(),
              "--output",
              output.u8string(),
              "--las",
              las.u8string(),
              "--image-mode",
              image_mode,
              "--ply-voxel-size",
              std::to_string(voxel),
              "--provenance",
              (work / "registration/provenance.json").u8string()};
          if (!masks) options.push_back("--no-include-masks");
          if (!ply) options.push_back("--no-include-ply");
          RunExport(Arguments(options));
          products = {output / "colmap_export.json"};
        }
        WriteJson(checkpoint, {{"version", 1},
                               {"contract_file",
                                FileMetadata(work / "native_contract.json")},
                               {"files", Inventory(work, products)}});
      }
      status["stages"].back()["status"] = "completed";
      status["stages"].back()["elapsed_seconds"] = Seconds(begin);
      status["active_stage"] = nullptr;
      status["elapsed_seconds"] = Seconds(started);
      WriteJson(work / "pipeline_status.json", status);
      std::cerr << "Pipeline stage " << stage << ": completed ("
                << Seconds(begin) << "s)\n";
    }
    status["status"] = "completed";
    status["elapsed_seconds"] = Seconds(started);
    status["resources"] = Resources();
    WriteJson(work / "pipeline_status.json", status);
    WriteJson(work / "run_result.json", status);
  } catch (const std::exception& error) {
    status["status"] = "failed";
    status["error"] = error.what();
    if (!status["stages"].empty()) status["stages"].back()["status"] = "failed";
    status["elapsed_seconds"] = Seconds(started);
    WriteJson(work / "pipeline_status.json", status);
    throw;
  }
  std::cout << Json{{"runtime", "C++/CUDA"},
                    {"work", work.u8string()},
                    {"output", output.u8string()},
                    {"stage", requested},
                    {"elapsed_seconds", Seconds(started)}}
                   .dump()
            << '\n';
}

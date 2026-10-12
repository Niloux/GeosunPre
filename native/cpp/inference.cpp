#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/version.h>

#include <cwctype>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "array_io.h"
#include "frontend.h"
#include "runtime.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <psapi.h>
#endif

torch::Tensor LoadTensor(const fs::path& path) {
  auto array = geosunpre::ReadArray(path);
  if (array.dtype != "<f4")
    throw std::runtime_error("Expected float32 model input: " +
                             path.u8string());
  std::vector<int64_t> shape(array.shape.begin(), array.shape.end());
  return torch::from_blob(array.bytes.data(), shape, torch::kFloat32)
      .to(torch::kCUDA);
}
void SaveTensor(const fs::path& path, torch::Tensor tensor) {
  tensor = tensor.detach().to(torch::kCPU).contiguous();
  geosunpre::Array array;
  if (tensor.scalar_type() == torch::kInt64)
    array.dtype = "<i8";
  else {
    tensor = tensor.to(torch::kFloat32);
    array.dtype = "<f4";
  }
  array.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
  const auto* bytes = reinterpret_cast<const uint8_t*>(tensor.data_ptr());
  array.bytes.assign(bytes, bytes + tensor.numel() * tensor.element_size());
  geosunpre::WriteArray(path, array);
}
Json InferenceInfo() {
  Json info = {{"runtime", "LibTorch C++/CUDA"},
               {"torch", TORCH_VERSION},
               {"cuda_available", torch::cuda::is_available()},
               {"devices", torch::cuda::device_count()}};
#ifdef _WIN32
  std::vector<HMODULE> modules(1024);
  DWORD required = 0;
  if (!EnumProcessModules(GetCurrentProcess(), modules.data(),
                          DWORD(modules.size() * sizeof(HMODULE)), &required))
    throw std::runtime_error("Cannot inspect inference modules");
  if (required > modules.size() * sizeof(HMODULE))
    throw std::runtime_error("Inference module inventory exceeds buffer");
  bool python = false;
  for (size_t i = 0; i < required / sizeof(HMODULE); ++i) {
    wchar_t name[MAX_PATH];
    if (!GetModuleBaseNameW(GetCurrentProcess(), modules[i], name, MAX_PATH))
      throw std::runtime_error("Cannot inspect inference module name");
    std::wstring value(name);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](wchar_t c) { return std::towlower(c); });
    if (value.rfind(L"python", 0) == 0 || value == L"torch_python.dll")
      python = true;
  }
  info["python_runtime_loaded"] = python;
#endif
  return info;
}
void Infer(const Arguments& args) {
  args.Validate({"--models", "--request", "--output"});
  if (!torch::cuda::is_available())
    throw std::runtime_error("CUDA inference is unavailable");
  const auto models = args.Path("--models"),
             request_path = args.Path("--request"),
             output = args.Path("--output");
  const auto pending =
      fs::path(output.native() + fs::path(".pending").native());
  if (fs::exists(output) || fs::exists(pending))
    throw std::runtime_error("Choose a fresh inference output");
  const auto manifest = ReadJson(models / "manifest.json");
  if (manifest.at("version") != 1 || manifest.at("attention") != "efficient")
    throw std::runtime_error("Unsupported native model manifest");
  fs::create_directories(pending);
  c10::InferenceMode guard;
  at::globalContext().setSDPUseFlash(false);
  at::globalContext().setSDPUseMemEfficient(true);
  at::globalContext().setSDPUseMath(false);
  at::globalContext().setSDPUseCuDNN(false);
  std::map<std::string, torch::jit::Module> loaded;
  Json results = Json::array();
  const auto started = Clock::now();
  const auto request = ReadJson(request_path);
  for (const auto& job : request.at("jobs")) {
    const auto name = job.at("model").get<std::string>();
    const auto label = job.value("name", name);
    if (!manifest.at("models").contains(name))
      throw std::runtime_error("Unknown model: " + name);
    if (!loaded.count(name)) {
      std::ifstream stream(models / fs::u8path(name + ".pt"), std::ios::binary);
      if (!stream) throw std::runtime_error("Missing native model: " + name);
      loaded.emplace(name, torch::jit::load(stream, torch::kCUDA));
      loaded.at(name).eval();
      std::cerr << "Native CUDA: loaded " << name << '\n';
    }
    std::vector<torch::jit::IValue> inputs;
    for (const auto& input : job.at("inputs")) {
      auto path = fs::u8path(input.get<std::string>());
      if (path.is_relative()) path = request_path.parent_path() / path;
      inputs.emplace_back(LoadTensor(path));
    }
    const auto run_started = Clock::now();
    const auto result = loaded.at(name).forward(inputs);
    std::vector<torch::Tensor> outputs;
    if (result.isTensor())
      outputs.push_back(result.toTensor());
    else
      for (const auto& value : result.toTuple()->elements())
        outputs.push_back(value.toTensor());
    Json files = Json::array();
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto file = label + "_" + std::to_string(i) + ".npy";
      SaveTensor(pending / fs::u8path(file), outputs[i]);
      files.push_back(file);
    }
    std::cerr << "Native CUDA " << label << ": completed in "
              << Seconds(run_started) << "s\n";
    results.push_back({{"model", name},
                       {"name", label},
                       {"outputs", files},
                       {"seconds", Seconds(run_started)}});
  }
  WriteJson(pending / "inference.json", {{"environment", InferenceInfo()},
                                         {"jobs", results},
                                         {"seconds", Seconds(started)},
                                         {"resources", Resources()}});
  fs::rename(pending, output);
  std::cout << output.u8string() << '\n';
}
int Main(const std::vector<std::string>& argv) {
  try {
    Arguments args(argv);
    if (args.command == "info") {
      args.Validate({});
      std::cout << InferenceInfo().dump(2) << '\n';
    } else if (args.command == "infer")
      Infer(args);
    else if (args.command == "frontend") {
      RunFrontend(args);
      auto report = ReadJson(args.Path("--work") / "frontend_runtime.json");
      report["environment"] = InferenceInfo();
      WriteJson(args.Path("--work") / "frontend_runtime.json", report);
    } else if (args.command == "help" || args.command == "--help")
      std::cout
          << "geosunpre-inference info\ngeosunpre-inference infer --models "
             "MODELS --request JSON --output NEW_DIR\ngeosunpre-inference "
             "frontend --manifest MANIFEST --work WORK --models MODELS --stage "
             "masks|features|matches\n";
    else
      throw std::invalid_argument("Unknown inference command: " + args.command);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "geosunpre-inference: " << error.what() << '\n';
    return 1;
  }
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) { return Main(Utf8Arguments(argc, argv)); }
#else
int main(int argc, char** argv) { return Main(Utf8Arguments(argc, argv)); }
#endif

#include <windows.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Args = std::vector<std::wstring>;

void Help() {
  std::cout << "Usage: geosun.exe --scene PATH --output PATH\n"
               "  --step all|preprocess|train|export  (default: all)\n"
               "  --texel-size NUMBER                 (default: 0.02)\n"
               "  --preprocess-arg VALUE              (repeat for each argument)\n"
               "  --train-arg VALUE                   (repeat for each argument)\n";
}

fs::path BundleRoot() {
  std::wstring path(32768, L'\0');
  const auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (!size || size >= path.size())
    throw std::runtime_error("Cannot locate geosun.exe");
  path.resize(size);
  return fs::path(path).parent_path();
}

// Encode an argument for the Windows C runtime, including quotes and trailing backslashes.
std::wstring Quote(const std::wstring& value) {
  std::wstring quoted = L"\"";
  size_t slashes = 0;
  for (const auto ch : value) {
    if (ch == L'\\') {
      ++slashes;
      continue;
    }
    quoted.append(ch == L'"' ? 2 * slashes + 1 : slashes, L'\\');
    quoted += ch;
    slashes = 0;
  }
  quoted.append(2 * slashes, L'\\');
  return quoted + L'"';
}

[[noreturn]] void ProcessError(const char* action, const fs::path& program, DWORD error) {
  throw std::system_error(static_cast<int>(error), std::system_category(),
                          std::string(action) + " " + program.u8string());
}

void Run(const fs::path& program, Args args, const Args& extra, const char* stage) {
  args.insert(args.end(), extra.begin(), extra.end());
  auto command = Quote(program.native());
  for (const auto& arg : args)
    command += L" " + Quote(arg);
  std::cout << "> " << stage << std::endl;
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(program.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                      &startup, &process))
    ProcessError("Cannot start", program, GetLastError());
  CloseHandle(process.hThread);
  const auto waited = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD code = 0;
  const bool read = waited == WAIT_OBJECT_0 && GetExitCodeProcess(process.hProcess, &code);
  const auto error = read ? ERROR_SUCCESS : GetLastError();
  CloseHandle(process.hProcess);
  if (!read)
    ProcessError("Cannot get exit code from", program, error);
  if (code)
    throw std::runtime_error(std::string(stage) + " failed (exit code " + std::to_string(code) + ").");
}

void RequireFile(const fs::path& path) {
  if (!fs::is_regular_file(path))
    throw std::runtime_error("Missing file: " + path.u8string());
}

void RequireAbsent(const fs::path& path) {
  if (fs::exists(path))
    throw std::runtime_error("Path already exists: " + path.u8string());
}

void CreatePending(const fs::path& path) {
  if (!fs::create_directory(path))
    throw std::runtime_error("Path already exists: " + path.u8string());
}

void CheckOsgb(const fs::path& path) {
  if (fs::is_directory(path)) {
    for (const auto& entry : fs::recursive_directory_iterator(path))
      if (entry.is_regular_file() && _wcsicmp(entry.path().extension().c_str(), L".osgb") == 0)
        return;
  }
  throw std::runtime_error("No OSGB files in " + path.u8string());
}

void Pipeline(int argc, wchar_t** argv) {
  fs::path scene, out;
  std::wstring step = L"all", texel = L"0.02";
  Args preprocess_args, train_args;
  for (int i = 1; i < argc; ++i) {
    const std::wstring option = argv[i];
    if (++i == argc)
      throw std::runtime_error("Missing value for " + fs::path(option).u8string());
    const std::wstring value = argv[i];
    if (option == L"--scene")
      scene = value;
    else if (option == L"--output")
      out = value;
    else if (option == L"--step")
      step = value;
    else if (option == L"--texel-size")
      texel = value;
    else if (option == L"--preprocess-arg")
      preprocess_args.push_back(value);
    else if (option == L"--train-arg")
      train_args.push_back(value);
    else
      throw std::runtime_error("Unknown option: " + fs::path(option).u8string());
  }
  if (out.empty())
    throw std::runtime_error("Specify --output with the output directory.");
  if (step != L"all" && step != L"preprocess" && step != L"train" && step != L"export")
    throw std::runtime_error("--step must be all, preprocess, train or export.");
  std::wistringstream number(texel);
  number.imbue(std::locale::classic());
  double size = 0;
  if (!(number >> size) || number.peek() != WEOF || !std::isfinite(size) || size <= 0)
    throw std::runtime_error("--texel-size must be a positive number.");

  const auto root = BundleRoot();
  out = fs::absolute(out).lexically_normal();
  const auto geosunpre = root / L"preprocess/geosunpre.exe";
  const auto trainer = root / L"trainer/bin/geosun_3dgs_train.exe";
  const auto config = root / L"trainer/config/trainer.json";
  const auto delivery = out / L"delivery";
  const auto train = out / L"train";
  const auto export_dir = out / L"export";
  const auto export_pending = out / L"export.pending";
  const bool export_results = step == L"all" || step == L"export";
  if (export_results) {
    RequireAbsent(export_dir);
    RequireAbsent(export_pending);
  }

  if (step == L"all" || step == L"preprocess") {
    if (fs::exists(delivery) && !fs::exists(out / L"delivery.work")) {
      std::cout << "Delivery already exists: " << delivery.u8string() << '\n';
    } else {
      if (scene.empty())
        throw std::runtime_error("Specify --scene with the scene directory.");
      Run(geosunpre, {L"run", L"--input", scene.native(), L"--output", delivery.native()},
          preprocess_args, "Preprocess");
    }
    Run(geosunpre, {L"check", L"--output", delivery.native()}, {}, "Delivery check");
  }

  if (step == L"all" || step == L"train") {
    RequireFile(delivery / L"result.json");
    RequireAbsent(train);
    const auto pending = out / L"train.pending";
    for (const auto ch : root.native()) {
      if (ch > 127) {
        std::cerr << "Warning: bundle path is not ASCII; the trainer falls back to CPU JPEG decoding.\n";
        break;
      }
    }
    CreatePending(pending);
    Run(trainer, {L"--config", config.native(), L"-d", delivery.native(), L"-o", pending.native()},
        train_args, "Training");
    fs::rename(pending, train);
    std::cout << "Training: " << train.u8string() << '\n';
  }

  if (export_results) {
    const auto project = train / L"project.licht";
    RequireFile(project);
    CreatePending(export_pending);
    const auto mesh = export_pending / L"mesh.ply";
    Run(trainer, {L"mesh", project.native(), L"-d", delivery.native(), L"-o", mesh.native(),
                  L"--texel-size", texel, L"--export", L"ply", L"--output-name", L"splats"},
        {}, "Export 3DGS PLY, mesh and OSGB");
    RequireFile(export_pending / L"splats.ply");
    RequireFile(mesh);
    CheckOsgb(export_pending / L"mesh_osgb");
    fs::rename(export_pending, export_dir);
    std::cout << "Export: " << export_dir.u8string() << '\n';
  }
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc == 1 || (argc == 2 && std::wstring(argv[1]) == L"--help")) {
    Help();
    return 0;
  }
  try {
    Pipeline(argc, argv);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "geosun: " << error.what() << '\n';
    return 1;
  }
}

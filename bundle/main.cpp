#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Args = std::vector<std::wstring>;

void Help() {
  std::cout << "Usage: Geosun.exe --scene PATH --output PATH\n"
               "  --project FILE.pro                  (default: most recently modified .pro)\n"
               "  --step all|preprocess|train|export  (default: all)\n"
               "  --texel-size NUMBER                 (default: 0.02)\n"
               "  --preprocess-arg VALUE              (repeat for each argument)\n"
               "  --train-arg VALUE                   (repeat for each argument)\n"
               "  Logs are written under --output/logs/.\n";
}

fs::path BundleRoot() {
  std::wstring path(32768, L'\0');
  const auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (!size || size >= path.size())
    throw std::runtime_error("Cannot locate Geosun.exe");
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

std::string Timestamp() {
  SYSTEMTIME st{};
  GetLocalTime(&st);
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "[%04u-%02u-%02u %02u:%02u:%02u.%03u]",
                static_cast<unsigned>(st.wYear), static_cast<unsigned>(st.wMonth),
                static_cast<unsigned>(st.wDay), static_cast<unsigned>(st.wHour),
                static_cast<unsigned>(st.wMinute), static_cast<unsigned>(st.wSecond),
                static_cast<unsigned>(st.wMilliseconds));
  return std::string(buffer);
}

std::string FormatSeconds(double seconds) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(2) << seconds;
  return out.str();
}

struct UniqueHandle {
  HANDLE handle = INVALID_HANDLE_VALUE;
  UniqueHandle() = default;
  explicit UniqueHandle(HANDLE value) : handle(value) {}
  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;
  ~UniqueHandle() { Close(); }
  void Close() noexcept {
    if (handle != INVALID_HANDLE_VALUE && handle != nullptr) {
      CloseHandle(handle);
      handle = INVALID_HANDLE_VALUE;
    }
  }
  bool Valid() const noexcept { return handle != INVALID_HANDLE_VALUE && handle != nullptr; }
};

class LogSink {
 public:
  explicit LogSink(const fs::path& path) {
    if (const auto parent = path.parent_path(); !parent.empty())
      fs::create_directories(parent);
    // Security attributes stay null so child processes do not inherit log handles.
    handle_.handle = CreateFileW(path.c_str(), FILE_APPEND_DATA,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!handle_.Valid())
      ProcessError("Cannot open log", path, GetLastError());
  }

  void WriteRaw(const char* data, std::size_t size) const {
    std::size_t offset = 0;
    while (offset < size) {
      const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(size - offset, 1u << 20));
      DWORD written = 0;
      if (!WriteFile(handle_.handle, data + offset, chunk, &written, nullptr) || written == 0) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "Cannot write log file");
      }
      offset += static_cast<std::size_t>(written);
    }
  }

  void WriteLine(const std::string& message) const {
    const auto line = Timestamp() + " " + message + "\n";
    WriteRaw(line.data(), line.size());
  }

 private:
  UniqueHandle handle_;
};

std::wstring BuildCommand(const fs::path& program, Args args, const Args& extra) {
  args.insert(args.end(), extra.begin(), extra.end());
  auto command = Quote(program.native());
  for (const auto& arg : args)
    command += L" " + Quote(arg);
  return command;
}

void FinishProcess(const fs::path& program, UniqueHandle& process_handle, const char* stage,
                   const LogSink& geosun_log,
                   std::chrono::steady_clock::time_point started) {
  const auto waited = WaitForSingleObject(process_handle.handle, INFINITE);
  DWORD code = 0;
  const bool read = waited == WAIT_OBJECT_0 && GetExitCodeProcess(process_handle.handle, &code);
  const auto error = read ? ERROR_SUCCESS : GetLastError();
  process_handle.Close();
  if (!read)
    ProcessError("Cannot get exit code from", program, error);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  geosun_log.WriteLine("Finished " + std::string(stage) + " in " + FormatSeconds(elapsed) +
                       "s (exit code " + std::to_string(code) + ")");
  if (code)
    throw std::runtime_error(std::string(stage) + " failed (exit code " + std::to_string(code) +
                             ").");
}

void RunTee(const fs::path& program, Args args, const Args& extra, const char* stage,
            const LogSink& geosun_log, const fs::path& stage_log_path) {
  LogSink stage_log(stage_log_path);
  stage_log.WriteLine("=== " + std::string(stage) + " ===");
  auto command = BuildCommand(program, std::move(args), extra);
  geosun_log.WriteLine("Start " + std::string(stage) + ": " + fs::path(command).u8string());
  std::cout << "> " << stage << std::endl;

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE raw_read = INVALID_HANDLE_VALUE;
  HANDLE raw_write = INVALID_HANDLE_VALUE;
  if (!CreatePipe(&raw_read, &raw_write, &sa, 0))
    ProcessError("Cannot create output pipe for", program, GetLastError());
  UniqueHandle read_pipe(raw_read);
  UniqueHandle write_pipe(raw_write);
  if (!SetHandleInformation(read_pipe.handle, HANDLE_FLAG_INHERIT, 0))
    ProcessError("Cannot configure output pipe for", program, GetLastError());

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = nullptr;
  startup.hStdOutput = write_pipe.handle;
  startup.hStdError = write_pipe.handle;
  PROCESS_INFORMATION process{};
  const auto started = std::chrono::steady_clock::now();
  if (!CreateProcessW(program.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                      &startup, &process))
    ProcessError("Cannot start", program, GetLastError());
  UniqueHandle thread_handle(process.hThread);
  UniqueHandle process_handle(process.hProcess);
  thread_handle.Close();
  write_pipe.Close();

  std::string pending_line;
  char buffer[4096];
  DWORD read_bytes = 0;
  while (ReadFile(read_pipe.handle, buffer, sizeof(buffer), &read_bytes, nullptr) &&
         read_bytes > 0) {
    std::cout.write(buffer, static_cast<std::streamsize>(read_bytes));
    std::cout.flush();
    for (DWORD i = 0; i < read_bytes; ++i) {
      const char ch = buffer[i];
      if (ch == '\r')
        continue;
      if (ch == '\n') {
        stage_log.WriteLine(pending_line);
        pending_line.clear();
      } else {
        pending_line.push_back(ch);
      }
    }
  }
  if (!pending_line.empty())
    stage_log.WriteLine(pending_line);
  read_pipe.Close();
  FinishProcess(program, process_handle, stage, geosun_log, started);
}

struct TrainerLogGuard {
  fs::path session_log;
  fs::path target_log;
  const char* stage;
  std::uintmax_t start_offset;
  bool env_set = false;

  ~TrainerLogGuard() {
    if (env_set)
      SetEnvironmentVariableW(L"LFS_HOME", nullptr);
    try {
      std::error_code ec;
      if (!fs::is_regular_file(session_log, ec) || ec)
        return;
      const auto current_size = static_cast<std::uintmax_t>(fs::file_size(session_log, ec));
      if (ec || current_size == 0)
        return;
      const std::uintmax_t read_from = current_size < start_offset ? 0 : start_offset;
      if (current_size <= read_from)
        return;
      std::ifstream in(session_log, std::ios::binary);
      if (!in)
        return;
      in.seekg(static_cast<std::streamoff>(read_from), std::ios::beg);
      std::string slice(static_cast<std::size_t>(current_size - read_from), '\0');
      in.read(slice.data(), static_cast<std::streamsize>(slice.size()));
      const auto bytes_read = static_cast<std::size_t>(in.gcount());
      if (bytes_read == 0)
        return;
      LogSink stage_log(target_log);
      stage_log.WriteLine("=== " + std::string(stage) + " ===");
      stage_log.WriteRaw(slice.data(), bytes_read);
      if (slice[bytes_read - 1] != '\n')
        stage_log.WriteRaw("\n", 1);
    } catch (const std::exception& error) {
      std::cerr << "Warning: cannot archive " << stage << " log: " << error.what() << '\n';
    }
  }
};

void RunTrainer(const fs::path& program, Args args, const Args& extra, const char* stage,
                const LogSink& geosun_log, const fs::path& logs_dir,
                const fs::path& target_log_path) {
  const auto runtime_home = logs_dir / L"runtime";
  const auto session_log = runtime_home / L"logs/geosun-3dgs-trainer.log";
  std::error_code ec;
  const auto initial_size = fs::is_regular_file(session_log, ec)
                                ? static_cast<std::uintmax_t>(fs::file_size(session_log, ec))
                                : std::uintmax_t{0};
  TrainerLogGuard guard{session_log, target_log_path, stage, ec ? std::uintmax_t{0} : initial_size};
  if (!SetEnvironmentVariableW(L"LFS_HOME", runtime_home.c_str()))
    ProcessError("Cannot set LFS_HOME for", program, GetLastError());
  guard.env_set = true;

  auto command = BuildCommand(program, std::move(args), extra);
  geosun_log.WriteLine("Start " + std::string(stage) + ": " + fs::path(command).u8string());
  std::cout << "> " << stage << std::endl;

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  const auto started = std::chrono::steady_clock::now();
  if (!CreateProcessW(program.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                      &startup, &process))
    ProcessError("Cannot start", program, GetLastError());
  UniqueHandle thread_handle(process.hThread);
  UniqueHandle process_handle(process.hProcess);
  thread_handle.Close();
  FinishProcess(program, process_handle, stage, geosun_log, started);
}

void RequireFile(const fs::path& path) {
  if (!fs::is_regular_file(path))
    throw std::runtime_error("Missing file: " + path.u8string());
}

void MoveResult(const fs::path& source, const fs::path& target) {
  std::error_code ec;
  for (int attempt = 0; attempt < 50; ++attempt) {
    fs::rename(source, target, ec);
    if (!ec)
      return;
    const auto code = ec.value();
    if (code != ERROR_ACCESS_DENIED && code != ERROR_SHARING_VIOLATION &&
        code != ERROR_LOCK_VIOLATION)
      break;
    Sleep(100);
  }
  throw fs::filesystem_error("Cannot move result directory", source, target, ec);
}

fs::path PendingResult(fs::path result) {
  result += ".pending";
  return result;
}

void StartPending(const fs::path& path, const LogSink& log) {
  if (fs::exists(path)) {
    auto archived = path;
    archived += ".interrupted-" +
                std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    MoveResult(path, archived);
    log.WriteLine("Kept interrupted results at " + archived.u8string());
  }
  if (!fs::create_directory(path))
    throw std::runtime_error("Path already exists: " + path.u8string());
}

struct ResultDirectory {
  fs::path path;
  const wchar_t* marker;

  void CheckDestination() const {
    if (!fs::exists(path))
      return;
    const auto attributes = GetFileAttributesW(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
      ProcessError("Cannot inspect result directory", path, GetLastError());
    if (!(attributes & FILE_ATTRIBUTE_DIRECTORY) || (attributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
        (!fs::is_regular_file(path / marker) && !fs::is_empty(path)))
      throw std::runtime_error("Existing path is not a Geosun result directory: " + path.u8string());
  }
};

// Each record is created before moving its old result. A missing .pending then means its
// replacement has been installed. Keep the committed marker until all old results are removed.
void RecoverPublish(const fs::path& output) {
  const auto journal = output / L".geosun-publish";
  if (!fs::exists(journal))
    return;
  const auto committed = journal / L"committed";
  const bool published = fs::is_directory(committed);
  for (const auto* name : {L"export", L"train"}) {
    const auto record = journal / name;
    if (!fs::exists(record))
      continue;
    const auto previous = record / L"previous";
    if (published) {
      fs::remove_all(previous);
    } else {
      const auto result = output / name;
      const auto pending = PendingResult(result);
      if (!fs::exists(pending) && fs::exists(result))
        MoveResult(result, pending);
      if (fs::exists(previous))
        MoveResult(previous, result);
    }
    fs::remove(record);
  }
  fs::remove(committed);
  fs::remove(journal);
}

void PublishResults(const fs::path& output, const std::vector<ResultDirectory>& results,
                    const LogSink& log) {
  const auto journal = output / L".geosun-publish";
  if (!fs::create_directory(journal))
    throw std::runtime_error("Cannot start publishing results at " + journal.u8string());
  try {
    for (const auto& result : results) {
      result.CheckDestination();
      const auto record = journal / result.path.filename();
      fs::create_directory(record);
      if (fs::exists(result.path))
        MoveResult(result.path, record / L"previous");
      MoveResult(PendingResult(result.path), result.path);
    }
    fs::create_directory(journal / L"committed");
  } catch (const std::exception& error) {
    try {
      RecoverPublish(output);
    } catch (const std::exception& recovery) {
      throw std::runtime_error(std::string(error.what()) + "\nCannot restore prior results: " +
                               recovery.what());
    }
    throw;
  }
  try {
    RecoverPublish(output);
  } catch (const fs::filesystem_error& error) {
    const std::string warning = "Results published. Previous files could not be cleaned up: " +
                                std::string(error.what());
    std::cerr << warning << '\n';
    log.WriteLine(warning);
  }
  for (const auto& result : results) {
    const auto message = "Published " + result.path.u8string();
    log.WriteLine(message);
    std::cout << message << '\n';
  }
}

void CheckOsgb(const fs::path& path) {
  if (fs::is_directory(path)) {
    for (const auto& entry : fs::recursive_directory_iterator(path))
      if (entry.is_regular_file() && _wcsicmp(entry.path().extension().c_str(), L".osgb") == 0)
        return;
  }
  throw std::runtime_error("No OSGB files in " + path.u8string());
}

fs::path ExistingAncestor(fs::path path) {
  path = fs::absolute(path).lexically_normal();
  while (!path.empty() && !fs::exists(path)) {
    const auto parent = path.parent_path();
    if (parent == path)
      break;
    path = parent;
  }
  return path;
}

bool IsBundleOutputDirectory(const fs::path& name) {
  for (const auto* dir : {L"delivery", L"delivery.work", L"delivery.pending", L"train",
                          L"train.pending", L"export", L"export.pending", L"logs",
                          L".geosun-publish"})
    if (_wcsicmp(name.c_str(), dir) == 0)
      return true;
  for (const std::wstring_view prefix : {L"delivery.pending.interrupted-", L"train.pending.interrupted-",
                                        L"export.pending.interrupted-"})
    if (_wcsnicmp(name.c_str(), prefix.data(), prefix.size()) == 0)
      return true;
  return false;
}

std::uint64_t DirectoryBytes(const fs::path& scene, const fs::path& out) {
  const auto root = fs::weakly_canonical(scene), output = fs::weakly_canonical(out);
  const bool same_root = _wcsicmp(root.c_str(), output.c_str()) == 0;
  std::uint64_t total = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec),
       end; it != end; ++it) {
    if (ec)
      break;
    const auto& entry = *it;
    // A rerun can already have caches and deliveries under the scene root.
    if (same_root ? it.depth() == 0 && IsBundleOutputDirectory(entry.path().filename())
                  : _wcsicmp(entry.path().c_str(), output.c_str()) == 0) {
      it.disable_recursion_pending();
      continue;
    }
    if (entry.is_regular_file(ec)) {
      const auto bytes = entry.file_size(ec);
      if (!ec)
        total += static_cast<std::uint64_t>(bytes);
    }
  }
  return total;
}

std::string FormatGiB(std::uint64_t bytes) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(1)
      << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return out.str();
}

void CheckDiskSpace(const fs::path& out, const fs::path& scene, const std::wstring& step,
                    bool need_preprocess) {
  std::uint64_t required = 0;
  if (need_preprocess)
    required += std::max<std::uint64_t>(8ULL << 30, DirectoryBytes(scene, out) * 3 / 2);
  if (step == L"all" || step == L"train")
    required += 6ULL << 30;
  if (step == L"all" || step == L"export")
    required += 4ULL << 30;
  if (!required)
    return;
  const auto probe = ExistingAncestor(out);
  const auto info = fs::space(probe);
  const auto available = static_cast<std::uint64_t>(info.available);
  if (available < required) {
    throw std::runtime_error("Insufficient free disk space on " + probe.u8string() + ": " +
                             FormatGiB(available) + " GiB available, at least " +
                             FormatGiB(required) + " GiB required for --step " +
                             fs::path(step).u8string() + ".");
  }
}

void Pipeline(int argc, wchar_t** argv) {
  fs::path scene, out;
  std::wstring pcc_project, step = L"all", texel = L"0.02";
  Args preprocess_args, train_args;
  for (int i = 1; i < argc; ++i) {
    const std::wstring option = argv[i];
    if (++i == argc)
      throw std::runtime_error("Missing value for " + fs::path(option).u8string());
    const std::wstring value = argv[i];
    if (option == L"--scene")
      scene = value;
    else if (option == L"--project")
      pcc_project = value;
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
  const auto logs = out / L"logs";
  fs::create_directories(logs);
  // A child still writing after the launcher is closed must keep the output locked.
  SECURITY_ATTRIBUTES lock_attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  UniqueHandle output_lock(CreateFileW((logs / L"geosun.lock").c_str(), GENERIC_READ | GENERIC_WRITE,
                                      0, &lock_attributes, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                                      nullptr));
  if (!output_lock.Valid())
    ProcessError("Cannot lock output directory", out, GetLastError());
  RecoverPublish(out);

  const ResultDirectory train{out / L"train", L"project.licht"};
  const ResultDirectory exported{out / L"export", L"splats.ply"};
  const auto train_pending = PendingResult(train.path);
  const auto export_pending = PendingResult(exported.path);
  const bool train_results = step == L"all" || step == L"train";
  const bool export_results = step == L"all" || step == L"export";
  const bool replace_together = step == L"all" && (fs::exists(train.path) || fs::exists(exported.path));
  if (train_results)
    train.CheckDestination();
  if (export_results)
    exported.CheckDestination();
  const bool run_preprocess = step == L"all" || step == L"preprocess";
  const bool delivery_ready = fs::exists(delivery) && !fs::exists(out / L"delivery.work");
  const bool need_preprocess = run_preprocess && !delivery_ready;
  if (need_preprocess) {
    if (scene.empty())
      throw std::runtime_error("Specify --scene with the scene directory.");
    if (!fs::is_directory(scene))
      throw std::runtime_error("Scene directory does not exist: " + scene.u8string());
  }
  if (step == L"train")
    RequireFile(delivery / L"result.json");
  if (step == L"export")
    RequireFile(train.path / train.marker);

  LogSink geosun_log(logs / L"geosun.log");
  geosun_log.WriteLine("Session start: step=" + fs::path(step).u8string() +
                       " scene=" + (scene.empty() ? std::string("<none>") : scene.u8string()) +
                       " output=" + out.u8string() +
                       " texel_size=" + fs::path(texel).u8string());
  try {
    CheckDiskSpace(out, scene, step, need_preprocess);
    geosun_log.WriteLine("Disk space check passed.");

    const Args project_args = pcc_project.empty() ? Args{} : Args{L"--project", pcc_project};
    if (need_preprocess) {
      preprocess_args.insert(preprocess_args.end(), project_args.begin(), project_args.end());
      RunTee(geosunpre, {L"run", L"--input", scene.native(), L"--output", delivery.native()},
             preprocess_args, "Preprocess", geosun_log, logs / L"preprocess.log");
    } else if (run_preprocess || !project_args.empty()) {
      const std::string line = "Delivery already exists: " + delivery.u8string();
      std::cout << line << '\n';
      geosun_log.WriteLine(line);
      RunTee(geosunpre, {L"check", L"--output", delivery.native()}, project_args, "Delivery check",
             geosun_log, logs / L"preprocess.log");
    }

    if (train_results) {
      RequireFile(delivery / L"result.json");
      for (const auto ch : root.native()) {
        if (ch > 127) {
          const std::string warning =
              "Warning: bundle path is not ASCII; the trainer falls back to CPU JPEG decoding.";
          std::cerr << warning << '\n';
          geosun_log.WriteLine(warning);
          break;
        }
      }
      StartPending(train_pending, geosun_log);
      RunTrainer(trainer, {L"--config", config.native(), L"-d", delivery.native(), L"-o",
                           train_pending.native()},
                 train_args, "Training", geosun_log, logs, logs / L"train.log");
      RequireFile(train_pending / train.marker);
      if (!replace_together)
        PublishResults(out, {train}, geosun_log);
    }

    if (export_results) {
      const auto project = (replace_together ? train_pending : train.path) / train.marker;
      StartPending(export_pending, geosun_log);
      const auto mesh = export_pending / L"mesh.ply";
      RunTrainer(trainer, {L"mesh", project.native(), L"-d", delivery.native(), L"-o", mesh.native(),
                           L"--texel-size", texel, L"--export", L"ply", L"--output-name", L"splats"},
                 {}, "Export 3DGS PLY, mesh and OSGB", geosun_log, logs, logs / L"export.log");
      RequireFile(export_pending / exported.marker);
      RequireFile(mesh);
      CheckOsgb(export_pending / L"mesh_osgb");
      if (!replace_together)
        PublishResults(out, {exported}, geosun_log);
    }
    if (replace_together)
      PublishResults(out, {train, exported}, geosun_log);
    geosun_log.WriteLine("Session completed successfully.");
  } catch (const std::exception& error) {
    geosun_log.WriteLine(std::string("ERROR: ") + error.what());
    throw;
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

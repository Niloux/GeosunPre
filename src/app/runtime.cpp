#include "app/runtime.h"
#include "app/build_info.h"
#include <iostream>
#include <mutex>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <sys/stat.h>
#endif

namespace geosun {
fs::path ExecutablePath() {
#ifdef _WIN32
  std::wstring path(32768, L'\0');
  const auto size = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
  if (!size || size >= path.size())
    throw std::runtime_error("Cannot locate executable");
  path.resize(size);
  return fs::path(path);
#else
  return fs::read_symlink("/proc/self/exe");
#endif
}
fs::path DefaultModels() {
#ifdef _WIN32
  std::wstring configured(32768, L'\0');
  const auto length =
      GetEnvironmentVariableW(L"GEOSUNPRE_MODELS", configured.data(), DWORD(configured.size()));
  if (length >= configured.size())
    throw std::runtime_error("GEOSUNPRE_MODELS path is too long");
  if (length) {
    configured.resize(length);
    return fs::absolute(fs::path(configured));
  }
#else
  const auto* configured = std::getenv("GEOSUNPRE_MODELS");
  if (configured && *configured)
    return fs::absolute(fs::u8path(configured));
#endif
  return ExecutablePath().parent_path() / "models";
}
Json BuildInfo() {
  return {{"version", std::string(BuildVersion)},
          {"revision", std::string(BuildRevision)},
          {"built_utc", std::string(BuildTime)},
          {"pipeline_version", PipelineVersion},
          {"executable", ExecutablePath().u8string()}};
}
double Seconds(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}
Arguments::Arguments(const std::vector<std::string>& args) {
  if (args.size() < 2) {
    command = "help";
    return;
  }
  command = args[1];
  if (std::find(args.begin() + 2, args.end(), "--help") != args.end()) {
    command = "help";
    return;
  }
  for (size_t i = 2; i < args.size(); ++i) {
    const auto& name = args[i];
    if (name == "--no-ply" || name == "--las-supervision" || name == "--no-las-supervision" ||
        name == "--refine" || name == "--debug" || name == "--crop-valid-domain" ||
        name == "--no-crop-valid-domain") {
      if (!values.emplace(name, "true").second)
        throw std::invalid_argument("Repeated option: " + name);
    } else {
      if (name.rfind("--", 0) != 0 || i + 1 == args.size())
        throw std::invalid_argument("Expected --option value: " + name);
      if (!values.emplace(name, args[++i]).second)
        throw std::invalid_argument("Repeated option: " + name);
    }
  }
}
std::string Arguments::Get(const std::string& name, const std::string& fallback) const {
  const auto it = values.find(name);
  return it == values.end() ? fallback : it->second;
}
fs::path Arguments::Path(const std::string& name) const {
  const auto value = Get(name);
  if (value.empty())
    throw std::invalid_argument("Missing " + name);
  return fs::absolute(fs::u8path(value));
}
int Arguments::Integer(const std::string& name, int fallback) const {
  const auto value = Get(name);
  if (value.empty())
    return fallback;
  size_t used;
  const int number = std::stoi(value, &used);
  if (used != value.size() || number < 1)
    throw std::invalid_argument("Expected positive integer: " + name);
  return number;
}
double Arguments::Number(const std::string& name, double fallback) const {
  const auto value = Get(name);
  if (value.empty())
    return fallback;
  size_t used;
  const double number = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(number) || number <= 0)
    throw std::invalid_argument("Expected positive finite number: " + name);
  return number;
}
void Arguments::Validate(std::initializer_list<const char*> options) const {
  for (const auto& [name, value] : values)
    if (std::find(options.begin(), options.end(), name) == options.end())
      throw std::invalid_argument("Unknown option: " + name);
}
Json ReadJson(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Cannot read " + path.u8string());
  return Json::parse(stream);
}
void WriteJson(const fs::path& path, const Json& value) {
  auto temporary = path;
  temporary += ".writing";
  std::ofstream stream(temporary);
  if (!stream)
    throw std::runtime_error("Cannot write " + path.u8string());
  stream << value.dump(2) << '\n';
  stream.close();
  if (!stream)
    throw std::runtime_error("Incomplete write: " + path.u8string());
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
    throw std::runtime_error("Cannot publish JSON: " + path.u8string());
#else
  fs::rename(temporary, path);
#endif
}
Json FileMetadata(const fs::path& path) {
#ifdef _WIN32
  WIN32_FILE_ATTRIBUTE_DATA data;
  if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data))
    throw std::runtime_error("Cannot stat " + path.u8string());
  ULARGE_INTEGER ticks;
  ticks.LowPart = data.ftLastWriteTime.dwLowDateTime;
  ticks.HighPart = data.ftLastWriteTime.dwHighDateTime;
  return {{"size_bytes", (uint64_t(data.nFileSizeHigh) << 32) | data.nFileSizeLow},
          {"mtime_ns", (ticks.QuadPart - 116444736000000000ULL) * 100}};
#else
  struct stat data{};
  if (stat(path.c_str(), &data))
    throw std::runtime_error("Cannot stat " + path.u8string());
  return {{"size_bytes", data.st_size},
          {"mtime_ns", int64_t(data.st_mtim.tv_sec) * 1000000000 + data.st_mtim.tv_nsec}};
#endif
}
Json Resources() {
#ifdef _WIN32
  FILETIME created, exited, kernel, user;
  PROCESS_MEMORY_COUNTERS memory{};
  if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) ||
      !GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
    throw std::runtime_error("Cannot read native process resources");
  auto seconds = [](const FILETIME& value) {
    ULARGE_INTEGER ticks;
    ticks.LowPart = value.dwLowDateTime;
    ticks.HighPart = value.dwHighDateTime;
    return ticks.QuadPart / 1e7;
  };
  return {{"cpu_seconds", seconds(kernel) + seconds(user)},
          {"rss_peak_bytes", memory.PeakWorkingSetSize}};
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("Cannot read native process resources");
  return {{"cpu_seconds", usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
                              (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6},
          {"rss_peak_bytes", usage.ru_maxrss * 1024}};
#endif
}
#ifdef _WIN32
std::vector<std::string> Utf8Arguments(int argc, wchar_t** argv) {
  std::vector<std::string> utf8;
  for (int i = 0; i < argc; ++i) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string value(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, value.data(), size, nullptr, nullptr);
    value.pop_back();
    utf8.push_back(std::move(value));
  }
  return utf8;
}
#else
std::vector<std::string> Utf8Arguments(int argc, char** argv) { return {argv, argv + argc}; }
#endif

}  // namespace geosun

namespace geosun {
void PublishFile(const fs::path& temporary, const fs::path& path) {
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
    throw std::runtime_error("Cannot publish: " + path.u8string());
#else
  fs::rename(temporary, path);
#endif
}
void Log(const std::string& message) {
  // Concurrent stages log whole lines.
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  std::cerr << message << std::endl;
}
}  // namespace geosun

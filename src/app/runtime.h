#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <initializer_list>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace geosun {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int PipelineVersion = 22;
fs::path ExecutablePath();
fs::path DefaultModels();
Json BuildInfo();
double Seconds(Clock::time_point start);
Json ReadJson(const fs::path& path);
void WriteJson(const fs::path& path, const Json& value);
Json FileMetadata(const fs::path& path);
Json Resources();
void PublishFile(const fs::path& temporary, const fs::path& path);
// Keeps a .pending directory left by an interrupted run under a timestamped name.
void ArchiveInterrupted(const fs::path& pending);
void Log(const std::string& message);
// Calls function(i) for every i < count on workers threads, handing out chunk indices at a time.
// A worker exception is rethrown once every worker has finished.
template <class Function>
void ParallelFor(size_t count, int workers, size_t chunk, Function function) {
  std::atomic<size_t> next{0};
  std::vector<std::future<void>> jobs;
  for (int i = 0; i < workers; ++i)
    jobs.push_back(std::async(std::launch::async, [&] {
      for (;;) {
        const size_t start = next.fetch_add(chunk);
        if (start >= count)
          break;
        for (size_t j = start; j < std::min(count, start + chunk); ++j)
          function(j);
      }
    }));
  for (auto& job : jobs)
    job.get();
}
struct Arguments {
  std::string command;
  std::map<std::string, std::string> values;
  explicit Arguments(const std::vector<std::string>& args);
  std::string Get(const std::string& name, const std::string& fallback = "") const;
  fs::path Path(const std::string& name) const;
  int Integer(const std::string& name, int fallback) const;
  double Number(const std::string& name, double fallback) const;
  void Validate(const std::vector<std::string>& options) const;
};
#ifdef _WIN32
std::vector<std::string> Utf8Arguments(int argc, wchar_t** argv);
#else
std::vector<std::string> Utf8Arguments(int argc, char** argv);
#endif
}  // namespace geosun

#pragma once
#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace geosun {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int PipelineVersion = 11;
fs::path ExecutablePath();
fs::path DefaultModels();
Json BuildInfo();
double Seconds(Clock::time_point start);
Json ReadJson(const fs::path& path);
void WriteJson(const fs::path& path, const Json& value);
Json FileMetadata(const fs::path& path);
Json Resources();
void PublishFile(const fs::path& temporary, const fs::path& path);
void Log(const std::string& message);
struct Arguments {
  std::string command;
  std::map<std::string, std::string> values;
  explicit Arguments(const std::vector<std::string>& args);
  std::string Get(const std::string& name, const std::string& fallback = "") const;
  fs::path Path(const std::string& name) const;
  int Integer(const std::string& name, int fallback) const;
  double Number(const std::string& name, double fallback) const;
  void Validate(std::initializer_list<const char*> options) const;
};
#ifdef _WIN32
std::vector<std::string> Utf8Arguments(int argc, wchar_t** argv);
#else
std::vector<std::string> Utf8Arguments(int argc, char** argv);
#endif
}  // namespace geosun

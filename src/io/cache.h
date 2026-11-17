#pragma once
#include "app/runtime.h"
#include <array>
#include <fstream>
#include <type_traits>

namespace geosun {
class BinaryWriter {
  fs::path path_, temporary_;
  std::ofstream stream_;
  // Values are a few bytes each; collecting them keeps stream calls per write to one per MiB.
  std::vector<char> buffer_;
  void Flush();

 public:
  explicit BinaryWriter(const fs::path& path);
  void Bytes(const void* data, size_t count);
  template <class T>
  void Value(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    Bytes(&v, sizeof(T));
  }
  void String(const std::string& v);
  void Commit();
};
class BinaryReader {
  std::ifstream stream_;
  uint64_t remaining_;

 public:
  explicit BinaryReader(const fs::path& path);
  void Bytes(void* data, size_t count);
  template <class T>
  T Value() {
    T v;
    static_assert(std::is_trivially_copyable_v<T>);
    Bytes(&v, sizeof(T));
    return v;
  }
  std::string String();
  void End();
};
struct FeatureSet {
  uint32_t image = 0, dimension = 256;
  std::vector<std::array<float, 2>> pixels, normalized;
  std::vector<float> scores, descriptors;
  std::vector<std::array<uint8_t, 3>> colors;
  std::array<float, 256> retrieval{};
};
struct Match {
  uint32_t a, b;
  float score;
};
static_assert(sizeof(Match) == 12);
struct MatchSet {
  uint32_t a, b;
  std::vector<Match> matches;
};
using MatchArchiveIndex = std::map<std::pair<uint32_t, uint32_t>, uint64_t>;
class MatchArchiveWriter {
  fs::path path_;
  std::fstream stream_;
  std::vector<char> buffer_;

 public:
  explicit MatchArchiveWriter(const fs::path& path);
  ~MatchArchiveWriter();
  MatchArchiveWriter(const MatchArchiveWriter&) = delete;
  MatchArchiveWriter& operator=(const MatchArchiveWriter&) = delete;
  void Append(const MatchSet& matches);
  void Flush();
};
void WriteFeatures(const fs::path& path, const FeatureSet& features);
FeatureSet ReadFeatures(const fs::path& path);
MatchArchiveIndex IndexMatchArchive(const fs::path& path, bool truncate_incomplete);
MatchSet ReadMatchRecord(std::ifstream& stream, uint64_t offset, uint32_t expected_a,
                         uint32_t expected_b);
fs::path FeaturePath(const fs::path& work, uint32_t image);
fs::path MatchesPath(const fs::path& work);
fs::path SelectedModelPath(const fs::path& work, const Json& result);
}  // namespace geosun

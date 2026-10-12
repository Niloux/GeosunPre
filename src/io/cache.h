#pragma once
#include "app/runtime.h"
#include <array>
#include <fstream>
#include <type_traits>

namespace geosun {
class BinaryWriter {
  fs::path path_, temporary_;
  std::ofstream stream_;

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
  std::array<float, 256> retrieval{};
};
struct Match {
  uint32_t a, b;
  float score;
};
struct MatchSet {
  uint32_t a, b;
  std::vector<Match> matches;
};
void WriteFeatures(const fs::path& path, const FeatureSet& features);
FeatureSet ReadFeatures(const fs::path& path);
void WriteMatches(const fs::path& path, const MatchSet& matches);
MatchSet ReadMatches(const fs::path& path);
fs::path FeaturePath(const fs::path& work, uint32_t image);
fs::path MatchPath(const fs::path& work, uint32_t a, uint32_t b);
}  // namespace geosun

#include "io/cache.h"
#include <cmath>
#include <cstring>

namespace geosun {
BinaryWriter::BinaryWriter(const fs::path& path) : path_(path), temporary_(path) {
  fs::create_directories(path.parent_path());
  temporary_ += ".writing";
  stream_.open(temporary_, std::ios::binary);
  if (!stream_)
    throw std::runtime_error("Cannot write: " + path.u8string());
}
void BinaryWriter::Bytes(const void* data, size_t count) {
  if (count)
    stream_.write(static_cast<const char*>(data), count);
  if (!stream_)
    throw std::runtime_error("Incomplete binary write: " + path_.u8string());
}
void BinaryWriter::String(const std::string& v) {
  Value(uint64_t(v.size()));
  Bytes(v.data(), v.size());
}
void BinaryWriter::Commit() {
  stream_.close();
  if (!stream_)
    throw std::runtime_error("Cannot close binary output");
  PublishFile(temporary_, path_);
}
BinaryReader::BinaryReader(const fs::path& path)
    : stream_(path, std::ios::binary), remaining_(fs::file_size(path)) {
  if (!stream_)
    throw std::runtime_error("Cannot read: " + path.u8string());
}
void BinaryReader::Bytes(void* data, size_t count) {
  if (count > remaining_)
    throw std::runtime_error("Truncated binary data");
  if (count && !stream_.read(static_cast<char*>(data), count))
    throw std::runtime_error("Cannot read binary data");
  remaining_ -= count;
}
std::string BinaryReader::String() {
  const auto n = Value<uint64_t>();
  if (n > remaining_)
    throw std::runtime_error("Invalid string length");
  std::string v(size_t(n), '\0');
  Bytes(v.data(), v.size());
  return v;
}
void BinaryReader::End() {
  if (remaining_)
    throw std::runtime_error("Unexpected binary payload");
}
fs::path FeaturePath(const fs::path& work, uint32_t id) {
  return work / "cache/features" / (std::to_string(id) + ".bin");
}
fs::path MatchPath(const fs::path& work, uint32_t a, uint32_t b) {
  return work / "cache/matches" / (std::to_string(a) + "_" + std::to_string(b) + ".bin");
}
void WriteFeatures(const fs::path& path, const FeatureSet& f) {
  BinaryWriter out(path);
  out.String("GEOSUN_FEATURES_2");
  out.Value(f.image);
  out.Value(f.dimension);
  out.Value(uint64_t(f.pixels.size()));
  for (size_t i = 0; i < f.pixels.size(); ++i) {
    out.Value(f.pixels[i]);
    out.Value(f.normalized[i]);
    out.Value(f.scores[i]);
  }
  out.Bytes(f.descriptors.data(), f.descriptors.size() * sizeof(float));
  out.Value(f.retrieval);
  out.Bytes(f.colors.data(), f.colors.size() * sizeof(f.colors[0]));
  out.Commit();
}
FeatureSet ReadFeatures(const fs::path& path) {
  BinaryReader in(path);
  if (in.String() != "GEOSUN_FEATURES_2")
    throw std::runtime_error("Unsupported feature cache");
  FeatureSet f;
  f.image = in.Value<uint32_t>();
  f.dimension = in.Value<uint32_t>();
  const auto n = in.Value<uint64_t>();
  if (f.dimension != 256 || n < 32 || n > 3072)
    throw std::runtime_error("Invalid feature dimensions");
  f.pixels.resize(n);
  f.normalized.resize(n);
  f.scores.resize(n);
  f.descriptors.resize(n * f.dimension);
  for (size_t i = 0; i < n; ++i) {
    f.pixels[i] = in.Value<std::array<float, 2>>();
    f.normalized[i] = in.Value<std::array<float, 2>>();
    f.scores[i] = in.Value<float>();
  }
  in.Bytes(f.descriptors.data(), f.descriptors.size() * sizeof(float));
  f.retrieval = in.Value<std::array<float, 256>>();
  f.colors.resize(n);
  in.Bytes(f.colors.data(), f.colors.size() * sizeof(f.colors[0]));
  in.End();
  for (float x : f.descriptors)
    if (!std::isfinite(x))
      throw std::runtime_error("Nonfinite cached descriptor");
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(f.pixels[i][0]) || !std::isfinite(f.pixels[i][1]) ||
        !std::isfinite(f.normalized[i][0]) || !std::isfinite(f.normalized[i][1]) ||
        !std::isfinite(f.scores[i]))
      throw std::runtime_error("Nonfinite cached feature");
  return f;
}
void WriteMatches(const fs::path& path, const MatchSet& m) {
  BinaryWriter out(path);
  out.String("GEOSUN_MATCHES_1");
  out.Value(m.a);
  out.Value(m.b);
  out.Value(uint64_t(m.matches.size()));
  for (const auto& p : m.matches) {
    out.Value(p.a);
    out.Value(p.b);
    out.Value(p.score);
  }
  out.Commit();
}
MatchSet ReadMatches(const fs::path& path) {
  BinaryReader in(path);
  if (in.String() != "GEOSUN_MATCHES_1")
    throw std::runtime_error("Unsupported match cache");
  MatchSet m;
  m.a = in.Value<uint32_t>();
  m.b = in.Value<uint32_t>();
  const auto n = in.Value<uint64_t>();
  if (n > 3072 || m.a >= m.b)
    throw std::runtime_error("Invalid match cache");
  m.matches.resize(n);
  for (auto& p : m.matches) {
    p.a = in.Value<uint32_t>();
    p.b = in.Value<uint32_t>();
    p.score = in.Value<float>();
    if (p.a >= 3072 || p.b >= 3072 || !std::isfinite(p.score))
      throw std::runtime_error("Invalid cached match");
  }
  in.End();
  return m;
}
}  // namespace geosun

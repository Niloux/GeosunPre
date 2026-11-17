#include "io/cache.h"
#include <cmath>
#include <cstring>

namespace geosun {
fs::path SelectedModelPath(const fs::path& work, const Json& result) {
  const auto selected = result.value("selected_model", std::string("aligned"));
  if (selected != "aligned" && selected != "shutter" && selected != "cleaned")
    throw std::runtime_error("Unknown selected reconstruction cache: " + selected);
  return work / "cache" / (selected + ".bin");
}
BinaryWriter::BinaryWriter(const fs::path& path) : path_(path), temporary_(path) {
  fs::create_directories(path.parent_path());
  temporary_ += ".writing";
  stream_.open(temporary_, std::ios::binary);
  if (!stream_)
    throw std::runtime_error("Cannot write: " + path.u8string());
}
void BinaryWriter::Flush() {
  if (!buffer_.empty())
    stream_.write(buffer_.data(), std::streamsize(buffer_.size()));
  buffer_.clear();
  if (!stream_)
    throw std::runtime_error("Incomplete binary write: " + path_.u8string());
}
void BinaryWriter::Bytes(const void* data, size_t count) {
  constexpr size_t capacity = 1 << 20;
  const auto* bytes = static_cast<const char*>(data);
  if (count >= capacity) {
    Flush();
    stream_.write(bytes, std::streamsize(count));
    if (!stream_)
      throw std::runtime_error("Incomplete binary write: " + path_.u8string());
    return;
  }
  if (buffer_.size() + count > capacity)
    Flush();
  buffer_.insert(buffer_.end(), bytes, bytes + count);
}
void BinaryWriter::String(const std::string& v) {
  Value(uint64_t(v.size()));
  Bytes(v.data(), v.size());
}
void BinaryWriter::Commit() {
  Flush();
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
fs::path MatchesPath(const fs::path& work) {
  return work / "cache/matches.bin";
}
void WriteFeatures(const fs::path& path, const FeatureSet& f) {
  BinaryWriter out(path);
  out.String("GEOSUN_FEATURES");
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
  if (in.String() != "GEOSUN_FEATURES")
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
namespace {
constexpr char MatchArchiveMagic[8] = {'G', 'S', 'M', 'A', 'T', 'C', 'H', '1'};
constexpr uint32_t MatchArchiveEndTag = 0x4D454E44u;
constexpr size_t MatchArchiveBufferBytes = 1 << 20;
void ValidateMatch(const Match& p) {
  if (p.a >= 3072 || p.b >= 3072 || !std::isfinite(p.score))
    throw std::runtime_error("Invalid cached match");
}
}  // namespace
MatchArchiveWriter::MatchArchiveWriter(const fs::path& path) : path_(path) {
  fs::create_directories(path.parent_path());
  buffer_.reserve(MatchArchiveBufferBytes);
  if (fs::exists(path)) {
    const uint64_t size = fs::file_size(path);
    if (size < sizeof(MatchArchiveMagic))
      throw std::runtime_error("Truncated match archive header: " + path.u8string());
    stream_.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!stream_)
      throw std::runtime_error("Cannot open match archive for append: " + path.u8string());
    char magic[8]{};
    if (!stream_.read(magic, sizeof(magic)) ||
        std::memcmp(magic, MatchArchiveMagic, sizeof(magic)) != 0)
      throw std::runtime_error("Unsupported match archive: " + path.u8string());
    stream_.seekp(0, std::ios::end);
  } else {
    stream_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream_)
      throw std::runtime_error("Cannot create match archive: " + path.u8string());
    stream_.write(MatchArchiveMagic, sizeof(MatchArchiveMagic));
    stream_.flush();
    if (!stream_)
      throw std::runtime_error("Cannot write match archive header: " + path.u8string());
  }
}
MatchArchiveWriter::~MatchArchiveWriter() = default;
void MatchArchiveWriter::Append(const MatchSet& m) {
  if (m.a >= m.b || m.matches.size() > 3072)
    throw std::runtime_error("Invalid match record");
  for (const auto& p : m.matches)
    ValidateMatch(p);
  const uint32_t count = uint32_t(m.matches.size());
  const size_t record_bytes = 12 + size_t(count) * sizeof(Match) + 4;
  if (buffer_.size() + record_bytes > MatchArchiveBufferBytes)
    Flush();
  const size_t offset = buffer_.size();
  buffer_.resize(offset + record_bytes);
  char* dst = buffer_.data() + offset;
  std::memcpy(dst, &m.a, 4);
  std::memcpy(dst + 4, &m.b, 4);
  std::memcpy(dst + 8, &count, 4);
  if (count)
    std::memcpy(dst + 12, m.matches.data(), size_t(count) * sizeof(Match));
  std::memcpy(dst + 12 + size_t(count) * sizeof(Match), &MatchArchiveEndTag, 4);
}
void MatchArchiveWriter::Flush() {
  if (!buffer_.empty()) {
    stream_.write(buffer_.data(), std::streamsize(buffer_.size()));
    buffer_.clear();
  }
  stream_.flush();
  if (!stream_)
    throw std::runtime_error("Incomplete match archive write: " + path_.u8string());
}
MatchArchiveIndex IndexMatchArchive(const fs::path& path, bool truncate_incomplete) {
  MatchArchiveIndex index;
  if (!fs::exists(path))
    return index;
  const uint64_t file_size = fs::file_size(path);
  if (file_size < sizeof(MatchArchiveMagic)) {
    if (!truncate_incomplete)
      throw std::runtime_error("Truncated match archive: " + path.u8string());
    if (file_size > 0) {
      std::ifstream in(path, std::ios::binary);
      char magic[8]{};
      in.read(magic, std::streamsize(file_size));
      if (!in || std::memcmp(magic, MatchArchiveMagic, size_t(file_size)) != 0)
        throw std::runtime_error("Unsupported match archive: " + path.u8string());
    }
    fs::remove(path);
    return index;
  }
  std::vector<char> io_buffer(MatchArchiveBufferBytes);
  std::ifstream in;
  in.rdbuf()->pubsetbuf(io_buffer.data(), std::streamsize(io_buffer.size()));
  in.open(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot read match archive: " + path.u8string());
  char magic[8]{};
  if (!in.read(magic, sizeof(magic)) ||
      std::memcmp(magic, MatchArchiveMagic, sizeof(magic)) != 0)
    throw std::runtime_error("Unsupported match archive: " + path.u8string());
  uint64_t offset = sizeof(MatchArchiveMagic), last_valid = offset;
  bool truncated = false;
  std::vector<Match> matches(3072);
  while (offset < file_size) {
    if (file_size - offset < 12) {
      truncated = true;
      break;
    }
    uint32_t header[3]{};
    if (!in.read(reinterpret_cast<char*>(header), sizeof(header)))
      throw std::runtime_error("Cannot read match archive record");
    const uint32_t a = header[0], b = header[1], count = header[2];
    if (a >= b || count > 3072)
      throw std::runtime_error("Invalid match archive record");
    const uint64_t record_bytes = 12 + uint64_t(count) * sizeof(Match) + 4;
    if (file_size - offset < record_bytes) {
      truncated = true;
      break;
    }
    if (count && !in.read(reinterpret_cast<char*>(matches.data()),
                          std::streamsize(size_t(count) * sizeof(Match))))
      throw std::runtime_error("Cannot read match archive payload");
    for (uint32_t i = 0; i < count; ++i)
      ValidateMatch(matches[i]);
    uint32_t end_tag = 0;
    if (!in.read(reinterpret_cast<char*>(&end_tag), sizeof(end_tag)))
      throw std::runtime_error("Cannot read match archive end tag");
    if (end_tag != MatchArchiveEndTag)
      throw std::runtime_error("Corrupted match archive end tag");
    index.emplace(std::make_pair(a, b), offset);
    offset += record_bytes;
    last_valid = offset;
  }
  if (truncated) {
    if (!truncate_incomplete)
      throw std::runtime_error("Truncated match archive: " + path.u8string());
    in.close();
    fs::resize_file(path, last_valid);
  }
  return index;
}
MatchSet ReadMatchRecord(std::ifstream& stream, uint64_t offset, uint32_t expected_a,
                         uint32_t expected_b) {
  stream.seekg(std::streamoff(offset), std::ios::beg);
  uint32_t header[3]{};
  if (!stream.read(reinterpret_cast<char*>(header), sizeof(header)))
    throw std::runtime_error("Cannot read match archive record");
  const uint32_t a = header[0], b = header[1], count = header[2];
  if (a != expected_a || b != expected_b || a >= b || count > 3072)
    throw std::runtime_error("Invalid match cache");
  MatchSet m{a, b, {}};
  m.matches.resize(count);
  if (count && !stream.read(reinterpret_cast<char*>(m.matches.data()),
                            std::streamsize(size_t(count) * sizeof(Match))))
    throw std::runtime_error("Truncated match cache payload");
  uint32_t end_tag = 0;
  if (!stream.read(reinterpret_cast<char*>(&end_tag), sizeof(end_tag)) ||
      end_tag != MatchArchiveEndTag)
    throw std::runtime_error("Corrupted match cache record");
  for (const auto& p : m.matches)
    ValidateMatch(p);
  return m;
}
}  // namespace geosun

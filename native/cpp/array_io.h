#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace geosunpre {
struct Array {
  std::string dtype;
  std::vector<size_t> shape;
  std::vector<uint8_t> bytes;
  size_t Count() const;
  size_t ItemSize() const;
};
Array ReadArray(const std::filesystem::path& path);
void WriteArray(const std::filesystem::path& path, const Array& array);
std::map<std::string, Array> ReadArrays(
    const std::filesystem::path& path,
    const std::vector<std::string>& fields = {});
void WriteArrays(const std::filesystem::path& path,
                 const std::map<std::string, Array>& arrays,
                 bool compress = false);
std::string ArrayString(const Array& array);
Array StringArray(const std::string& value);
}  // namespace geosunpre

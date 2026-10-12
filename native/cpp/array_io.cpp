#include "array_io.h"

#include <zlib.h>

#include <algorithm>
#include <codecvt>
#include <cstring>
#include <fstream>
#include <locale>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace geosunpre {
namespace {
template <typename T>
T Get(const uint8_t* data) {
  T v;
  std::memcpy(&v, data, sizeof(v));
  return v;
}
template <typename T>
void Put(std::vector<uint8_t>& data, T v) {
  const auto* begin = reinterpret_cast<const uint8_t*>(&v);
  data.insert(data.end(), begin, begin + sizeof(v));
}
std::vector<uint8_t> Read(std::istream& stream, size_t size) {
  std::vector<uint8_t> bytes(size);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("Truncated array or archive");
  return bytes;
}
Array Decode(const std::vector<uint8_t>& data) {
  if (data.size() < 10 || std::memcmp(data.data(), "\x93NUMPY", 6))
    throw std::runtime_error("Expected a NumPy array");
  const size_t prefix = data[6] == 1 ? 10 : 12;
  if (data.size() < prefix) throw std::runtime_error("Truncated NumPy header");
  const size_t length =
      prefix == 10 ? Get<uint16_t>(&data[8]) : Get<uint32_t>(&data[8]);
  if (prefix + length > data.size())
    throw std::runtime_error("Truncated NumPy header");
  const std::string header(reinterpret_cast<const char*>(&data[prefix]),
                           length);
  std::smatch match;
  Array out;
  if (!std::regex_search(header, match,
                         std::regex("'descr'\\s*:\\s*'([^']+)'")))
    throw std::runtime_error("Missing array dtype");
  out.dtype = match[1];
  if (out.dtype.front() == '>')
    throw std::runtime_error("Expected a little-endian array");
  if (!std::regex_search(header, match,
                         std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)")))
    throw std::runtime_error("Missing array shape");
  std::stringstream dimensions(match[1]);
  std::string dimension;
  while (std::getline(dimensions, dimension, ',')) {
    if (dimension.find_first_not_of(' ') != std::string::npos)
      out.shape.push_back(std::stoull(dimension));
  }
  if (out.Count() * out.ItemSize() != data.size() - prefix - length)
    throw std::runtime_error("Array shape and payload disagree");
  out.bytes.assign(data.begin() + prefix + length, data.end());
  if (header.find("True") != std::string::npos && out.shape.size() > 1) {
    // Existing descriptor caches use Fortran order; normalize the payload once.
    auto column_major = std::move(out.bytes);
    out.bytes.resize(column_major.size());
    std::vector<size_t> strides(out.shape.size(), 1);
    for (size_t k = 1; k < strides.size(); ++k)
      strides[k] = strides[k - 1] * out.shape[k - 1];
    const auto item_size = out.ItemSize();
    for (size_t i = 0; i < out.Count(); ++i) {
      size_t remainder = i, source = 0;
      for (size_t k = out.shape.size(); k-- > 0;) {
        source += (remainder % out.shape[k]) * strides[k];
        remainder /= out.shape[k];
      }
      std::memcpy(out.bytes.data() + i * item_size,
                  column_major.data() + source * item_size, item_size);
    }
  }
  return out;
}
std::vector<uint8_t> Encode(const Array& array) {
  if (array.Count() * array.ItemSize() != array.bytes.size())
    throw std::runtime_error("Array shape and payload disagree");
  std::ostringstream text;
  text << "{'descr': '" << array.dtype
       << "', 'fortran_order': False, 'shape': (";
  for (const auto n : array.shape) text << n << ", ";
  text << "), }";
  std::string header = text.str();
  header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
  header.push_back('\n');
  if (header.size() > 65535)
    throw std::runtime_error("Array header is too large");
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  Put<uint16_t>(out, static_cast<uint16_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), array.bytes.begin(), array.bytes.end());
  return out;
}
std::vector<uint8_t> Inflate(const std::vector<uint8_t>& data, size_t size) {
  std::vector<uint8_t> output(size);
  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(data.data());
  stream.avail_in = static_cast<uInt>(data.size());
  stream.next_out = output.data();
  stream.avail_out = static_cast<uInt>(output.size());
  if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
    throw std::runtime_error("Cannot initialize ZIP decompression");
  const auto result = inflate(&stream, Z_FINISH);
  inflateEnd(&stream);
  if (result != Z_STREAM_END || stream.total_out != size)
    throw std::runtime_error("Invalid compressed array");
  return output;
}
std::vector<uint8_t> Deflate(const std::vector<uint8_t>& input) {
  z_stream stream{};
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("Cannot initialize ZIP compression");
  std::vector<uint8_t> output(
      deflateBound(&stream, static_cast<uLong>(input.size())));
  stream.next_in = const_cast<Bytef*>(input.data());
  stream.avail_in = static_cast<uInt>(input.size());
  stream.next_out = output.data();
  stream.avail_out = static_cast<uInt>(output.size());
  const auto result = deflate(&stream, Z_FINISH);
  const auto written = stream.total_out;
  deflateEnd(&stream);
  if (result != Z_STREAM_END) throw std::runtime_error("Cannot compress array");
  output.resize(written);
  return output;
}
}  // namespace

size_t Array::Count() const {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<size_t>());
}
size_t Array::ItemSize() const {
  if (dtype.size() < 3 || (dtype[0] != '<' && dtype[0] != '|'))
    throw std::runtime_error("Unsupported array dtype: " + dtype);
  return std::stoull(dtype.substr(2)) * (dtype[1] == 'U' ? 4 : 1);
}
Array ReadArray(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot read array: " + path.u8string());
  return Decode(Read(stream, std::filesystem::file_size(path)));
}
void WriteArray(const std::filesystem::path& path, const Array& array) {
  std::ofstream stream(path, std::ios::binary);
  const auto data = Encode(array);
  stream.write(reinterpret_cast<const char*>(data.data()), data.size());
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write array: " + path.u8string());
}
std::map<std::string, Array> ReadArrays(
    const std::filesystem::path& path, const std::vector<std::string>& fields) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot read archive: " + path.u8string());
  const auto size = std::filesystem::file_size(path);
  const auto length = std::min<uint64_t>(size, 65557);
  stream.seekg(size - length);
  const auto tail = Read(stream, length);
  int64_t end = static_cast<int64_t>(tail.size()) - 22;
  while (end >= 0 && Get<uint32_t>(&tail[end]) != 0x06054b50) --end;
  if (end < 0) throw std::runtime_error("Missing ZIP directory");
  const auto count = Get<uint16_t>(&tail[end + 10]);
  stream.seekg(Get<uint32_t>(&tail[end + 16]));
  std::map<std::string, Array> arrays;
  for (unsigned i = 0; i < count; ++i) {
    const auto header = Read(stream, 46);
    if (Get<uint32_t>(header.data()) != 0x02014b50)
      throw std::runtime_error("Invalid ZIP directory");
    const auto method = Get<uint16_t>(&header[10]);
    const auto compressed = Get<uint32_t>(&header[20]),
               uncompressed = Get<uint32_t>(&header[24]);
    const auto name = Read(stream, Get<uint16_t>(&header[28]));
    stream.seekg(Get<uint16_t>(&header[30]) + Get<uint16_t>(&header[32]),
                 std::ios::cur);
    const std::string member(name.begin(), name.end());
    if (member.size() < 4 || member.substr(member.size() - 4) != ".npy")
      continue;
    const auto key = member.substr(0, member.size() - 4);
    if (!fields.empty() &&
        std::find(fields.begin(), fields.end(), key) == fields.end())
      continue;
    const auto directory_position = stream.tellg();
    stream.seekg(Get<uint32_t>(&header[42]));
    const auto local = Read(stream, 30);
    if (Get<uint32_t>(local.data()) != 0x04034b50)
      throw std::runtime_error("Invalid ZIP member");
    stream.seekg(Get<uint16_t>(&local[26]) + Get<uint16_t>(&local[28]),
                 std::ios::cur);
    auto data = Read(stream, compressed);
    if (method == 8)
      data = Inflate(data, uncompressed);
    else if (method != 0)
      throw std::runtime_error("Unsupported ZIP compression");
    if (data.size() != uncompressed ||
        crc32(0, data.data(), static_cast<uInt>(data.size())) !=
            Get<uint32_t>(&header[16]))
      throw std::runtime_error("Corrupt array archive");
    arrays.emplace(key, Decode(data));
    stream.seekg(directory_position);
  }
  for (const auto& field : fields)
    if (!arrays.count(field))
      throw std::runtime_error("Missing array: " + field);
  return arrays;
}
void WriteArrays(const std::filesystem::path& path,
                 const std::map<std::string, Array>& arrays, bool compress) {
  std::ofstream stream(path, std::ios::binary);
  std::vector<uint8_t> directory;
  for (const auto& [name, array] : arrays) {
    const auto bytes = Encode(array);
    const auto data = compress ? Deflate(bytes) : bytes;
    const auto filename = name + ".npy";
    const auto crc = crc32(0, bytes.data(), static_cast<uInt>(bytes.size()));
    const auto position = static_cast<uint32_t>(stream.tellp());
    std::vector<uint8_t> header;
    Put<uint32_t>(header, 0x04034b50);
    Put<uint16_t>(header, 20);
    Put<uint16_t>(header, 0);
    Put<uint16_t>(header, compress ? 8 : 0);
    Put<uint32_t>(header, 0);
    Put<uint32_t>(header, crc);
    Put<uint32_t>(header, static_cast<uint32_t>(data.size()));
    Put<uint32_t>(header, static_cast<uint32_t>(bytes.size()));
    Put<uint16_t>(header, static_cast<uint16_t>(filename.size()));
    Put<uint16_t>(header, 0);
    stream.write(reinterpret_cast<const char*>(header.data()), header.size());
    stream.write(filename.data(), filename.size());
    stream.write(reinterpret_cast<const char*>(data.data()), data.size());
    Put<uint32_t>(directory, 0x02014b50);
    Put<uint16_t>(directory, 20);
    directory.insert(directory.end(), header.begin() + 4, header.end());
    Put<uint16_t>(directory, 0);
    Put<uint16_t>(directory, 0);
    Put<uint16_t>(directory, 0);
    Put<uint32_t>(directory, 0);
    Put<uint32_t>(directory, position);
    directory.insert(directory.end(), filename.begin(), filename.end());
  }
  const auto position = static_cast<uint32_t>(stream.tellp());
  stream.write(reinterpret_cast<const char*>(directory.data()),
               directory.size());
  std::vector<uint8_t> end;
  Put<uint32_t>(end, 0x06054b50);
  Put<uint32_t>(end, 0);
  Put<uint16_t>(end, static_cast<uint16_t>(arrays.size()));
  Put<uint16_t>(end, static_cast<uint16_t>(arrays.size()));
  Put<uint32_t>(end, static_cast<uint32_t>(directory.size()));
  Put<uint32_t>(end, position);
  Put<uint16_t>(end, 0);
  stream.write(reinterpret_cast<const char*>(end.data()), end.size());
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write array archive: " + path.u8string());
}
std::string ArrayString(const Array& array) {
  if (array.shape.size() || array.dtype.substr(0, 2) != "<U")
    throw std::runtime_error("Expected a Unicode scalar");
  std::u32string text(array.bytes.size() / 4, 0);
  std::memcpy(text.data(), array.bytes.data(), array.bytes.size());
  while (!text.empty() && !text.back()) text.pop_back();
  return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.to_bytes(
      text);
}
Array StringArray(const std::string& value) {
  const auto text =
      std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.from_bytes(
          value);
  Array array{"<U" + std::to_string(std::max<size_t>(1, text.size())), {}, {}};
  array.bytes.resize(array.ItemSize(), 0);
  std::memcpy(array.bytes.data(), text.data(), text.size() * 4);
  return array;
}
}  // namespace geosunpre

#include "io/crop.h"
#include "sfm/shutter.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
extern "C" {
#include <jpeglib.h>
}

namespace geosun {
namespace {
struct JpegError {
  jpeg_error_mgr manager;
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};
void JpegExit(j_common_ptr info) {
  auto* error = reinterpret_cast<JpegError*>(info->err);
  (*info->err->format_message)(info, error->message);
  std::longjmp(error->jump, 1);
}
std::vector<uint8_t> ReadBytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  std::vector<uint8_t> bytes(fs::file_size(path));
  if (!stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
    throw std::runtime_error("Cannot read: " + path.u8string());
  return bytes;
}
void WriteBytes(const fs::path& path, const uint8_t* data, size_t size) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data), std::streamsize(size));
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write: " + path.u8string());
}
// Inclusive extent [begin, end) of the non-zero pixels of all masks of one lens.
struct Extent {
  int x0 = INT_MAX, y0 = INT_MAX, x1 = 0, y1 = 0;
};
bool IsExifApp1(const uint8_t* data, size_t size) {
  return size >= 6 && !std::memcmp(data, "Exif\0\0", 6);
}
std::vector<uint8_t> BuildCalibratedExifApp1(double relative_ev) {
  const double ev = std::clamp(relative_ev, -12.0, 12.0);
  const double t_i = 0.01 * std::exp2(ev);
  uint32_t num = 1, den = 100;
  if (t_i >= 0.01) {
    den = 1000000;
    num = static_cast<uint32_t>(std::clamp<long long>(std::llround(t_i * 1000000.0), 1LL, 4000000000LL));
  } else {
    num = 1000;
    den = static_cast<uint32_t>(std::clamp<long long>(std::llround(1000.0 / t_i), 1LL, 4000000000LL));
  }
  std::vector<uint8_t> payload(102, 0);
  std::memcpy(payload.data(), "Exif\0\0", 6);
  uint8_t* t = payload.data() + 6;
  t[0] = 'I';
  t[1] = 'I';
  auto put_u16 = [&](size_t off, uint16_t v) {
    t[off] = uint8_t(v & 0xffu);
    t[off + 1] = uint8_t((v >> 8) & 0xffu);
  };
  auto put_u32 = [&](size_t off, uint32_t v) {
    t[off] = uint8_t(v & 0xffu);
    t[off + 1] = uint8_t((v >> 8) & 0xffu);
    t[off + 2] = uint8_t((v >> 16) & 0xffu);
    t[off + 3] = uint8_t((v >> 24) & 0xffu);
  };
  auto put_entry = [&](size_t off, uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    put_u16(off, tag);
    put_u16(off + 2, type);
    put_u32(off + 4, count);
    put_u32(off + 8, value);
  };
  put_u16(2, 42);
  put_u32(4, 8);
  // IFD0 at 8: Orientation = 1, ExifIFD = 38.
  put_u16(8, 2);
  put_entry(10, 0x0112, 3, 1, 1);
  put_entry(22, 0x8769, 4, 1, 38);
  put_u32(34, 0);
  // ExifIFD at 38: ExposureTime at 80, FNumber at 88, ISOSpeedRatings = 100.
  put_u16(38, 3);
  put_entry(40, 0x829A, 5, 1, 80);
  put_entry(52, 0x829D, 5, 1, 88);
  put_entry(64, 0x8827, 3, 1, 100);
  put_u32(76, 0);
  put_u32(80, num);
  put_u32(84, den);
  put_u32(88, 1);
  put_u32(92, 1);
  return payload;
}
std::optional<double> ParseTiffExifEv(const uint8_t* tiff, size_t size) {
  if (size < 8)
    return std::nullopt;
  const bool little = tiff[0] == 'I' && tiff[1] == 'I';
  const bool big = tiff[0] == 'M' && tiff[1] == 'M';
  if (!little && !big)
    return std::nullopt;
  auto fits = [&](size_t off, size_t n) { return off <= size && n <= size - off; };
  auto u16 = [&](size_t off) -> std::optional<uint16_t> {
    if (!fits(off, 2))
      return std::nullopt;
    return little ? uint16_t(tiff[off] | (uint16_t(tiff[off + 1]) << 8))
                  : uint16_t((uint16_t(tiff[off]) << 8) | tiff[off + 1]);
  };
  auto u32 = [&](size_t off) -> std::optional<uint32_t> {
    if (!fits(off, 4))
      return std::nullopt;
    if (little)
      return uint32_t(tiff[off]) | (uint32_t(tiff[off + 1]) << 8) |
             (uint32_t(tiff[off + 2]) << 16) | (uint32_t(tiff[off + 3]) << 24);
    return (uint32_t(tiff[off]) << 24) | (uint32_t(tiff[off + 1]) << 16) |
           (uint32_t(tiff[off + 2]) << 8) | uint32_t(tiff[off + 3]);
  };
  const auto magic = u16(2);
  const auto ifd0 = u32(4);
  if (!magic || *magic != 42 || !ifd0)
    return std::nullopt;

  std::optional<double> exposure_time, f_number, iso, iso_sos, iso_rei, iso_speed, shutter_apex,
      aperture_apex;
  std::optional<uint32_t> exif_ifd;

  auto read_unsigned = [&](uint16_t type, uint32_t count, size_t entry_off,
                           uint32_t val_off) -> std::optional<double> {
    size_t elem = 0;
    if (type == 3)
      elem = 2;
    else if (type == 4)
      elem = 4;
    else if (type == 5)
      elem = 8;
    if (elem == 0 || count == 0)
      return std::nullopt;
    const size_t data_off = elem * size_t(count) <= 4 ? entry_off + 8 : size_t(val_off);
    if (type == 3) {
      const auto v = u16(data_off);
      return v ? std::optional<double>(double(*v)) : std::nullopt;
    }
    if (type == 4) {
      const auto v = u32(data_off);
      return v ? std::optional<double>(double(*v)) : std::nullopt;
    }
    const auto num = u32(data_off), den = u32(data_off + 4);
    if (!num || !den || *den == 0)
      return std::nullopt;
    return double(*num) / double(*den);
  };
  auto read_signed = [&](uint16_t type, uint32_t count, size_t entry_off,
                         uint32_t val_off) -> std::optional<double> {
    if (type == 3 || type == 4 || type == 5)
      return read_unsigned(type, count, entry_off, val_off);
    size_t elem = 0;
    if (type == 9)
      elem = 4;
    else if (type == 10)
      elem = 8;
    if (elem == 0 || count == 0)
      return std::nullopt;
    const size_t data_off = elem * size_t(count) <= 4 ? entry_off + 8 : size_t(val_off);
    if (type == 9) {
      const auto v = u32(data_off);
      return v ? std::optional<double>(double(int32_t(*v))) : std::nullopt;
    }
    const auto num = u32(data_off), den = u32(data_off + 4);
    if (!num || !den || *den == 0)
      return std::nullopt;
    return double(int32_t(*num)) / double(int32_t(*den));
  };

  auto parse_ifd = [&](uint32_t offset) -> bool {
    const auto count = u16(offset);
    if (!count || *count > 256 || !fits(size_t(offset) + 2, size_t(*count) * 12))
      return false;
    for (uint16_t i = 0; i < *count; ++i) {
      const size_t eoff = size_t(offset) + 2 + size_t(i) * 12;
      const auto tag = u16(eoff), type = u16(eoff + 2);
      const auto cnt = u32(eoff + 4), val = u32(eoff + 8);
      if (!tag || !type || !cnt || !val)
        return false;
      switch (*tag) {
      case 0x8769: {
        const auto v = read_unsigned(*type, *cnt, eoff, *val);
        if (v && *v >= 0.0 && *v <= double(UINT32_MAX))
          exif_ifd = uint32_t(*v);
        break;
      }
      case 0x829A:
        exposure_time = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x829D:
        f_number = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x8827:
        iso = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x8831:
        iso_sos = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x8832:
        iso_rei = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x8833:
        iso_speed = read_unsigned(*type, *cnt, eoff, *val);
        break;
      case 0x9201:
        shutter_apex = read_signed(*type, *cnt, eoff, *val);
        break;
      case 0x9202:
        aperture_apex = read_unsigned(*type, *cnt, eoff, *val);
        break;
      default:
        break;
      }
    }
    return true;
  };
  if (!parse_ifd(*ifd0))
    return std::nullopt;
  if (exif_ifd && !parse_ifd(*exif_ifd))
    return std::nullopt;

  auto pos_opt = [](std::optional<double> v) -> std::optional<double> {
    return v && std::isfinite(*v) && *v > 0.0 ? v : std::nullopt;
  };
  std::optional<double> t = pos_opt(exposure_time);
  if (!t && shutter_apex && std::isfinite(*shutter_apex))
    t = pos_opt(std::exp2(-*shutter_apex));
  std::optional<double> fno = pos_opt(f_number);
  if (!fno && aperture_apex && std::isfinite(*aperture_apex))
    fno = pos_opt(std::exp2(*aperture_apex * 0.5));
  std::optional<double> spd;
  for (const auto& c : {iso, iso_sos, iso_rei, iso_speed}) {
    if (auto ok = pos_opt(c)) {
      spd = ok;
      break;
    }
  }
  if (!t && !fno && !spd)
    return std::nullopt;
  const double ev = std::log2(t.value_or(1.0) / (fno.value_or(1.0) * fno.value_or(1.0)) *
                              spd.value_or(1.0));
  return std::isfinite(ev) ? std::optional<double>(ev) : std::nullopt;
}
}  // namespace
std::array<CropWindow, 2> ValidDomainCrop(const Scene& scene, const Reconstruction& model,
                                          const fs::path& masks, int readout_axis, int mcu,
                                          int workers) {
  std::vector<const Exposure*> registered;
  for (const auto& e : scene.images)
    if (model.images[e.id].registered)
      registered.push_back(&e);
  // Each mask's own extent; the row and column maxima bound its non-zero pixels.
  std::vector<Extent> boxes(registered.size());
  ParallelFor(registered.size(), workers, 1, [&](size_t i) {
    const auto& e = *registered[i];
    const auto mask = ReadImage(masks / (std::to_string(e.id) + ".png"), cv::IMREAD_GRAYSCALE);
    const auto& camera = model.cameras[e.camera];
    if (mask.cols != camera.width || mask.rows != camera.height)
      throw std::runtime_error("Static mask size differs from its camera: " + e.name);
    cv::Mat columns, rows;
    cv::reduce(mask, columns, 0, cv::REDUCE_MAX);
    cv::reduce(mask, rows, 1, cv::REDUCE_MAX);
    auto& box = boxes[i];
    for (int x = 0; x < mask.cols; ++x)
      if (columns.at<uint8_t>(0, x)) {
        box.x0 = std::min(box.x0, x);
        box.x1 = x + 1;
      }
    for (int y = 0; y < mask.rows; ++y)
      if (rows.at<uint8_t>(y, 0)) {
        box.y0 = std::min(box.y0, y);
        box.y1 = y + 1;
      }
  });
  std::array<Extent, 2> extents;
  for (size_t i = 0; i < registered.size(); ++i) {
    const auto& box = boxes[i];
    if (box.x1 <= box.x0)
      continue;
    auto& extent = extents[registered[i]->camera];
    extent.x0 = std::min(extent.x0, box.x0);
    extent.y0 = std::min(extent.y0, box.y0);
    extent.x1 = std::max(extent.x1, box.x1);
    extent.y1 = std::max(extent.y1, box.y1);
  }
  // Observations of the original rolling-shutter photos may sit a few pixels off the masks.
  constexpr int Margin = 16;
  std::array<CropWindow, 2> windows;
  for (int c = 0; c < 2; ++c) {
    const auto& camera = model.cameras[c];
    const auto& extent = extents[c];
    if (extent.x1 <= extent.x0 || extent.y1 <= extent.y0)
      throw std::runtime_error("Camera " + camera.label + " has no valid mask pixels to crop to");
    // Start on the MCU grid along both axes; along the readout axis also cut the far side by the
    // same amount so the window stays centred.
    const std::array<int, 2> size{camera.width, camera.height};
    std::array<int, 2> begin{std::max(0, extent.x0 - Margin) / mcu * mcu,
                             std::max(0, extent.y0 - Margin) / mcu * mcu};
    const std::array<int, 2> end{std::min(size[0], extent.x1 + Margin),
                                 std::min(size[1], extent.y1 + Margin)};
    const int a = readout_axis;
    begin[a] = std::min(begin[a], (size[a] - end[a]) / mcu * mcu);
    std::array<int, 2> extent_size{end[0] - begin[0], end[1] - begin[1]};
    extent_size[a] = size[a] - 2 * begin[a];
    // Downstream trainers divide sizes by small resize factors; round the free axis up to the MCU
    // edge when the image has room.
    const int b = 1 - a;
    extent_size[b] = std::min((extent_size[b] + mcu - 1) / mcu * mcu, size[b] - begin[b]);
    auto& w = windows[c];
    w.x = begin[0];
    w.y = begin[1];
    w.width = extent_size[0];
    w.height = extent_size[1];
    if (w.width <= 0 || w.height <= 0 || w.x + w.width > size[0] || w.y + w.height > size[1])
      throw std::runtime_error("Invalid crop window for camera " + camera.label);
  }
  return windows;
}
namespace {
void CropCameras(std::array<Camera, 2>& cameras, const std::array<CropWindow, 2>& windows) {
  for (int c = 0; c < 2; ++c) {
    auto& camera = cameras[c];
    camera.width = windows[c].width;
    camera.height = windows[c].height;
    camera.params[2] -= windows[c].x;
    camera.params[3] -= windows[c].y;
  }
}
}  // namespace
Scene CropScene(const Scene& scene, const std::array<CropWindow, 2>& windows) {
  auto cropped = scene;
  CropCameras(cropped.cameras, windows);
  return cropped;
}
Reconstruction CropModel(const Reconstruction& model, const std::array<CropWindow, 2>& windows,
                         const ShutterMotion* motion) {
  auto cropped = model;
  CropCameras(cropped.cameras, windows);
  for (auto& image : cropped.images) {
    const Eigen::Vector2d offset(windows[image.camera].x, windows[image.camera].y);
    for (auto& pixel : image.pixels) {
      pixel -= offset;
      const auto& camera = cropped.cameras[image.camera];
      if (pixel.x() < 0 || pixel.y() < 0 || pixel.x() > camera.width || pixel.y() > camera.height)
        throw std::runtime_error("Observation lies outside the crop window");
    }
  }
  for (auto& point : cropped.points)
    point.error = motion ? ShutterPointError(cropped, *motion, point) : cropped.PointError(point);
  return cropped;
}
ShutterMotion CropMotion(const ShutterMotion& motion, const Reconstruction& uncropped,
                         const std::array<CropWindow, 2>& windows) {
  auto cropped = motion;
  std::array<double, 2> scale{};
  for (int c = 0; c < 2; ++c) {
    const auto& camera = uncropped.cameras[c];
    scale[c] = motion.axis ? double(windows[c].height) / camera.height
                           : double(windows[c].width) / camera.width;
    cropped.readout[c] *= scale[c];
  }
  for (size_t i = 0; i < uncropped.images.size(); ++i)
    cropped.rotation[i] *= scale[uncropped.images[i].camera];
  return cropped;
}
int JpegMcuSize(const fs::path& path) {
  const auto bytes = ReadBytes(path);
  jpeg_decompress_struct info{};
  JpegError error{};
  info.err = jpeg_std_error(&error.manager);
  error.manager.error_exit = JpegExit;
  if (setjmp(error.jump)) {
    jpeg_destroy_decompress(&info);
    throw std::runtime_error(std::string("Cannot read JPEG header: ") + error.message);
  }
  jpeg_create_decompress(&info);
  jpeg_mem_src(&info, bytes.data(), static_cast<unsigned long>(bytes.size()));
  jpeg_read_header(&info, TRUE);
  const int mcu = std::max(info.max_h_samp_factor, info.max_v_samp_factor) * DCTSIZE;
  jpeg_destroy_decompress(&info);
  return mcu;
}
void CropJpeg(const fs::path& source, const fs::path& destination, const CropWindow& window) {
  const auto bytes = ReadBytes(source);
  jpeg_decompress_struct in{};
  jpeg_compress_struct out{};
  JpegError error{};
  unsigned char* buffer = nullptr;
  unsigned long buffer_size = 0;
  // Everything with a destructor exists before setjmp, so a libjpeg error jumps past none.
  std::vector<jvirt_barray_ptr> arrays;
  std::vector<JDIMENSION> width_blocks, height_blocks;
  // Both objects share one error manager, so a single jump target covers them.
  in.err = out.err = jpeg_std_error(&error.manager);
  error.manager.error_exit = JpegExit;
  if (setjmp(error.jump)) {
    // The output buffer is left to the process: after the jump its pointer may be stale.
    jpeg_destroy_compress(&out);
    jpeg_destroy_decompress(&in);
    throw std::runtime_error("Cannot crop JPEG " + source.u8string() + ": " + error.message);
  }
  jpeg_create_decompress(&in);
  jpeg_create_compress(&out);
  jpeg_mem_src(&in, bytes.data(), static_cast<unsigned long>(bytes.size()));
  jpeg_save_markers(&in, JPEG_COM, 0xffff);
  for (int m = 0; m < 16; ++m)
    jpeg_save_markers(&in, JPEG_APP0 + m, 0xffff);
  jpeg_read_header(&in, TRUE);
  const int mcu_w = in.max_h_samp_factor * DCTSIZE, mcu_h = in.max_v_samp_factor * DCTSIZE;
  if (window.x % mcu_w || window.y % mcu_h || window.x + window.width > int(in.image_width) ||
      window.y + window.height > int(in.image_height) || window.width <= 0 || window.height <= 0) {
    jpeg_destroy_compress(&out);
    jpeg_destroy_decompress(&in);
    throw std::runtime_error("Crop window is off the JPEG MCU grid or outside " + source.u8string());
  }
  // Destination coefficient arrays must be requested before the source realizes its own.
  arrays.resize(in.num_components);
  width_blocks.resize(in.num_components);
  height_blocks.resize(in.num_components);
  for (int ci = 0; ci < in.num_components; ++ci) {
    const auto& comp = in.comp_info[ci];
    width_blocks[ci] = static_cast<JDIMENSION>(
        (long(window.width) * comp.h_samp_factor + mcu_w - 1) / mcu_w);
    height_blocks[ci] = static_cast<JDIMENSION>(
        (long(window.height) * comp.v_samp_factor + mcu_h - 1) / mcu_h);
    const auto rounded_w = (width_blocks[ci] + comp.h_samp_factor - 1) / comp.h_samp_factor *
                           comp.h_samp_factor;
    const auto rounded_h = (height_blocks[ci] + comp.v_samp_factor - 1) / comp.v_samp_factor *
                           comp.v_samp_factor;
    arrays[ci] = (*in.mem->request_virt_barray)(reinterpret_cast<j_common_ptr>(&in), JPOOL_IMAGE,
                                                TRUE, rounded_w, rounded_h, comp.v_samp_factor);
  }
  jvirt_barray_ptr* source_arrays = jpeg_read_coefficients(&in);
  for (int ci = 0; ci < in.num_components; ++ci) {
    const auto& comp = in.comp_info[ci];
    const JDIMENSION x_blocks = window.x / mcu_w * comp.h_samp_factor;
    const JDIMENSION y_blocks = window.y / mcu_h * comp.v_samp_factor;
    const JDIMENSION rows = (height_blocks[ci] + comp.v_samp_factor - 1) / comp.v_samp_factor *
                            comp.v_samp_factor;
    const JDIMENSION cols = (width_blocks[ci] + comp.h_samp_factor - 1) / comp.h_samp_factor *
                            comp.h_samp_factor;
    for (JDIMENSION row = 0; row < rows; row += comp.v_samp_factor) {
      JBLOCKARRAY destination_rows = (*in.mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(&in), arrays[ci], row, comp.v_samp_factor, TRUE);
      for (int k = 0; k < comp.v_samp_factor; ++k) {
        // Rows past the source's last block row repeat it; they lie outside the image.
        const JDIMENSION source_row =
            std::min<JDIMENSION>(row + k + y_blocks, comp.height_in_blocks - 1);
        JBLOCKARRAY source_rows = (*in.mem->access_virt_barray)(
            reinterpret_cast<j_common_ptr>(&in), source_arrays[ci], source_row, 1, FALSE);
        for (JDIMENSION col = 0; col < cols; ++col) {
          const JDIMENSION source_col =
              std::min<JDIMENSION>(col + x_blocks, comp.width_in_blocks - 1);
          std::memcpy(destination_rows[k][col], source_rows[0][source_col], sizeof(JBLOCK));
        }
      }
    }
  }
  jpeg_mem_dest(&out, &buffer, &buffer_size);
  jpeg_copy_critical_parameters(&in, &out);
  out.image_width = window.width;
  out.image_height = window.height;
  jpeg_write_coefficients(&out, arrays.data());
  for (auto* marker = in.marker_list; marker; marker = marker->next) {
    // The compressor already wrote JFIF and Adobe headers matching the copied parameters.
    if (out.write_JFIF_header && marker->marker == JPEG_APP0 && marker->data_length >= 5 &&
        !std::memcmp(marker->data, "JFIF", 5))
      continue;
    if (out.write_Adobe_marker && marker->marker == JPEG_APP0 + 14 && marker->data_length >= 5 &&
        !std::memcmp(marker->data, "Adobe", 5))
      continue;
    jpeg_write_marker(&out, marker->marker, marker->data, marker->data_length);
  }
  jpeg_finish_compress(&out);
  jpeg_finish_decompress(&in);
  jpeg_destroy_compress(&out);
  jpeg_destroy_decompress(&in);
  WriteBytes(destination, buffer, buffer_size);
  std::free(buffer);
}
std::optional<double> ReadJpegExifEv(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return std::nullopt;
  std::vector<uint8_t> bytes(65536);
  stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
  bytes.resize(size_t(stream.gcount()));
  if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8)
    return std::nullopt;
  size_t pos = 2;
  while (pos < bytes.size()) {
    if (bytes[pos] != 0xff)
      return std::nullopt;
    while (pos < bytes.size() && bytes[pos] == 0xff)
      ++pos;
    if (pos >= bytes.size())
      return std::nullopt;
    const uint8_t marker = bytes[pos++];
    if (marker == 0xda || marker == 0xd9)
      return std::nullopt;
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7))
      continue;
    if (pos + 2 > bytes.size())
      return std::nullopt;
    const size_t seg_len = (size_t(bytes[pos]) << 8) | size_t(bytes[pos + 1]);
    if (seg_len < 2 || pos + seg_len > bytes.size())
      return std::nullopt;
    const size_t payload_off = pos + 2;
    const size_t payload_len = seg_len - 2;
    if (marker == 0xe1 && IsExifApp1(bytes.data() + payload_off, payload_len))
      return ParseTiffExifEv(bytes.data() + payload_off + 6, payload_len - 6);
    pos += seg_len;
  }
  return std::nullopt;
}
void CropMask(const fs::path& source, const fs::path& destination, const CropWindow& window) {
  const auto mask = ReadImage(source, cv::IMREAD_GRAYSCALE);
  if (window.x + window.width > mask.cols || window.y + window.height > mask.rows)
    throw std::runtime_error("Crop window lies outside mask " + source.u8string());
  WriteImage(destination, mask(cv::Rect(window.x, window.y, window.width, window.height)).clone());
}
void ExportNormalizedJpeg(const fs::path& source, const fs::path& destination,
                          const std::optional<CropWindow>& window,
                          const PhotometricCorrection& correction, int jpeg_quality) {
  static const std::array<float, 256> kSrgbToLin = [] {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) {
      const double c = double(i) / 255.0;
      t[size_t(i)] = float(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
    }
    return t;
  }();
  static constexpr int kLinLutSize = 16384;
  static const std::vector<uint8_t> kLinToSrgb = []() {
    std::vector<uint8_t> t(kLinLutSize + 1);
    for (int i = 0; i <= kLinLutSize; ++i) {
      const double x = double(i) / double(kLinLutSize);
      const double s = x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
      t[size_t(i)] = static_cast<uint8_t>(std::clamp(std::lround(s * 255.0), 0L, 255L));
    }
    return t;
  }();

  const auto& cam = correction.source_camera;
  const double fx = cam.params[0], fy = cam.params[1], cx = cam.params[2], cy = cam.params[3];
  const double k1 = cam.params[4], k2 = cam.params[5], k3 = cam.params[6], k4 = cam.params[7];
  if (fx <= 0.0 || fy <= 0.0)
    throw std::runtime_error("Invalid camera focal length for photometric export");
  double rd_max = 0.0;
  for (const double px : {0.0, double(cam.width)}) {
    for (const double py : {0.0, double(cam.height)}) {
      rd_max = std::max(rd_max, std::hypot((px - cx) / fx, (py - cy) / fy));
    }
  }
  rd_max = std::max(rd_max, 1e-6);
  constexpr int kVigLutSize = 4096;
  constexpr double kMaxThetaRad = 80.0 * Pi / 180.0;
  std::array<std::vector<float>, 3> inv_vig_lut;
  for (int ch = 0; ch < 3; ++ch)
    inv_vig_lut[size_t(ch)].assign(kVigLutSize + 1, 1.0f);
  for (int i = 0; i <= kVigLutSize; ++i) {
    const double rd = rd_max * (double(i) / double(kVigLutSize));
    double theta = rd;
    if (rd > 1e-12) {
      for (int it = 0; it < 30; ++it) {
        const double t2 = theta * theta;
        const double f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - rd;
        const double df =
            1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
        if (df <= 1e-6) {
          theta = kMaxThetaRad;
          break;
        }
        const double delta = f / df;
        theta -= delta;
        if (theta >= kMaxThetaRad) {
          theta = kMaxThetaRad;
          break;
        }
        if (std::abs(delta) < 1e-10)
          break;
      }
    }
    const double rho = std::clamp(theta / kMaxThetaRad, 0.0, 1.0);
    const double r2 = rho * rho;
    for (int ch = 0; ch < 3; ++ch) {
      const auto& vc = correction.vig_coeffs_rgb[size_t(ch)];
      const double vig =
          std::clamp(std::exp2(vc[0] * r2 + vc[1] * r2 * r2 + vc[2] * r2 * r2 * r2), 0.40, 1.05);
      inv_vig_lut[size_t(ch)][size_t(i)] = float(1.0 / vig);
    }
  }

  const auto bytes = ReadBytes(source);
  jpeg_decompress_struct in{};
  jpeg_compress_struct out{};
  JpegError error{};
  unsigned char* buffer = nullptr;
  unsigned long buffer_size = 0;
  std::vector<uint8_t> decoded_rgb;
  std::vector<uint8_t> out_rgb;
  std::vector<uint8_t> calibrated_exif;
  in.err = out.err = jpeg_std_error(&error.manager);
  error.manager.error_exit = JpegExit;
  if (setjmp(error.jump)) {
    jpeg_destroy_compress(&out);
    jpeg_destroy_decompress(&in);
    throw std::runtime_error("Cannot normalize JPEG " + source.u8string() + ": " + error.message);
  }
  jpeg_create_decompress(&in);
  jpeg_create_compress(&out);
  jpeg_mem_src(&in, bytes.data(), static_cast<unsigned long>(bytes.size()));
  jpeg_save_markers(&in, JPEG_COM, 0xffff);
  for (int m = 0; m < 16; ++m)
    jpeg_save_markers(&in, JPEG_APP0 + m, 0xffff);
  jpeg_read_header(&in, TRUE);
  in.out_color_space = JCS_RGB;
  jpeg_start_decompress(&in);
  const int src_w = int(in.output_width), src_h = int(in.output_height);
  if (in.output_components != 3 || src_w != cam.width || src_h != cam.height) {
    jpeg_destroy_compress(&out);
    jpeg_destroy_decompress(&in);
    throw std::runtime_error("Unexpected image geometry/channels in " + source.u8string());
  }
  const int wx = window ? window->x : 0;
  const int wy = window ? window->y : 0;
  const int out_w = window ? window->width : src_w;
  const int out_h = window ? window->height : src_h;
  if (wx < 0 || wy < 0 || out_w <= 0 || out_h <= 0 || wx + out_w > src_w || wy + out_h > src_h) {
    jpeg_destroy_compress(&out);
    jpeg_destroy_decompress(&in);
    throw std::runtime_error("Crop window lies outside " + source.u8string());
  }

  decoded_rgb.resize(size_t(src_w) * size_t(src_h) * 3);
  while (in.output_scanline < in.output_height) {
    JSAMPROW row_ptr = decoded_rgb.data() + size_t(in.output_scanline) * size_t(src_w) * 3;
    jpeg_read_scanlines(&in, &row_ptr, 1);
  }

  out_rgb.resize(size_t(out_w) * size_t(out_h) * 3);
  const double exp_gain = std::exp2(-correction.pixel_ev_shift);
  const float g_r = float(exp_gain * correction.wb_gains[0]);
  const float g_g = float(exp_gain * correction.wb_gains[1]);
  const float g_b = float(exp_gain * correction.wb_gains[2]);
  const double rd_scale = double(kVigLutSize) / rd_max;
  constexpr float kHighlightKnee = 0.92f;
  constexpr float kHighlightHeadroom = 1.0f - kHighlightKnee;
  for (int y = 0; y < out_h; ++y) {
    const uint8_t* src_row = decoded_rgb.data() + (size_t(wy + y) * size_t(src_w) + size_t(wx)) * 3;
    uint8_t* dst_row = out_rgb.data() + size_t(y) * size_t(out_w) * 3;
    const double dy = (double(wy + y) + 0.5 - cy) / fy;
    const double dy2 = dy * dy;
    for (int x = 0; x < out_w; ++x) {
      const double dx = (double(wx + x) + 0.5 - cx) / fx;
      const double rd = std::sqrt(dx * dx + dy2);
      const double fidx = std::min(rd * rd_scale, double(kVigLutSize));
      const int idx0 = std::min(int(fidx), kVigLutSize - 1);
      const float frac = float(fidx - double(idx0));
      const float iv_r = inv_vig_lut[0][size_t(idx0)] * (1.0f - frac) +
                         inv_vig_lut[0][size_t(idx0 + 1)] * frac;
      const float iv_g = inv_vig_lut[1][size_t(idx0)] * (1.0f - frac) +
                         inv_vig_lut[1][size_t(idx0 + 1)] * frac;
      const float iv_b = inv_vig_lut[2][size_t(idx0)] * (1.0f - frac) +
                         inv_vig_lut[2][size_t(idx0 + 1)] * frac;

      const size_t px_off = size_t(x) * 3;
      auto soft_knee = [&](float v) -> float {
        if (v <= kHighlightKnee)
          return v;
        const float over = v - kHighlightKnee;
        return std::min(1.0f,
                        kHighlightKnee + kHighlightHeadroom * (over / (over + kHighlightHeadroom)));
      };
      const float lr = soft_knee(std::max(0.0f, kSrgbToLin[src_row[px_off + 0]] * g_r * iv_r));
      const float lg = soft_knee(std::max(0.0f, kSrgbToLin[src_row[px_off + 1]] * g_g * iv_g));
      const float lb = soft_knee(std::max(0.0f, kSrgbToLin[src_row[px_off + 2]] * g_b * iv_b));
      dst_row[px_off + 0] = kLinToSrgb[size_t(lr * float(kLinLutSize) + 0.5f)];
      dst_row[px_off + 1] = kLinToSrgb[size_t(lg * float(kLinLutSize) + 0.5f)];
      dst_row[px_off + 2] = kLinToSrgb[size_t(lb * float(kLinLutSize) + 0.5f)];
    }
  }

  jpeg_mem_dest(&out, &buffer, &buffer_size);
  out.image_width = JDIMENSION(out_w);
  out.image_height = JDIMENSION(out_h);
  out.input_components = 3;
  out.in_color_space = JCS_RGB;
  jpeg_set_defaults(&out);
  jpeg_set_quality(&out, std::clamp(jpeg_quality, 1, 100), TRUE);
  for (int ci = 0; ci < out.num_components; ++ci) {
    out.comp_info[ci].h_samp_factor = 1;
    out.comp_info[ci].v_samp_factor = 1;
  }
  jpeg_start_compress(&out, TRUE);
  calibrated_exif = BuildCalibratedExifApp1(correction.residual_ev);
  jpeg_write_marker(&out, JPEG_APP0 + 1, calibrated_exif.data(),
                    static_cast<unsigned int>(calibrated_exif.size()));
  for (auto* marker = in.marker_list; marker; marker = marker->next) {
    if (out.write_JFIF_header && marker->marker == JPEG_APP0 && marker->data_length >= 5 &&
        !std::memcmp(marker->data, "JFIF", 5))
      continue;
    if (out.write_Adobe_marker && marker->marker == JPEG_APP0 + 14 && marker->data_length >= 5 &&
        !std::memcmp(marker->data, "Adobe", 5))
      continue;
    if (marker->marker == JPEG_APP0 + 1 && IsExifApp1(marker->data, marker->data_length))
      continue;
    jpeg_write_marker(&out, marker->marker, marker->data, marker->data_length);
  }
  while (out.next_scanline < out.image_height) {
    JSAMPROW row_ptr = out_rgb.data() + size_t(out.next_scanline) * size_t(out_w) * 3;
    jpeg_write_scanlines(&out, &row_ptr, 1);
  }
  jpeg_finish_compress(&out);
  jpeg_finish_decompress(&in);
  jpeg_destroy_compress(&out);
  jpeg_destroy_decompress(&in);
  WriteBytes(destination, buffer, buffer_size);
  std::free(buffer);
}
}  // namespace geosun

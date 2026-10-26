#include "io/crop.h"
#include "sfm/shutter.h"
#include <climits>
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
}  // namespace
std::array<CropWindow, 2> ValidDomainCrop(const Scene& scene, const Reconstruction& model,
                                          const fs::path& masks, int readout_axis, int mcu) {
  std::array<Extent, 2> extents;
  for (const auto& e : scene.images) {
    if (!model.images[e.id].registered)
      continue;
    const auto mask = ReadImage(masks / (std::to_string(e.id) + ".png"), cv::IMREAD_GRAYSCALE);
    const auto& camera = model.cameras[e.camera];
    if (mask.cols != camera.width || mask.rows != camera.height)
      throw std::runtime_error("Static mask size differs from its camera: " + e.name);
    cv::Mat points;
    cv::findNonZero(mask, points);
    if (points.empty())
      continue;
    const auto box = cv::boundingRect(points);
    auto& extent = extents[e.camera];
    extent.x0 = std::min(extent.x0, box.x);
    extent.y0 = std::min(extent.y0, box.y);
    extent.x1 = std::max(extent.x1, box.x + box.width);
    extent.y1 = std::max(extent.y1, box.y + box.height);
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
  for (auto& point : cropped.points) {
    if (motion) {
      point.error = ShutterPointError(cropped, *motion, point);
      continue;
    }
    double error = 0;
    for (const auto& o : point.track) {
      const auto& image = cropped.images[o.image];
      error += (cropped.cameras[image.camera].Project(image.pose.Apply(point.xyz)) -
                image.pixels[o.feature])
                   .norm();
    }
    point.error = point.track.empty() ? 0 : error / point.track.size();
  }
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
void CropMask(const fs::path& source, const fs::path& destination, const CropWindow& window) {
  const auto mask = ReadImage(source, cv::IMREAD_GRAYSCALE);
  if (window.x + window.width > mask.cols || window.y + window.height > mask.rows)
    throw std::runtime_error("Crop window lies outside mask " + source.u8string());
  WriteImage(destination, mask(cv::Rect(window.x, window.y, window.width, window.height)).clone());
}
}  // namespace geosun

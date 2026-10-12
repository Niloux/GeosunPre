#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>

namespace {
constexpr double pi = 3.14159265358979323846;
double Cubic(double x) {
  x = std::abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return (((-.5 * x + 2.5) * x - 4) * x) + 2;
  return 0;
}
struct Coefficients {
  int start;
  std::vector<int32_t> values;
};
struct ImageMetadata {
  int width = 0, height = 0, orientation = 1;
  Json exposure = nullptr;
};
ImageMetadata ReadJpegMetadata(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  auto byte = [&]() {
    const int value = stream.get();
    if (value < 0)
      throw std::runtime_error("Truncated JPEG header: " + path.u8string());
    return value;
  };
  if (byte() != 0xff || byte() != 0xd8)
    throw std::runtime_error("Expected JPEG image: " + path.u8string());
  ImageMetadata result;
  for (;;) {
    if (byte() != 0xff) throw std::runtime_error("Invalid JPEG marker");
    int marker;
    do {
      marker = byte();
    } while (marker == 0xff);
    if (marker == 0xda || marker == 0xd9) break;
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    const int length = (byte() << 8) + byte() - 2;
    if (length < 0) throw std::runtime_error("Invalid JPEG segment");
    std::vector<uint8_t> data(size_t(length), 0);
    if (!stream.read(reinterpret_cast<char*>(data.data()), length))
      throw std::runtime_error("Truncated JPEG segment");
    if (((marker >= 0xc0 && marker <= 0xc3) ||
         (marker >= 0xc5 && marker <= 0xc7) ||
         (marker >= 0xc9 && marker <= 0xcb) ||
         (marker >= 0xcd && marker <= 0xcf)) &&
        length >= 5) {
      result.height = (data[1] << 8) | data[2];
      result.width = (data[3] << 8) | data[4];
    }
    if (marker == 0xe1 && length >= 14 &&
        !std::memcmp(data.data(), "Exif\0\0", 6)) {
      const size_t base = 6;
      const bool little = data[base] == 'I' && data[base + 1] == 'I';
      if (!little && !(data[base] == 'M' && data[base + 1] == 'M'))
        throw std::runtime_error("Invalid EXIF byte order");
      auto integer = [&](size_t offset, size_t bytes) {
        if (offset + bytes > data.size() - base)
          throw std::runtime_error("Truncated EXIF field");
        uint32_t value = 0;
        for (size_t i = 0; i < bytes; ++i)
          value |= uint32_t(data[base + offset + i])
                   << (8 * (little ? i : bytes - i - 1));
        return value;
      };
      if (integer(2, 2) != 42)
        throw std::runtime_error("Invalid EXIF TIFF header");
      const size_t table = integer(4, 4), count = integer(table, 2);
      for (size_t i = 0; i < count; ++i) {
        const size_t entry = table + 2 + i * 12;
        const auto tag = integer(entry, 2), type = integer(entry + 2, 2),
                   n = integer(entry + 4, 4);
        if (tag == 274 && type == 3 && n == 1)
          result.orientation = int(integer(entry + 8, 2));
        if (tag == 33434 && type == 5 && n == 1) {
          const size_t value = integer(entry + 8, 4);
          const auto denominator = integer(value + 4, 4);
          if (!denominator) throw std::runtime_error("Invalid exposure time");
          const double exposure = double(integer(value, 4)) / denominator;
          if (exposure) result.exposure = exposure;
        }
      }
    }
  }
  if (!result.width || !result.height)
    throw std::runtime_error("Missing JPEG dimensions");
  return result;
}
std::vector<Coefficients> ResizeWeights(int input, int output) {
  const double scale = double(input) / output,
               filter_scale = std::max(1., scale), support = 2 * filter_scale;
  std::vector<Coefficients> coefficients;
  for (int i = 0; i < output; ++i) {
    const double center = (i + .5) * scale;
    const int lo = std::max(0, int(center - support + .5));
    const int hi = std::min(input, int(center + support + .5));
    std::vector<double> raw;
    double sum = 0;
    for (int j = lo; j < hi; ++j) {
      raw.push_back(Cubic((j - center + .5) / filter_scale));
      sum += raw.back();
    }
    Coefficients row{lo, {}};
    for (double weight : raw) {
      weight = weight / sum * (1 << 22);
      row.values.push_back(int32_t(weight < 0 ? weight - .5 : weight + .5));
    }
    coefficients.push_back(std::move(row));
  }
  return coefficients;
}
std::pair<int, double> TimeName(const std::string& name, int& frame,
                                std::string& camera) {
  std::smatch parts;
  if (!std::regex_match(name, parts,
                        std::regex("(\\d+)-(\\d+)-(\\d+)_(\\d+)-([LRM])\\.JPG",
                                   std::regex::icase)))
    throw std::runtime_error("Unrecognized image filename: " + name);
  frame = std::stoi(parts[4]);
  camera = parts[5];
  std::transform(camera.begin(), camera.end(), camera.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return {std::stoi(parts[1]),
          std::stod(parts[2]) + std::stod(parts[3]) / 1000};
}
}  // namespace
Camera Camera::FromJson(const Json& value) {
  return {value.at("label"), value.at("width"), value.at("height"),
          value.at("params").get<std::array<double, 8>>(),
          value.at("principal_offset_px").get<std::array<double, 2>>()};
}
Json ImageHeader(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  std::array<uint8_t, 29> header{};
  stream.read(reinterpret_cast<char*>(header.data()), header.size());
  if (stream.gcount() == header.size() &&
      !std::memcmp(header.data(), "\x89PNG\r\n\x1a\n", 8)) {
    if (std::memcmp(header.data() + 12, "IHDR", 4))
      throw std::runtime_error("Missing PNG image header");
    auto integer = [&](int start) {
      uint32_t v = 0;
      for (int i = 0; i < 4; ++i) v = (v << 8) | header[start + i];
      return v;
    };
    return {{"width", integer(16)},
            {"height", integer(20)},
            {"orientation", 1},
            {"mode", header[24] == 8 && header[25] == 0 ? "L" : "other"}};
  }
  const auto image = ReadJpegMetadata(path);
  return {{"width", image.width},
          {"height", image.height},
          {"orientation", image.orientation},
          {"exposure_seconds", image.exposure}};
}
Json Camera::ToJson() const {
  return {{"label", label},
          {"width", width},
          {"height", height},
          {"params", params},
          {"principal_offset_px", principal_offset}};
}
cv::Matx33d Camera::K() const {
  return {params[0], 0, params[2], 0, params[1], params[3], 0, 0, 1};
}
cv::Vec4d Camera::D() const {
  return {params[4], params[5], params[6], params[7]};
}
double Camera::Radius(double fov) const {
  if (!std::all_of(params.begin(), params.end(),
                   [](double v) { return std::isfinite(v); }) ||
      params[0] <= 0 || params[1] <= 0)
    throw std::runtime_error(label + ": invalid fisheye calibration");
  const double maximum = fov / 2 * pi / 180;
  for (int j = 0; j < 256; ++j) {
    const double theta = maximum * j / 255;
    double derivative = 1;
    for (int i = 0; i < 4; ++i)
      derivative += (2 * i + 3) * params[4 + i] * std::pow(theta, 2 * i + 2);
    if (derivative <= 0)
      throw std::runtime_error(label + ": nonmonotonic fisheye calibration");
  }
  double value = 1;
  for (int i = 0; i < 4; ++i)
    value += params[4 + i] * std::pow(maximum, 2 * i + 2);
  return maximum * value;
}
bool Camera::Valid(double x, double y, double fov) const {
  const double dx = (x - params[2]) / params[0],
               dy = (y - params[3]) / params[1], radius = Radius(fov);
  return std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 &&
         x < width && y < height && dx * dx + dy * dy <= radius * radius;
}
cv::Mat Camera::Mask(double fov) const {
  cv::Mat mask(height, width, CV_8UC1);
  const double radius = Radius(fov);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const double dx = (x + .5 - params[2]) / params[0],
                   dy = (y + .5 - params[3]) / params[1];
      mask.at<uint8_t>(y, x) = dx * dx + dy * dy <= radius * radius ? 255 : 0;
    }
  return mask;
}
std::vector<cv::Point2d> Camera::Project(
    const std::vector<cv::Point3d>& xyz) const {
  std::vector<cv::Point2d> output;
  cv::fisheye::projectPoints(xyz, output, cv::Vec3d{0, 0, 0},
                             cv::Vec3d{0, 0, 0}, K(), D());
  return output;
}
std::vector<cv::Point3d> Camera::Rays(
    const std::vector<cv::Point2d>& pixels) const {
  std::vector<cv::Point2d> normalized;
  cv::fisheye::undistortPoints(
      pixels, normalized, K(), D(), cv::noArray(), cv::noArray(),
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50,
                       1e-12));
  std::vector<cv::Point3d> rays;
  for (const auto& p : normalized) {
    const auto length = std::sqrt(p.x * p.x + p.y * p.y + 1);
    rays.emplace_back(p.x / length, p.y / length, 1 / length);
  }
  return rays;
}
cv::Mat ReadImage(const fs::path& path, int mode) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot read image: " + path.u8string());
  std::vector<uint8_t> bytes(fs::file_size(path));
  stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("Truncated image: " + path.u8string());
  auto image = cv::imdecode(bytes, mode);
  if (image.empty())
    throw std::runtime_error("Cannot decode image: " + path.u8string());
  return image;
}
void WriteImage(const fs::path& path, const cv::Mat& image) {
  std::vector<uint8_t> bytes;
  if (!cv::imencode(path.extension().u8string(), image, bytes))
    throw std::runtime_error("Cannot encode image");
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write image: " + path.u8string());
}
cv::Mat BicubicResize(const cv::Mat& image, int width, int height) {
  if (image.type() != CV_8UC3)
    throw std::runtime_error("Expected an 8-bit RGB image");
  const auto horizontal = ResizeWeights(image.cols, width),
             vertical = ResizeWeights(image.rows, height);
  cv::Mat intermediate(image.rows, width, CV_8UC3),
      output(height, width, CV_8UC3);
  for (int y = 0; y < image.rows; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        for (size_t j = 0; j < horizontal[x].values.size(); ++j)
          sum +=
              int64_t(image.at<cv::Vec3b>(y, horizontal[x].start + int(j))[c]) *
              horizontal[x].values[j];
        intermediate.at<cv::Vec3b>(y, x)[c] =
            static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        for (size_t j = 0; j < vertical[y].values.size(); ++j)
          sum += int64_t(intermediate.at<cv::Vec3b>(vertical[y].start + int(j),
                                                    x)[c]) *
                 vertical[y].values[j];
        output.at<cv::Vec3b>(y, x)[c] =
            static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
      }
  return output;
}
Json PrepareScene(const fs::path& input, const fs::path& work) {
  fs::path project;
  for (const auto& entry : fs::directory_iterator(input))
    if (entry.path().extension() == ".pro") {
      if (!project.empty())
        throw std::runtime_error("Expected exactly one .pro project");
      project = entry.path();
    }
  if (project.empty()) throw std::runtime_error("No .pro project found");
  const auto pro = ReadJson(project);
  Json cameras;
  for (const auto& device : pro.at("cameraDevices").at("devices")) {
    std::string label = device.at("deviceId");
    if (label.rfind("camera_", 0) == 0) label = label.substr(7);
    if (label != "L" && label != "R") continue;
    const auto& basic = device.at("dataBasicParams");
    const auto& sensor = device.at("dataSensorParams");
    const double pixel = basic.at("pixelSize"), focal = basic.at("focalLength");
    if (pixel <= 0 || focal <= 0)
      throw std::runtime_error("Invalid physical camera calibration");
    Camera camera;
    camera.label = label;
    camera.width = basic.at("photoWidth");
    camera.height = basic.at("photoHeight");
    camera.principal_offset = {sensor.at("principalX").get<double>() / pixel,
                               sensor.at("principalY").get<double>() / pixel};
    camera.params = {focal / pixel,
                     focal / pixel,
                     camera.width / 2. + camera.principal_offset[0],
                     camera.height / 2. - camera.principal_offset[1],
                     sensor.at("k1"),
                     sensor.at("k2"),
                     sensor.at("k3"),
                     sensor.at("p3")};
    camera.Radius();
    cameras[label] = camera.ToJson();
  }
  if (!cameras.contains("L") || !cameras.contains("R"))
    throw std::runtime_error("Both L/R cameras are required");
  const auto poses_path =
      input / fs::u8path(project.stem().u8string() + "_CameraPos_C2E.txt");
  std::ifstream pose_stream(poses_path);
  if (!pose_stream) throw std::runtime_error("Missing camera positions");
  std::map<std::string, Json> poses;
  std::string line;
  while (std::getline(pose_stream, line)) {
    if (line.substr(0, 3) == "\xef\xbb\xbf") line.erase(0, 3);
    if (line.find_first_not_of(" \r\t") == std::string::npos) continue;
    std::istringstream fields(line);
    std::string name, extra;
    int raw_camera, week;
    double seconds;
    std::array<double, 3> center;
    std::array<double, 4> quaternion;
    if (!(fields >> name >> raw_camera >> week >> seconds >> center[0] >>
          center[1] >> center[2] >> quaternion[0] >> quaternion[1] >>
          quaternion[2] >> quaternion[3]) ||
        (fields >> extra))
      throw std::runtime_error("Expected 11 camera position fields");
    if (!std::isfinite(seconds) ||
        !std::all_of(center.begin(), center.end(),
                     [](double v) { return std::isfinite(v); }))
      throw std::runtime_error("Invalid input pose position");
    double norm = 0;
    for (double v : quaternion) norm += v * v;
    if (!(norm > .99 * .99 && norm < 1.01 * 1.01))
      throw std::runtime_error("Invalid input pose quaternion");
    if (!poses
             .emplace(name, Json{{"camera_id_raw", raw_camera},
                                 {"week", week},
                                 {"seconds", seconds},
                                 {"center_raw", center},
                                 {"quaternion_raw", quaternion},
                                 {"convention", "unverified vendor C2E"},
                                 {"is_ground_truth", false}})
             .second)
      throw std::runtime_error("Duplicate camera position: " + name);
  }
  Json images = Json::array(), excluded = Json::object();
  std::set<std::string> names;
  std::set<std::pair<std::string, int>> frames;
  for (const auto& entry : fs::recursive_directory_iterator(input / "Image")) {
    auto extension = entry.path().extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (extension != ".jpg" && extension != ".jpeg") continue;
    const auto name = entry.path().filename().u8string();
    int frame;
    std::string label;
    const auto [week, seconds] = TimeName(name, frame, label);
    if (!cameras.contains(label)) {
      excluded["M_or_other_camera"] =
          excluded.value("M_or_other_camera", 0) + 1;
      continue;
    }
    if (!poses.count(name)) {
      excluded["no_camera_pose"] = excluded.value("no_camera_pose", 0) + 1;
      continue;
    }
    if (!names.insert(name).second || !frames.emplace(label, frame).second)
      throw std::runtime_error("Ambiguous exposure: " + name);
    if (poses.at(name).at("week") != week)
      throw std::runtime_error("Pose/image week mismatch");
    const auto camera = Camera::FromJson(cameras.at(label));
    const auto image = ReadJpegMetadata(entry.path());
    if (image.width != camera.width || image.height != camera.height ||
        image.orientation != 1)
      throw std::runtime_error(
          "Unexpected image dimensions or EXIF orientation");
    const auto file = FileMetadata(entry.path());
    images.push_back(
        {{"name",
          fs::relative(entry.path(), input / "Image").generic_u8string()},
         {"source", fs::absolute(entry.path()).u8string()},
         {"camera", label},
         {"frame_index", frame},
         {"week", week},
         {"seconds", seconds},
         {"exposure_seconds", image.exposure},
         {"size_bytes", file.at("size_bytes")},
         {"mtime_ns", file.at("mtime_ns")},
         {"semantic_mask", nullptr},
         {"prior", poses.at(name)}});
  }
  for (const auto& [name, pose] : poses) {
    int frame;
    std::string label;
    TimeName(name, frame, label);
    if (cameras.contains(label) && !names.count(name))
      throw std::runtime_error("Pose-bound image missing: " + name);
  }
  if (images.empty()) throw std::runtime_error("No selected L/R exposures");
  std::sort(images.begin(), images.end(), [](const Json& a, const Json& b) {
    return std::tuple(a.at("seconds"), a.at("camera"), a.at("name")) <
           std::tuple(b.at("seconds"), b.at("camera"), b.at("name"));
  });
  for (size_t i = 0; i < images.size(); ++i) {
    images[i]["image_id"] = i + 1;
    images[i]["exposure_id"] = i + 1;
  }
  Json scene = {
      {"contract_version", 3},
      {"input_root", fs::absolute(input).u8string()},
      {"pose_model", "independent_exposure"},
      {"selection", "L/R with exact CameraPos filename only; M excluded"},
      {"excluded", excluded},
      {"pixel_coordinates",
       "corner coordinates; top-left pixel centre (0.5, 0.5)"},
      {"fov_degrees", 160},
      {"principal_y_axis", "up"},
      {"principal_convention_verified", false},
      {"distortion_mapping", {"k1", "k2", "k3", "p3"}},
      {"pose_values_used_for_geometry", false},
      {"semantic_masks_available", false},
      {"cameras", cameras},
      {"images", images},
      {"source_files",
       {{"project", FileMetadata(project)},
        {"camera_pose", FileMetadata(poses_path)}}}};
  fs::create_directories(work);
  const auto manifest = work / "manifest.json";
  if (fs::exists(manifest)) {
    if (ReadJson(manifest) != scene)
      throw std::runtime_error(
          "Prepared inputs changed; choose a fresh work directory");
  } else
    WriteJson(manifest, scene);
  return scene;
}

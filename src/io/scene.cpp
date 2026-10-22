#include "io/scene.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <cstring>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
#include <Eigen/Eigenvalues>
namespace geosun {
namespace {
double Cubic(double x) {
  x = std::abs(x);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return (((-.5 * x + 2.5) * x - 4) * x) + 2;
  return 0;
}
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
    if (byte() != 0xff)
      throw std::runtime_error("Invalid JPEG marker");
    int marker;
    do {
      marker = byte();
    } while (marker == 0xff);
    if (marker == 0xda || marker == 0xd9)
      break;
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7))
      continue;
    const int length = (byte() << 8) + byte() - 2;
    if (length < 0)
      throw std::runtime_error("Invalid JPEG segment");
    std::vector<uint8_t> data(size_t(length), 0);
    if (!stream.read(reinterpret_cast<char*>(data.data()), length))
      throw std::runtime_error("Truncated JPEG segment");
    if (((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) ||
         (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) &&
        length >= 5) {
      result.height = (data[1] << 8) | data[2];
      result.width = (data[3] << 8) | data[4];
    }
    if (marker == 0xe1 && length >= 14 && !std::memcmp(data.data(), "Exif\0\0", 6)) {
      const size_t base = 6;
      const bool little = data[base] == 'I' && data[base + 1] == 'I';
      if (!little && !(data[base] == 'M' && data[base + 1] == 'M'))
        throw std::runtime_error("Invalid EXIF byte order");
      auto integer = [&](size_t offset, size_t bytes) {
        if (offset + bytes > data.size() - base)
          throw std::runtime_error("Truncated EXIF field");
        uint32_t value = 0;
        for (size_t i = 0; i < bytes; ++i)
          value |= uint32_t(data[base + offset + i]) << (8 * (little ? i : bytes - i - 1));
        return value;
      };
      if (integer(2, 2) != 42)
        throw std::runtime_error("Invalid EXIF TIFF header");
      const size_t table = integer(4, 4), count = integer(table, 2);
      for (size_t i = 0; i < count; ++i) {
        const size_t entry = table + 2 + i * 12;
        const auto tag = integer(entry, 2), type = integer(entry + 2, 2), n = integer(entry + 4, 4);
        if (tag == 274 && type == 3 && n == 1)
          result.orientation = int(integer(entry + 8, 2));
        if (tag == 33434 && type == 5 && n == 1) {
          const size_t value = integer(entry + 8, 4);
          const auto denominator = integer(value + 4, 4);
          if (!denominator)
            throw std::runtime_error("Invalid exposure time");
          const double exposure = double(integer(value, 4)) / denominator;
          if (exposure)
            result.exposure = exposure;
        }
      }
    }
  }
  if (!result.width || !result.height)
    throw std::runtime_error("Missing JPEG dimensions");
  return result;
}
std::pair<int, double> TimeName(const std::string& name, int& frame, std::string& camera) {
  std::smatch parts;
  if (!std::regex_match(name, parts,
                        std::regex("(\\d+)-(\\d+)-(\\d+)_(\\d+)-([LRM])\\.JPG", std::regex::icase)))
    throw std::runtime_error("Unrecognized image filename: " + name);
  frame = std::stoi(parts[4]);
  camera = parts[5];
  std::transform(camera.begin(), camera.end(), camera.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return {std::stoi(parts[1]), std::stod(parts[2]) + std::stod(parts[3]) / 1000};
}
}  // namespace
Json ImageHeader(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  std::array<uint8_t, 29> header{};
  stream.read(reinterpret_cast<char*>(header.data()), header.size());
  if (stream.gcount() == std::streamsize(header.size()) &&
      !std::memcmp(header.data(), "\x89PNG\r\n\x1a\n", 8)) {
    if (std::memcmp(header.data() + 12, "IHDR", 4))
      throw std::runtime_error("Missing PNG image header");
    auto integer = [&](int start) {
      uint32_t v = 0;
      for (int i = 0; i < 4; ++i)
        v = (v << 8) | header[start + i];
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
cv::Mat ReadImage(const fs::path& path, int mode) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot read image: " + path.u8string());
  std::vector<uint8_t> bytes(fs::file_size(path));
  stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!stream)
    throw std::runtime_error("Truncated image: " + path.u8string());
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
std::vector<ResizeCoefficients> ResizeWeights(int input, int output) {
  const double scale = double(input) / output, filter_scale = std::max(1., scale),
               support = 2 * filter_scale;
  std::vector<ResizeCoefficients> coefficients;
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
    ResizeCoefficients row{lo, {}};
    for (double weight : raw) {
      weight = weight / sum * (1 << 22);
      row.values.push_back(int32_t(weight < 0 ? weight - .5 : weight + .5));
    }
    coefficients.push_back(std::move(row));
  }
  return coefficients;
}

Json Scene::ToJson() const {
  Json exposures = Json::array();
  for (const auto& e : images)
    exposures.push_back(
        {{"id", e.id},
         {"camera", e.camera},
         {"name", e.name},
         {"source", e.source.u8string()},
         {"frame", e.frame},
         {"week", e.week},
         {"seconds", e.seconds},
         {"pose_seconds", e.pose_seconds},
         {"navigation_raw", {e.navigation_raw.x(), e.navigation_raw.y(), e.navigation_raw.z()}},
         {"source_file", e.source_file}});
  return {{"version", 1},
          {"input", input.u8string()},
          {"project", project.u8string()},
          {"las", las.u8string()},
          {"cameras", {cameras[0].ToJson(), cameras[1].ToJson()}},
          {"images", exposures},
          {"photos", enhanced ? "enhanced" : "original"},
          {"sources", sources}};
}
Scene Scene::FromJson(const Json& v) {
  if (v.at("version") != 1)
    throw std::runtime_error("Unsupported scene version");
  Scene s;
  s.input = fs::u8path(v.at("input").get<std::string>());
  s.project = fs::u8path(v.at("project").get<std::string>());
  s.las = fs::u8path(v.at("las").get<std::string>());
  s.sources = v.at("sources");
  // Scenes written before the choice existed always preferred enhanced copies.
  const auto photos = v.value("photos", "enhanced");
  if (photos != "enhanced" && photos != "original")
    throw std::runtime_error("Unknown scene photos: " + photos);
  s.enhanced = photos == "enhanced";
  for (int i = 0; i < 2; ++i)
    s.cameras[i] = Camera::FromJson(v.at("cameras")[i]);
  for (const auto& row : v.at("images")) {
    const auto n = row.at("navigation_raw").get<std::array<double, 3>>();
    s.images.push_back({row.at("id"), row.at("camera"), row.at("name"),
                        fs::u8path(row.at("source").get<std::string>()), row.at("frame"),
                        row.at("week"), row.at("seconds"), row.at("pose_seconds"),
                        Eigen::Vector3d(n[0], n[1], n[2]), row.at("source_file")});
  }
  return s;
}
Scene ReadScene(const fs::path& input, const fs::path& chosen_las, bool enhanced_photos) {
  Scene s;
  s.input = fs::weakly_canonical(input);
  s.enhanced = enhanced_photos;
  for (const auto& row : fs::directory_iterator(input))
    if (row.path().extension() == ".pro") {
      if (!s.project.empty())
        throw std::runtime_error("Expected exactly one .pro project");
      s.project = row.path();
    }
  if (s.project.empty())
    throw std::runtime_error("Missing .pro project");
  const auto pro = ReadJson(s.project);
  std::set<std::string> labels;
  for (const auto& device : pro.at("cameraDevices").at("devices")) {
    auto label = device.at("deviceId").get<std::string>();
    if (label.rfind("camera_", 0) == 0)
      label = label.substr(7);
    if (label != "L" && label != "R")
      continue;
    if (!labels.insert(label).second)
      throw std::runtime_error("Duplicate camera calibration");
    const auto& b = device.at("dataBasicParams");
    const auto& p = device.at("dataSensorParams");
    const double pixel = b.at("pixelSize"), focal = b.at("focalLength");
    if (pixel <= 0 || focal <= 0)
      throw std::runtime_error("Invalid physical camera calibration");
    Camera c;
    c.label = label;
    c.width = b.at("photoWidth");
    c.height = b.at("photoHeight");
    c.principal_offset = {p.at("principalX").get<double>() / pixel,
                          p.at("principalY").get<double>() / pixel};
    c.params = {focal / pixel,
                focal / pixel,
                c.width / 2. + c.principal_offset[0],
                c.height / 2. + c.principal_offset[1],
                p.at("k1"),
                p.at("k2"),
                p.at("k3"),
                p.at("p3")};
    c.Radius();
    s.cameras[label == "L" ? 0 : 1] = c;
  }
  if (labels.size() != 2)
    throw std::runtime_error("Missing L/R calibration");
  const auto stem = s.project.stem().u8string();
  const auto pos = input / fs::u8path(stem + "_CameraPos_C2E.txt");
  std::ifstream stream(pos);
  if (!stream)
    throw std::runtime_error("Missing camera positions");
  struct Position {
    int week;
    double seconds;
    Eigen::Vector3d center;
  };
  std::map<std::string, Position> positions;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.substr(0, 3) == "\xef\xbb\xbf")
      line.erase(0, 3);
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    std::istringstream fields(line);
    std::string name, extra;
    int raw, week;
    double sec, x, y, z;
    std::array<double, 4> q;
    if (!(fields >> name >> raw >> week >> sec >> x >> y >> z >> q[0] >> q[1] >> q[2] >> q[3]) ||
        (fields >> extra))
      throw std::runtime_error("Expected 11 CameraPos fields");
    double norm = 0;
    for (double a : q)
      norm += a * a;
    if (!std::isfinite(sec) || !Eigen::Vector3d(x, y, z).allFinite() || norm < .99 * .99 ||
        norm > 1.01 * 1.01)
      throw std::runtime_error("Invalid CameraPos record");
    if (!positions.emplace(name, Position{week, sec, {x, y, z}}).second)
      throw std::runtime_error("Duplicate camera position: " + name);
  }
  // Pose-bound JPEGs by file name; Image/L and Image/R hold only their own lens.
  auto index = [&](const fs::path& folder, const std::string& lens) {
    std::map<std::string, fs::path> files;
    if (!fs::is_directory(folder))
      return files;
    for (const auto& row : fs::recursive_directory_iterator(folder)) {
      if (!row.is_regular_file())
        continue;
      auto ext = row.path().extension().u8string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return char(std::tolower(c)); });
      if (ext != ".jpg" && ext != ".jpeg")
        continue;
      const auto name = row.path().filename().u8string();
      int frame;
      std::string label;
      TimeName(name, frame, label);
      if ((label != "L" && label != "R") || !positions.count(name))
        continue;
      if (!lens.empty() && label != lens)
        throw std::runtime_error("Image in wrong lens folder: " + row.path().u8string());
      if (!files.emplace(name, row.path()).second)
        throw std::runtime_error("Ambiguous exposure: " + name);
    }
    return files;
  };
  const auto originals = [&] {
    auto files = index(input / "Image/L", "L");
    files.merge(index(input / "Image/R", "R"));
    return files;
  }();
  const auto enhanced = enhanced_photos ? index(input / "Image/enhanced", "")
                                        : std::map<std::string, fs::path>{};
  std::set<std::pair<std::string, int>> frames;
  for (const auto& [name, position] : positions) {
    int frame;
    std::string label;
    const auto [week, seconds] = TimeName(name, frame, label);
    if (label != "L" && label != "R")
      continue;
    // Enabled enhanced photos take precedence; the delivered name keeps the lens folder layout.
    const auto original = originals.find(name), better = enhanced.find(name);
    if (original == originals.end() && better == enhanced.end())
      throw std::runtime_error("Pose-bound image missing: " + name);
    const auto path = better != enhanced.end() ? better->second : original->second;
    const auto relative = original != originals.end()
                              ? fs::relative(original->second, input / "Image").generic_u8string()
                              : label + "/" + name;
    if (!frames.emplace(label, frame).second)
      throw std::runtime_error("Ambiguous exposure: " + name);
    if (position.week != week)
      throw std::runtime_error("Image and CameraPos week differ");
    const uint32_t cid = label == "L" ? 0 : 1;
    const auto header = ReadJpegMetadata(path);
    const auto& c = s.cameras[cid];
    if (header.width != c.width || header.height != c.height || header.orientation != 1)
      throw std::runtime_error("Unexpected JPEG size/orientation: " + path.u8string());
    s.images.push_back({0, cid, relative, fs::absolute(path), frame, week, seconds,
                        position.seconds, position.center, FileMetadata(path)});
  }
  if (s.images.size() < 3)
    throw std::runtime_error("Too few eligible L/R exposures");
  std::sort(s.images.begin(), s.images.end(), [](const Exposure& a, const Exposure& b) {
    return std::tie(a.week, a.seconds, a.camera, a.name) <
           std::tie(b.week, b.seconds, b.camera, b.name);
  });
  for (uint32_t i = 0; i < s.images.size(); ++i)
    s.images[i].id = i;
  s.las = chosen_las.empty() ? input / "ResultOut" / fs::u8path(stem + ".las") : chosen_las;
  if (!fs::is_regular_file(s.las))
    throw std::runtime_error("Missing LAS: " + s.las.u8string());
  s.las = fs::absolute(s.las);
  const auto nav = input / fs::u8path(stem + "_pos.txt"),
             enh = input / fs::u8path(stem + "_pos_ENH.txt");
  s.sources = {{"project", FileMetadata(s.project)},
               {"camera_positions", FileMetadata(pos)},
               {"navigation", FileMetadata(nav)},
               {"enh", FileMetadata(enh)},
               {"las", FileMetadata(s.las)}};
  return s;
}
namespace {
struct NavigationRow {
  int week;
  double time;
  Eigen::Vector3d point;
};
// Navigation rows and the fitted ECEF-to-LAS-frame conversion of their positions.
struct Navigation {
  std::vector<NavigationRow> rows;
  Eigen::Vector3d center;
  Similarity conversion;
};
Navigation ReadNavigation(const Scene& scene) {
  using Row = NavigationRow;
  auto read = [](const fs::path& file) {
    std::ifstream stream(file);
    if (!stream)
      throw std::runtime_error("Cannot read navigation: " + file.u8string());
    std::vector<Row> rows;
    std::string line;
    while (std::getline(stream, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos)
        continue;
      std::istringstream fields(line);
      Row r;
      double x, y, z;
      if (!(fields >> r.week >> r.time >> x >> y >> z))
        throw std::runtime_error("Invalid navigation row");
      r.point = {x, y, z};
      if (!r.point.allFinite() || !std::isfinite(r.time))
        throw std::runtime_error("Nonfinite navigation row");
      rows.push_back(r);
    }
    return rows;
  };
  const auto stem = scene.project.stem().u8string();
  const auto a = read(scene.input / fs::u8path(stem + "_pos.txt")),
             b = read(scene.input / fs::u8path(stem + "_pos_ENH.txt"));
  if (a.size() != b.size() || a.size() < 3)
    throw std::runtime_error("Navigation row identities differ");
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto& row : a)
    center += row.point;
  center /= double(a.size());
  std::vector<Eigen::Vector3d> source, target;
  double low = INFINITY, high = -INFINITY;
  Eigen::Matrix3d spread = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].week != b[i].week || a[i].week != a[0].week ||
        (i && (a[i].time <= a[i - 1].time || b[i].time <= b[i - 1].time)))
      throw std::runtime_error("Invalid navigation identity/time sequence");
    low = std::min(low, a[i].time - b[i].time);
    high = std::max(high, a[i].time - b[i].time);
    source.push_back(a[i].point - center);
    target.push_back(b[i].point - b[0].point);
    spread += source.back() * source.back().transpose();
  }
  if (high - low > .02)
    throw std::runtime_error("Navigation timestamps have inconsistent row offsets");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(spread);
  if (eig.eigenvalues()[1] < eig.eigenvalues()[2] * 1e-8)
    throw std::runtime_error("Collinear navigation cannot establish the LAS frame");
  const auto conversion = FitSimilarity(source, target);
  if (std::abs(conversion.scale - 1) > .001)
    throw std::runtime_error("Navigation coordinate conversion scale differs from unity");
  for (size_t i = 0; i < source.size(); ++i)
    if ((conversion.Apply(source[i]) - target[i]).norm() > .02)
      throw std::runtime_error("Navigation frame conversion error exceeds 2 cm");
  for (const auto& image : scene.images)
    if (image.week != a[0].week)
      throw std::runtime_error("Camera and navigation GPS week differ");
  return {a, center, conversion};
}
}  // namespace
std::vector<Eigen::Vector3d> NavigationPositions(const Scene& scene) {
  const auto navigation = ReadNavigation(scene);
  std::vector<Eigen::Vector3d> positions;
  for (const auto& image : scene.images)
    positions.push_back(navigation.conversion.Apply(image.navigation_raw - navigation.center));
  return positions;
}
std::vector<Eigen::Vector3d> NavigationVelocities(const Scene& scene) {
  constexpr double half_window = .05;
  const auto navigation = ReadNavigation(scene);
  const auto& rows = navigation.rows;
  auto at = [&](double time) {
    if (time < rows.front().time || time > rows.back().time)
      throw std::runtime_error("Exposure time outside navigation trajectory");
    const auto next = std::upper_bound(rows.begin(), rows.end(), time,
                                       [](double t, const NavigationRow& r) { return t < r.time; });
    if (next == rows.end())
      return rows.back().point;
    const auto& b = *next;
    const auto& a = *(next - 1);
    return Eigen::Vector3d(a.point + (b.point - a.point) * ((time - a.time) / (b.time - a.time)));
  };
  std::vector<Eigen::Vector3d> velocities;
  for (const auto& image : scene.images) {
    const double low = std::max(rows.front().time, image.pose_seconds - half_window),
                 high = std::min(rows.back().time, image.pose_seconds + half_window);
    if (high - low < half_window)
      throw std::runtime_error("Navigation trajectory does not bracket exposure: " + image.name);
    velocities.push_back(navigation.conversion.scale *
                         (navigation.conversion.rotation * (at(high) - at(low))) / (high - low));
  }
  return velocities;
}
}  // namespace geosun

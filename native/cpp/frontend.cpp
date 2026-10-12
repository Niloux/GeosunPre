#include "frontend.h"

#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

#include "array_io.h"
#include "models.h"
#include "scene.h"

namespace {
template <typename T>
geosunpre::Array Array(const std::vector<T>& values, std::vector<size_t> shape,
                       const std::string& dtype) {
  geosunpre::Array array{dtype, std::move(shape), {}};
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  array.bytes.assign(bytes, bytes + values.size() * sizeof(T));
  return array;
}
geosunpre::Array Array(at::Tensor tensor, std::vector<size_t> shape) {
  tensor = tensor.detach().to(at::kCPU).to(at::kFloat).contiguous();
  geosunpre::Array array{"<f4", std::move(shape), {}};
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(tensor.data_ptr<float>());
  array.bytes.assign(bytes, bytes + tensor.numel() * sizeof(float));
  return array;
}
at::Tensor ImageTensor(const cv::Mat& rgb) {
  auto tensor = torch::empty({1, 3, rgb.rows, rgb.cols}, torch::kFloat32);
  auto* data = tensor.data_ptr<float>();
  for (int y = 0; y < rgb.rows; ++y)
    for (int x = 0; x < rgb.cols; ++x)
      for (int c = 0; c < 3; ++c)
        data[c * rgb.rows * rgb.cols + y * rgb.cols + x] =
            float(double(rgb.at<cv::Vec3b>(y, x)[c]) / 255.);
  return tensor.to(torch::kCUDA);
}
struct View {
  cv::Mat source, x, y, weight;
};
struct Fusion {
  std::vector<View> views;
  cv::Mat weight, fov;
};
Fusion Geometry(const Camera& camera) {
  constexpr int width = 900, height = 1200;
  const double radius = camera.Radius(),
               cosine = std::cos(80. * 3.14159265358979323846 / 180.);
  std::vector<cv::Point2d> pixels;
  pixels.reserve(width * height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      pixels.emplace_back((x + .5) * camera.width / width,
                          (y + .5) * camera.height / height);
  const auto rays = camera.Rays(pixels);
  Fusion fusion;
  fusion.weight = cv::Mat::zeros(height, width, CV_32FC1);
  fusion.fov = camera.Mask();
  for (const auto& rotation : std::vector<cv::Vec3d>{
           {0, 0, 0}, {0, 45, 0}, {0, -45, 0}, {45, 0, 0}, {-45, 0, 0}}) {
    cv::Matx33d R;
    cv::Rodrigues(rotation * (3.14159265358979323846 / 180.), R);
    const double focal =
        512. / (2 * std::tan(50. * 3.14159265358979323846 / 180.));
    std::vector<cv::Point3d> world;
    world.reserve(512 * 512);
    for (int y = 0; y < 512; ++y)
      for (int x = 0; x < 512; ++x) {
        const auto v =
            R * cv::Vec3d((x + .5 - 256) / focal, (y + .5 - 256) / focal, 1);
        world.emplace_back(v[0], v[1], v[2]);
      }
    const auto projected = camera.Project(world);
    View view{cv::Mat(512, 512, CV_32FC2), cv::Mat(height, width, CV_32FC1),
              cv::Mat(height, width, CV_32FC1),
              cv::Mat(height, width, CV_32FC1)};
    for (size_t i = 0; i < world.size(); ++i) {
      const auto& v = world[i];
      const bool valid =
          v.z / std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z) >= cosine;
      view.source.at<cv::Vec2f>(int(i / 512), int(i % 512)) =
          valid ? cv::Vec2f(float(projected[i].x - .5),
                            float(projected[i].y - .5))
                : cv::Vec2f(-10, -10);
    }
    for (size_t i = 0; i < rays.size(); ++i) {
      const auto& ray = rays[i];
      const auto local = R.t() * cv::Vec3d(ray.x, ray.y, ray.z);
      const double x = local[0] / std::max(local[2], 1e-6) * focal + 256 - .5,
                   y = local[1] / std::max(local[2], 1e-6) * focal + 256 - .5;
      const double dx = (pixels[i].x - camera.params[2]) / camera.params[0],
                   dy = (pixels[i].y - camera.params[3]) / camera.params[1];
      const bool valid = dx * dx + dy * dy <= radius * radius &&
                         local[2] > .1 && x >= 0 && y >= 0 && x < 511 &&
                         y < 511;
      const int row = int(i / width), col = int(i % width);
      const float weight =
          valid ? float(std::pow(std::max(local[2], 0.), 8)) : 0;
      view.x.at<float>(row, col) = float(x);
      view.y.at<float>(row, col) = float(y);
      view.weight.at<float>(row, col) = weight;
      fusion.weight.at<float>(row, col) += weight;
    }
    fusion.views.push_back(std::move(view));
  }
  return fusion;
}
Json Masks(const fs::path& work, const Json& scene, CudaModels& models,
           int limit) {
  Json result = scene, records = Json::array();
  std::map<std::string, Fusion> geometry;
  const auto root = work / "sfm/masks";
  fs::create_directories(root);
  const auto mean = models.Manifest()
                        .at("segmentation")
                        .at("mean")
                        .get<std::vector<float>>(),
             stddev = models.Manifest()
                          .at("segmentation")
                          .at("std")
                          .get<std::vector<float>>();
  const size_t total = std::min<size_t>(scene.at("images").size(), limit);
  for (size_t i = 0; i < total; ++i) {
    auto record = scene.at("images").at(i);
    const auto name = record.at("name").get<std::string>(),
               label = record.at("camera").get<std::string>();
    const auto path = root / "valid" / fs::u8path(name + ".png"),
               labels = root / "labels" / fs::u8path(name + ".png"),
               stats = root / "stats" / fs::u8path(name + ".json");
    const auto source =
        FileMetadata(fs::u8path(record.at("source").get<std::string>()));
    if (source.at("size_bytes") != record.at("size_bytes") ||
        source.at("mtime_ns") != record.at("mtime_ns"))
      throw std::runtime_error("Mask image source changed");
    if (fs::exists(stats)) {
      const auto saved = ReadJson(stats);
      if (saved.at("source_file") != source ||
          saved.at("mask_file") != FileMetadata(path) ||
          saved.at("labels_file") != FileMetadata(labels))
        throw std::runtime_error("Committed masks changed: " + name);
    } else {
      const auto camera = Camera::FromJson(scene.at("cameras").at(label));
      if (!geometry.count(label)) geometry.emplace(label, Geometry(camera));
      const auto& fusion = geometry.at(label);
      auto image = ReadImage(fs::u8path(record.at("source").get<std::string>()),
                             cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      cv::Mat sums = cv::Mat::zeros(1200, 900, CV_32FC2);
      for (const auto& view : fusion.views) {
        cv::Mat sampled;
        cv::remap(image, sampled, view.source, cv::Mat(), cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
        auto input = ImageTensor(sampled);
        for (int c = 0; c < 3; ++c)
          input.select(1, c).sub_(mean[c]).div_(stddev[c]);
        auto output = models.Predict("segmentation", {input})[0]
                          .to(at::kCPU)
                          .to(at::kFloat)
                          .contiguous();
        const auto* data = output.data_ptr<float>();
        cv::Mat probability(512, 512, CV_32FC2);
        for (int y = 0; y < 512; ++y)
          for (int x = 0; x < 512; ++x)
            probability.at<cv::Vec2f>(y, x) = {data[y * 512 + x],
                                               data[512 * 512 + y * 512 + x]};
        cv::Mat projected;
        cv::remap(probability, projected, view.x, view.y, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
        for (int y = 0; y < 1200; ++y)
          for (int x = 0; x < 900; ++x)
            sums.at<cv::Vec2f>(y, x) +=
                projected.at<cv::Vec2f>(y, x) * view.weight.at<float>(y, x);
      }
      cv::Mat sky(1200, 900, CV_8UC1), person(1200, 900, CV_8UC1);
      for (int y = 0; y < 1200; ++y)
        for (int x = 0; x < 900; ++x) {
          const auto probability =
              sums.at<cv::Vec2f>(y, x) /
              std::max(fusion.weight.at<float>(y, x), 1e-10f);
          sky.at<uint8_t>(y, x) = probability[0] > .90f;
          person.at<uint8_t>(y, x) = probability[1] > .55f;
        }
      const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
      cv::erode(sky, sky, kernel);
      cv::dilate(person, person, kernel);
      cv::resize(sky, sky, {camera.width, camera.height}, 0, 0,
                 cv::INTER_NEAREST);
      cv::resize(person, person, {camera.width, camera.height}, 0, 0,
                 cv::INTER_NEAREST);
      cv::Mat semantic(camera.height, camera.width, CV_8UC1),
          reason(camera.height, camera.width, CV_8UC1);
      size_t fov_pixels = 0, retained = 0;
      for (int y = 0; y < camera.height; ++y)
        for (int x = 0; x < camera.width; ++x) {
          const uint8_t category = person.at<uint8_t>(y, x) ? 2
                                   : sky.at<uint8_t>(y, x)  ? 1
                                                            : 0;
          reason.at<uint8_t>(y, x) = category;
          semantic.at<uint8_t>(y, x) = category == 2   ? 0
                                       : category == 1 ? 128
                                                       : 255;
          if (fusion.fov.at<uint8_t>(y, x)) {
            ++fov_pixels;
            if (!category) ++retained;
          }
        }
      WriteImage(path, semantic);
      WriteImage(labels, reason);
      fs::create_directories(stats.parent_path());
      WriteJson(stats,
                {{"name", name},
                 {"source_file", source},
                 {"fov_pixels", fov_pixels},
                 {"static_valid_pixels_in_fov", retained},
                 {"static_fraction_in_fov", double(retained) / fov_pixels},
                 {"mask_file", FileMetadata(path)},
                 {"labels_file", FileMetadata(labels)}});
    }
    record["semantic_mask"] = fs::absolute(path).u8string();
    records.push_back(record);
    if (i == 0 || (i + 1) % 40 == 0 || i + 1 == total)
      std::cerr << "Native masks " << i + 1 << '/' << total << '\n';
  }
  result["images"] = records;
  result["semantic_masks_available"] = true;
  fs::create_directories(work / "sfm");
  WriteJson(work / "sfm/manifest.json", result);
  return result;
}
Json FeatureIdentity(const Json& scene, const Json& record,
                     const Json& config) {
  const auto source = fs::u8path(record.at("source").get<std::string>());
  const auto file = FileMetadata(source);
  if (file.at("size_bytes") != record.at("size_bytes") ||
      file.at("mtime_ns") != record.at("mtime_ns"))
    throw std::runtime_error("Prepared image changed: " +
                             record.at("name").get<std::string>());
  return {
      {"image_file", file},
      {"mask_file", record.at("semantic_mask").is_null()
                        ? Json(nullptr)
                        : FileMetadata(fs::u8path(
                              record.at("semantic_mask").get<std::string>()))},
      {"camera",
       scene.at("cameras").at(record.at("camera").get<std::string>())},
      {"fov", scene.at("fov_degrees")},
      {"config", config}};
}
std::vector<size_t> Select(const std::vector<cv::Point2f>& pixels,
                           const std::vector<float>& probabilities,
                           const Camera& camera) {
  std::vector<size_t> ranked(pixels.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  std::stable_sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) {
    return probabilities[a] > probabilities[b];
  });
  std::vector<int> cells;
  std::set<int> occupied;
  for (const auto& point : pixels) {
    int cell = std::min(7, int(point.y / camera.height * 8)) * 8 +
               std::min(7, int(point.x / camera.width * 8));
    cells.push_back(cell);
    occupied.insert(cell);
  }
  std::vector<bool> keep(pixels.size(), false);
  const size_t quota =
      std::max<size_t>(1, 3072 / std::max<size_t>(1, occupied.size()));
  for (int cell : occupied) {
    size_t count = 0;
    for (size_t index : ranked)
      if (cells[index] == cell && count++ < quota) keep[index] = true;
  }
  std::vector<size_t> selected;
  for (size_t index : ranked)
    if (keep[index] && selected.size() < 3072) selected.push_back(index);
  for (size_t index : ranked)
    if (!keep[index] && selected.size() < 3072) selected.push_back(index);
  return selected;
}
void Features(const fs::path& work, const Json& scene, CudaModels& models,
              int limit, bool diagnostics) {
  const auto& config = models.Manifest().at("frontend");
  Json catalog = Json::object();
  size_t hits = 0, misses = 0;
  std::vector<float> retrieval;
  const size_t total = std::min<size_t>(scene.at("images").size(), limit);
  for (size_t index = 0; index < total; ++index) {
    const auto& record = scene.at("images").at(index);
    const auto name = record.at("name").get<std::string>();
    const auto identity = FeatureIdentity(scene, record, config);
    const auto path = work / "features" / fs::u8path(name + ".npz");
    std::map<std::string, geosunpre::Array> arrays;
    if (fs::exists(path)) {
      arrays = geosunpre::ReadArrays(path);
      const auto metadata =
          Json::parse(geosunpre::ArrayString(arrays.at("metadata")));
      if (geosunpre::ArrayString(arrays.at("token")) != name ||
          metadata.at("inputs") != identity)
        throw std::runtime_error("Feature cache inputs changed: " + name);
      ++hits;
    } else {
      ++misses;
      const auto camera = Camera::FromJson(
          scene.at("cameras").at(record.at("camera").get<std::string>()));
      auto image = ReadImage(fs::u8path(record.at("source").get<std::string>()),
                             cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      const double scale = 1536. / std::max(camera.width, camera.height);
      auto detector_image =
          ImageTensor(BicubicResize(image, int(scale * camera.width) / 8 * 8,
                                    int(scale * camera.height) / 8 * 8));
      auto descriptor_image = ImageTensor(BicubicResize(image, 784, 784));
      if (diagnostics) {
        fs::create_directories(work / "diagnostics");
        const auto suffix = index ? "_" + std::to_string(index) : std::string();
        geosunpre::WriteArray(
            work / fs::u8path("diagnostics/detector_input" + suffix + ".npy"),
            Array(detector_image, {1, 3, size_t(detector_image.size(2)),
                                   size_t(detector_image.size(3))}));
        geosunpre::WriteArray(
            work / fs::u8path("diagnostics/descriptor_input" + suffix + ".npy"),
            Array(descriptor_image, {1, 3, 784, 784}));
      }
      auto detected = models.Predict("detector", {detector_image});
      auto normalized = detected[0].to(at::kCPU).to(at::kFloat).contiguous(),
           scores = detected[1].to(at::kCPU).to(at::kFloat).contiguous();
      const auto* xy = normalized.data_ptr<float>();
      const auto* probabilities = scores.data_ptr<float>();
      cv::Mat mask;
      if (!record.at("semantic_mask").is_null())
        mask =
            ReadImage(fs::u8path(record.at("semantic_mask").get<std::string>()),
                      cv::IMREAD_GRAYSCALE);
      const double radius = camera.Radius(scene.at("fov_degrees"));
      std::vector<cv::Point2f> candidates;
      std::vector<float> confidence;
      std::vector<int64_t> original;
      for (int64_t i = 0; i < normalized.size(1); ++i) {
        const float px = (xy[2 * i] + 1.f) * float(camera.width) / 2.f,
                    py = (xy[2 * i + 1] + 1.f) * float(camera.height) / 2.f;
        const double dx = (px - camera.params[2]) / camera.params[0],
                     dy = (py - camera.params[3]) / camera.params[1];
        if (!(std::isfinite(px) && std::isfinite(py) && px >= 0 && py >= 0 &&
              px < camera.width && py < camera.height &&
              dx * dx + dy * dy <= radius * radius))
          continue;
        if (!mask.empty() &&
            mask.at<uint8_t>(int(std::floor(py)), int(std::floor(px))) != 255)
          continue;
        candidates.emplace_back(px, py);
        confidence.push_back(probabilities[i]);
        original.push_back(i);
      }
      const auto selected = Select(candidates, confidence, camera);
      if (selected.size() < 32)
        throw std::runtime_error("Fewer than 32 valid features: " + name);
      std::vector<int64_t> indices;
      std::vector<float> pixels, chosen_scores;
      for (size_t i : selected) {
        indices.push_back(original[i]);
        pixels.insert(pixels.end(), {candidates[i].x, candidates[i].y});
        chosen_scores.push_back(confidence[i]);
      }
      auto selected_indices =
          torch::from_blob(indices.data(), {int64_t(indices.size())},
                           torch::kInt64)
              .to(torch::kCUDA);
      auto keypoints = detected[0].index_select(1, selected_indices);
      auto descriptions =
          models.Predict("descriptor", {descriptor_image, keypoints})[0];
      Json metadata = {{"inputs", identity},
                       {"token", name},
                       {"name", name},
                       {"detected", normalized.size(1)},
                       {"valid_before_quota", candidates.size()},
                       {"selected", indices.size()},
                       {"descriptor_resolution", {784, 784}},
                       {"pixel_coordinates", scene.at("pixel_coordinates")}};
      arrays = {{"keypoints", Array(pixels, {indices.size(), 2}, "<f4")},
                {"normalized_keypoints", Array(keypoints, {indices.size(), 2})},
                {"descriptors", Array(descriptions, {indices.size(), 256})},
                {"scores", Array(chosen_scores, {indices.size()}, "<f4")},
                {"metadata", geosunpre::StringArray(metadata.dump())},
                {"token", geosunpre::StringArray(name)}};
      fs::create_directories(path.parent_path());
      auto temporary = path;
      temporary += ".pending";
      geosunpre::WriteArrays(temporary, arrays);
      fs::rename(temporary, path);
    }
    const auto n = arrays.at("keypoints").shape.at(0);
    const auto* description =
        reinterpret_cast<const float*>(arrays.at("descriptors").bytes.data());
    std::vector<double> mean(256, 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < 256; ++j) mean[j] += description[256 * i + j];
    double norm = 0;
    for (double& v : mean) {
      v /= n;
      norm += v * v;
    }
    norm = std::max(1e-12, std::sqrt(norm));
    for (double v : mean) retrieval.push_back(float(v / norm));
    catalog[name] = {
        {"token", name}, {"path", path.u8string()}, {"keypoints", n}};
    if (index == 0 || (index + 1) % 40 == 0 || index + 1 == total)
      std::cerr << "Native features " << index + 1 << '/' << total << '\n';
  }
  fs::create_directories(work / "sfm");
  WriteJson(work / "sfm/feature_catalog.json",
            {{"version", 1},
             {"config", config},
             {"features", catalog},
             {"cache", {{"feature_hits", hits}, {"feature_misses", misses}}},
             {"runtime", "C++/CUDA"}});
  geosunpre::WriteArray(work / "sfm/retrieval.npy",
                        Array(retrieval, {total, 256}, "<f4"));
}
struct TensorCache {
  struct Entry {
    at::Tensor keypoints, descriptors;
    size_t bytes;
    std::list<std::string>::iterator position;
  };
  std::map<std::string, Entry> entries;
  std::list<std::string> order;
  size_t bytes = 0, limit = 2ULL * 1024 * 1024 * 1024;
  bool initialized = false;
  void Initialize() {
    size_t free, total;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
      throw std::runtime_error("Cannot query CUDA memory");
    limit = std::min(limit, total / 8);
    if (const char* setting = std::getenv("GEOSUNPRE_GPU_FEATURE_CACHE_MIB")) {
      const std::string value(setting);
      size_t used = 0;
      if (value.empty() || value[0] == '-')
        throw std::runtime_error(
            "GPU cache size must be a nonnegative integer");
      const auto mib = std::stoull(value, &used);
      if (used != value.size() || mib > SIZE_MAX / (1024 * 1024))
        throw std::runtime_error("Invalid GPU feature cache size");
      limit = std::min(limit, size_t(mib) * 1024 * 1024);
    }
    initialized = true;
  }
  std::pair<at::Tensor, at::Tensor> Get(const fs::path& path,
                                        const std::string& name) {
    if (!initialized) Initialize();
    if (entries.count(name)) {
      auto& item = entries.at(name);
      order.splice(order.end(), order, item.position);
      return {item.keypoints, item.descriptors};
    }
    auto arrays =
        geosunpre::ReadArrays(path, {"normalized_keypoints", "descriptors"});
    auto tensor = [&](const char* field) {
      auto& a = arrays.at(field);
      return torch::from_blob(
                 a.bytes.data(),
                 {1, int64_t(a.shape.at(0)), int64_t(a.shape.at(1))},
                 torch::kFloat32)
          .to(torch::kCUDA);
    };
    auto keypoints = tensor("normalized_keypoints"),
         descriptors = tensor("descriptors");
    const size_t size = keypoints.nbytes() + descriptors.nbytes();
    while (!order.empty() && bytes + size > limit) {
      const auto old = order.front();
      bytes -= entries.at(old).bytes;
      entries.erase(old);
      order.pop_front();
    }
    if (size <= limit) {
      order.push_back(name);
      entries.emplace(
          name, Entry{keypoints, descriptors, size, std::prev(order.end())});
      bytes += size;
    }
    return {keypoints, descriptors};
  }
};
void Matches(const fs::path& work, const Json& scene, CudaModels& models) {
  using Key = std::pair<int, int>;
  const auto catalog = ReadJson(work / "sfm/feature_catalog.json");
  const int count = int(scene.at("images").size());
  const auto retrieval = geosunpre::ReadArray(work / "sfm/retrieval.npy");
  if (retrieval.shape != std::vector<size_t>{size_t(count), 256} ||
      catalog.at("features").size() != size_t(count))
    throw std::runtime_error("Incomplete native feature catalog");
  const auto* vectors = reinterpret_cast<const float*>(retrieval.bytes.data());
  std::vector<std::string> names;
  std::vector<double> times;
  std::map<std::string, std::vector<int>> sequences;
  for (int i = 0; i < count; ++i) {
    const auto& row = scene.at("images").at(i);
    names.push_back(row.at("name"));
    times.push_back(row.at("week").get<double>() * 604800 +
                    row.at("seconds").get<double>());
    sequences[row.at("camera")].push_back(i);
  }
  std::map<Key, std::set<std::string>> candidates;
  auto add = [&](int a, int b, const std::string& reason) {
    if (a != b) candidates[{std::min(a, b), std::max(a, b)}].insert(reason);
  };
  for (auto& [camera, ids] : sequences) {
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
      return std::pair(times[a], names[a]) < std::pair(times[b], names[b]);
    });
    for (size_t position = 0; position < ids.size(); ++position)
      for (size_t j = position + 1; j < std::min(position + 6, ids.size()); ++j)
        add(ids[position], ids[j], "temporal");
  }
  auto similarity = [&](int a, int b) {
    float score = 0;
    for (int j = 0; j < 256; ++j)
      score += vectors[256 * a + j] * vectors[256 * b + j];
    return score;
  };
  auto top = [&](int a, std::vector<int> ids, size_t number) {
    std::sort(ids.begin(), ids.end(), [&](int b, int c) {
      const float x = similarity(a, b), y = similarity(a, c);
      return x == y ? b < c : x > y;
    });
    if (ids.size() > number) ids.resize(number);
    return ids;
  };
  for (int a = 0; a < count; ++a)
    for (const auto& [camera, ids] : sequences) {
      const size_t center = std::lower_bound(ids.begin(), ids.end(), times[a],
                                             [&](int b, double time) {
                                               return times[b] < time;
                                             }) -
                            ids.begin();
      if (camera != scene.at("images").at(a).at("camera"))
        for (size_t j = center > 2 ? center - 2 : 0;
             j < std::min(ids.size(), center + 3); ++j)
          add(a, ids[j], "nearby_exposure");
      std::vector<int> eligible;
      std::map<int, std::vector<int>> buckets;
      for (size_t j = 0; j < ids.size(); ++j) {
        const auto gap = std::abs(int64_t(j) - int64_t(center));
        if (gap > 5) {
          eligible.push_back(ids[j]);
          buckets[int(std::floor(std::log2(double(gap))))].push_back(ids[j]);
        }
      }
      for (int b : top(a, eligible, 2)) add(a, b, "appearance_retrieval");
      for (const auto& [bucket, values] : buckets)
        for (int b : top(a, values, 1)) add(a, b, "multiscale_retrieval");
    }
  const auto initial_count = candidates.size();
  TensorCache tensors;
  std::map<Key, Json> entries;
  std::set<Key> supported;
  std::vector<std::set<int>> tried(count);
  size_t hits = 0, misses = 0;
  auto run = [&](const std::vector<Key>& pairs) {
    for (const auto& [a, b] : pairs) {
      const auto token = fs::u8path(names[a]).filename().u8string() + "__" +
                         fs::u8path(names[b]).filename().u8string();
      const auto path = work / "matches" / fs::u8path(token + ".npz");
      const auto file_a = fs::u8path(catalog.at("features")
                                         .at(names[a])
                                         .at("path")
                                         .get<std::string>()),
                 file_b = fs::u8path(catalog.at("features")
                                         .at(names[b])
                                         .at("path")
                                         .get<std::string>());
      Json identity = {{"a", names[a]},
                       {"b", names[b]},
                       {"config", models.Manifest().at("frontend")},
                       {"a_file", FileMetadata(file_a)},
                       {"b_file", FileMetadata(file_b)},
                       {"attention_backend", "efficient"}};
      std::map<std::string, geosunpre::Array> arrays;
      if (fs::exists(path)) {
        arrays = geosunpre::ReadArrays(path);
        if (geosunpre::ArrayString(arrays.at("token")) != token ||
            Json::parse(geosunpre::ArrayString(arrays.at("identity"))) !=
                identity)
          throw std::runtime_error("Match cache changed: " + token);
        ++hits;
      } else {
        ++misses;
        const auto [ka, da] = tensors.Get(file_a, names[a]);
        const auto [kb, db] = tensors.Get(file_b, names[b]);
        const auto output = models.Predict("matcher", {ka, kb, da, db});
        const auto matched = output[0].to(at::kCPU).contiguous(),
                   confidence =
                       output[1].to(at::kCPU).to(at::kFloat).contiguous();
        const auto* indices = matched.data_ptr<int64_t>();
        const auto* scores = confidence.data_ptr<float>();
        std::vector<uint32_t> matches;
        std::vector<float> values;
        for (int64_t i = 0; i < matched.numel(); ++i)
          if (indices[i] >= 0) {
            matches.push_back(uint32_t(i));
            matches.push_back(uint32_t(indices[i]));
            values.push_back(scores[i]);
          }
        arrays = {{"matches", Array(matches, {values.size(), 2}, "<u4")},
                  {"scores", Array(values, {values.size()}, "<f4")},
                  {"token", geosunpre::StringArray(token)},
                  {"identity", geosunpre::StringArray(identity.dump())}};
        fs::create_directories(path.parent_path());
        auto pending = path;
        pending += ".pending";
        geosunpre::WriteArrays(pending, arrays, true);
        fs::rename(pending, path);
      }
      const auto* scores =
          reinterpret_cast<const float*>(arrays.at("scores").bytes.data());
      size_t support = 0;
      for (size_t j = 0; j < arrays.at("scores").Count(); ++j)
        if (scores[j] >= .5f) ++support;
      entries[{a, b}] = {{"a", names[a]},
                         {"b", names[b]},
                         {"reasons", candidates.at({a, b})},
                         {"token", token},
                         {"path", path.u8string()},
                         {"raw_support", support}};
      tried[a].insert(b);
      tried[b].insert(a);
      if (support >= 30) supported.emplace(a, b);
      if (entries.size() == 1 || entries.size() % 200 == 0)
        std::cerr << "Native matches " << entries.size() << '/'
                  << candidates.size() << '\n';
    }
  };
  auto graph = [&](std::vector<int>& labels, std::vector<int>& degrees) {
    labels.resize(count);
    std::iota(labels.begin(), labels.end(), 0);
    degrees.assign(count, 0);
    auto root = [&](int i) {
      while (labels[i] != i) {
        labels[i] = labels[labels[i]];
        i = labels[i];
      }
      return i;
    };
    for (const auto& [a, b] : supported) {
      const int x = root(a), y = root(b);
      labels[std::max(x, y)] = std::min(x, y);
      ++degrees[a];
      ++degrees[b];
    }
    std::map<int, int> sizes;
    int weak = 0;
    for (int i = 0; i < count; ++i) {
      labels[i] = root(i);
      ++sizes[labels[i]];
      if (degrees[i] < 3) ++weak;
    }
    std::vector<int> populations;
    for (const auto& [root, size] : sizes) populations.push_back(size);
    std::sort(populations.rbegin(), populations.rend());
    return Json{{"components", sizes.size()},
                {"component_sizes", populations},
                {"images_below_three_supported_pairs", weak},
                {"supported_pairs", supported.size()}};
  };
  std::vector<Key> initial;
  for (const auto& [key, reasons] : candidates) initial.push_back(key);
  run(initial);
  std::vector<int> labels, degree;
  const auto before = graph(labels, degree);
  auto after = before;
  Json rounds = Json::array();
  for (int round = 0; round < 2; ++round) {
    if (after.at("components") == 1 &&
        after.at("images_below_three_supported_pairs") == 0)
      break;
    std::set<Key> repairs;
    std::map<int, std::map<Key, float>> bridges;
    for (int a = 0; a < count; ++a) {
      std::vector<int> available, external;
      for (int b = 0; b < count; ++b)
        if (b != a && !tried[a].count(b)) {
          available.push_back(b);
          if (labels[a] != labels[b]) external.push_back(b);
        }
      if (degree[a] < 3)
        for (int b : top(a, available, 4))
          repairs.emplace(std::min(a, b), std::max(a, b));
      for (int b : top(a, external, 2))
        bridges[labels[a]][{std::min(a, b), std::max(a, b)}] = similarity(a, b);
    }
    for (const auto& [label, choices] : bridges) {
      std::vector<std::pair<float, Key>> ordered;
      for (const auto& [key, score] : choices)
        ordered.emplace_back(-score, key);
      std::sort(ordered.begin(), ordered.end());
      for (size_t i = 0; i < std::min<size_t>(2, ordered.size()); ++i)
        repairs.insert(ordered[i].second);
    }
    if (repairs.empty()) break;
    for (const auto& [a, b] : repairs) add(a, b, "match_graph_repair");
    run({repairs.begin(), repairs.end()});
    after = graph(labels, degree);
    auto report = after;
    report["additional_pairs"] = repairs.size();
    rounds.push_back(report);
  }
  Json pairs = Json::array();
  for (const auto& [key, entry] : entries) pairs.push_back(entry);
  WriteJson(work / "sfm/pair_protocol.json",
            {{"version", 2},
             {"initial_pairs", initial_count},
             {"final_pairs", entries.size()},
             {"graph_before_repair", before},
             {"graph_after_repair", after},
             {"repair_rounds", rounds},
             {"selection_uses_input_pose_or_las", false}});
  WriteJson(work / "sfm/frontend.json",
            {{"version", 1},
             {"scene_manifest_file", FileMetadata(work / "sfm/manifest.json")},
             {"config", models.Manifest().at("frontend")},
             {"features", catalog.at("features")},
             {"pairs", pairs},
             {"cache", {{"match_hits", hits}, {"match_misses", misses}}}});
}
}  // namespace
void RunFrontend(const Arguments& args) {
  args.Validate({"--manifest", "--work", "--models", "--stage", "--limit",
                 "--diagnostics"});
  const auto work = args.Path("--work"), manifest = args.Path("--manifest");
  const auto scene = ReadJson(manifest);
  CudaModels models(args.Path("--models"));
  c10::InferenceMode guard;
  const auto stage = args.Get("--stage", "features");
  const int limit = args.Integer("--limit", INT_MAX);
  if (stage == "masks")
    Masks(work, scene, models, limit);
  else if (stage == "features") {
    fs::create_directories(work / "sfm");
    if (!fs::exists(work / "sfm/manifest.json"))
      WriteJson(work / "sfm/manifest.json", scene);
    Features(work, scene, models, limit, args.Get("--diagnostics") == "true");
  } else if (stage == "matches")
    Matches(work, scene, models);
  else
    throw std::runtime_error("Unsupported frontend stage");
  WriteJson(
      work / "frontend_runtime.json",
      {{"runtime", "C++/CUDA"}, {"stage", stage}, {"resources", Resources()}});
}

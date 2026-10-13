#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "frontend/preprocessing.h"
#include "io/cache.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <list>
#include <numeric>
#include <set>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <functional>

namespace geosun {
fs::path MaskPath(const fs::path& work, uint32_t id) {
  return work / "cache/masks" / (std::to_string(id) + ".png");
}
namespace {
struct Fusion {
  std::vector<MaskView> views;
  cv::Mat weight;
};
Fusion VirtualViews(const Camera& camera) {
  constexpr int w = 900, h = 1200;
  constexpr double Pi = 3.14159265358979323846;
  const double focal = 512 / (2 * std::tan(50 * Pi / 180)), cosine = std::cos(80 * Pi / 180);
  Fusion out;
  out.weight = cv::Mat::zeros(h, w, CV_32FC1);
  std::vector<Eigen::Vector3d> rays;
  rays.reserve(w * h);
  const double valid_radius = camera.Radius();
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const Eigen::Vector2d pixel((x + .5) * camera.width / w, (y + .5) * camera.height / h);
      const double dx = (pixel.x() - camera.params[2]) / camera.params[0],
                   dy = (pixel.y() - camera.params[3]) / camera.params[1];
      if (dx * dx + dy * dy <= valid_radius * valid_radius)
        rays.push_back(camera.Ray(pixel));
      else
        rays.push_back(Eigen::Vector3d::Zero());
    }
  for (const Eigen::Vector3d& angle :
       std::vector<Eigen::Vector3d>{{0, 0, 0}, {0, 45, 0}, {0, -45, 0}, {45, 0, 0}, {-45, 0, 0}}) {
    const double length = angle.norm() * Pi / 180;
    const Eigen::Matrix3d R =
        length == 0 ? Eigen::Matrix3d::Identity().eval()
                    : Eigen::AngleAxisd(length, angle.normalized()).toRotationMatrix();
    MaskView v{cv::Mat(512, 512, CV_32FC2), cv::Mat(h, w, CV_32FC1), cv::Mat(h, w, CV_32FC1),
               cv::Mat(h, w, CV_32FC1)};
    for (int y = 0; y < 512; ++y)
      for (int x = 0; x < 512; ++x) {
        const Eigen::Vector3d ray =
            R * Eigen::Vector3d((x + .5 - 256) / focal, (y + .5 - 256) / focal, 1);
        const auto xy = camera.Project(ray);
        v.source.at<cv::Vec2f>(y, x) = ray.normalized().z() >= cosine
                                           ? cv::Vec2f(float(xy.x() - .5), float(xy.y() - .5))
                                           : cv::Vec2f(-10, -10);
      }
    const double radius = camera.Radius();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const Eigen::Vector3d local = R.transpose() * rays[y * w + x];
        const double xx = local.x() / std::max(local.z(), 1e-6) * focal + 255.5,
                     yy = local.y() / std::max(local.z(), 1e-6) * focal + 255.5;
        const double dx = ((x + .5) * camera.width / w - camera.params[2]) / camera.params[0],
                     dy = ((y + .5) * camera.height / h - camera.params[3]) / camera.params[1];
        const bool valid = dx * dx + dy * dy <= radius * radius && local.z() > .1 && xx >= 0 &&
                           yy >= 0 && xx < 511 && yy < 511;
        const float weight = valid ? float(std::pow(std::max(local.z(), 0.), 8)) : 0;
        v.x.at<float>(y, x) = float(xx);
        v.y.at<float>(y, x) = float(yy);
        v.weight.at<float>(y, x) = weight;
        out.weight.at<float>(y, x) += weight;
      }
    out.views.push_back(std::move(v));
  }
  return out;
}
struct FrontendTiming {
  size_t computed = 0, reused = 0;
  double decode = 0, preprocess = 0, wait = 0, gpu_pipeline = 0, postprocess = 0, write = 0;
  Json Report(Clock::time_point started, Models* models = nullptr) const {
    Json report = {{"computed", computed},
                   {"reused", reused},
                   {"seconds", Seconds(started)},
                   {"decode_seconds", decode},
                   {"cpu_preprocess_seconds", preprocess},
                   {"input_wait_seconds", wait},
                   {"gpu_pipeline_host_seconds", gpu_pipeline},
                   {"cpu_postprocess_seconds", postprocess},
                   {"write_seconds", write}};
    if (models)
      report["inference"] = models->Profile();
    return report;
  }
};
struct PreparedImage {
  uint32_t id;
  cv::Mat rgb, mask;
  std::vector<cv::Mat> views;
  double decode = 0, preprocess = 0;
};
class ImageReader {
  const std::vector<Exposure>& images;
  const fs::path work;
  const std::array<Fusion, 2>* fusion;
  size_t next = 0;
  std::deque<std::future<PreparedImage>> ready;
  void Fill() {
    while (ready.size() < 2 && next < images.size()) {
      const auto record = images[next++];
      ready.push_back(std::async(std::launch::async, [this, record] {
        PreparedImage image;
        image.id = record.id;
        auto started = Clock::now();
        image.rgb = ReadImage(record.source, cv::IMREAD_COLOR);
        image.decode = Seconds(started);
        started = Clock::now();
        cv::cvtColor(image.rgb, image.rgb, cv::COLOR_BGR2RGB);
        if (fusion) {
          for (const auto& view : (*fusion)[record.camera].views) {
            cv::Mat sampled;
            cv::remap(image.rgb, sampled, view.source, cv::Mat(), cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT);
            image.views.push_back(std::move(sampled));
          }
          image.rgb.release();
        } else
          image.mask = ReadImage(MaskPath(work, record.id), cv::IMREAD_GRAYSCALE);
        image.preprocess = Seconds(started);
        return image;
      }));
    }
  }

 public:
  ImageReader(const std::vector<Exposure>& records, fs::path path,
              const std::array<Fusion, 2>* maps = nullptr)
      : images(records), work(std::move(path)), fusion(maps) {
    Fill();
  }
  PreparedImage Next() {
    auto image = ready.front().get();
    ready.pop_front();
    Fill();
    return image;
  }
};
class AsyncWriter {
  std::deque<std::future<double>> jobs;

 public:
  double seconds = 0;
  void Pop() {
    seconds += jobs.front().get();
    jobs.pop_front();
  }
  template <class Function>
  void Submit(Function write) {
    if (jobs.size() == 2)
      Pop();
    jobs.push_back(std::async(std::launch::async, [write = std::move(write)] {
      const auto started = Clock::now();
      write();
      return Seconds(started);
    }));
  }
  void Finish() {
    while (!jobs.empty())
      Pop();
  }
};
std::vector<size_t> Select(const std::vector<std::array<float, 3>>& points, const Camera& camera) {
  std::vector<size_t> order(points.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return points[a][2] > points[b][2]; });
  std::array<std::vector<size_t>, 64> cells;
  for (size_t i : order) {
    const int x = std::clamp(int(points[i][0] / camera.width * 8), 0, 7),
              y = std::clamp(int(points[i][1] / camera.height * 8), 0, 7);
    cells[y * 8 + x].push_back(i);
  }
  size_t occupied = 0;
  for (const auto& c : cells)
    if (!c.empty())
      ++occupied;
  if (!occupied)
    return {};
  const size_t quota = std::max<size_t>(1, 3072 / occupied);
  std::vector<uint8_t> use(points.size());
  for (const auto& cell : cells)
    for (size_t j = 0; j < std::min(quota, cell.size()); ++j)
      use[cell[j]] = 1;
  std::vector<size_t> result;
  for (size_t i : order)
    if (use[i] && result.size() < 3072)
      result.push_back(i);
  for (size_t i : order)
    if (!use[i] && result.size() < 3072)
      result.push_back(i);
  return result;
}
class FeatureCache {
  const fs::path work;
  std::list<uint32_t> order;
  struct Entry {
    FeatureSet value;
    Tensor keypoints, descriptors, mask;
    std::list<uint32_t>::iterator position;
  };
  std::map<uint32_t, Entry> entries;
  struct Loaded {
    FeatureSet value;
    double seconds;
  };
  std::map<uint32_t, std::future<Loaded>> pending;

 public:
  explicit FeatureCache(fs::path path) : work(std::move(path)) {}
  double read_seconds = 0, wait_seconds = 0, upload_seconds = 0;
  void Prefetch(uint32_t id) {
    if (entries.count(id) || pending.count(id) || pending.size() >= 4)
      return;
    const auto path = FeaturePath(work, id);
    pending.emplace(id, std::async(std::launch::async, [path] {
                      const auto started = Clock::now();
                      auto value = ReadFeatures(path);
                      return Loaded{std::move(value), Seconds(started)};
                    }));
  }
  const Entry& Get(uint32_t id) {
    auto it = entries.find(id);
    if (it != entries.end()) {
      order.splice(order.begin(), order, it->second.position);
      return it->second;
    }
    if (entries.size() >= 48) {
      entries.erase(order.back());
      order.pop_back();
    }
    order.push_front(id);
    Entry entry;
    const auto waiting = Clock::now();
    auto prefetched = pending.find(id);
    if (prefetched != pending.end()) {
      auto loaded = prefetched->second.get();
      read_seconds += loaded.seconds;
      entry.value = std::move(loaded.value);
      pending.erase(prefetched);
    } else {
      entry.value = ReadFeatures(FeaturePath(work, id));
      read_seconds += Seconds(waiting);
    }
    wait_seconds += Seconds(waiting);
    const auto uploading = Clock::now();
    const auto n = entry.value.pixels.size();
    entry.keypoints = Tensor::Upload({1, 3072, 2}, entry.value.normalized[0].data(), n * 2);
    entry.descriptors = Tensor::Upload({1, 3072, 256}, entry.value.descriptors.data(), n * 256);
    const std::vector<float> mask(n, 1.f);
    entry.mask = Tensor::Upload({1, 3072}, mask.data(), mask.size());
    upload_seconds += Seconds(uploading);
    entry.position = order.begin();
    return entries.emplace(id, std::move(entry)).first->second;
  }
};
}  // namespace
Json CheckModels(const fs::path& root, const fs::path& image) {
  auto rgb = ReadImage(image, cv::IMREAD_COLOR);
  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  const auto manifest = ReadJson(root / "manifest.json");
  const auto size = manifest.at("detector_size").get<std::array<int, 2>>();
  auto check = [](const Tensor& tensor, const std::vector<int64_t>& shape) {
    if (tensor.shape != shape)
      throw std::runtime_error("Model verification: unexpected output shape");
    const auto* values = tensor.Floats();
    for (size_t i = 0; i < tensor.Count(); ++i)
      if (!std::isfinite(values[i]))
        throw std::runtime_error("Model verification: nonfinite output");
  };
  Json report = {{"passed", true}, {"models", root.u8string()}, {"gpu", GpuInfo()}};
  std::vector<float> keypoints;
  Tensor descriptors;
  {
    Models models(root);
    ImagePreprocessor pixels(models.Stream());
    pixels.Upload(rgb);
    const auto logits = models.Predict("detector", {pixels.Normalize(size[0], size[1])})[0];
    check(logits, {1, 1, size[1], size[0]});
    for (const auto& point : DetectKeypoints(logits, size[0], size[1], 3072, models.Stream())) {
      keypoints.push_back(point[0]);
      keypoints.push_back(point[1]);
    }
    report["inference"]["detector"] = models.Profile().at("detector");
  }
  const int64_t count = int64_t(keypoints.size() / 2);
  if (count < 32)
    throw std::runtime_error("Model verification image has fewer than 32 features");
  {
    Models models(root);
    ImagePreprocessor pixels(models.Stream());
    pixels.Upload(rgb);
    descriptors = models.Predict(
        "descriptor", {pixels.Normalize(784, 784),
                       Tensor::Float({1, count, 2}, keypoints.data(), keypoints.size())})[0];
    check(descriptors, {1, count, 256});
    if (std::all_of(descriptors.Floats(), descriptors.Floats() + descriptors.Count(),
                    [](float value) { return value == 0; }))
      throw std::runtime_error("Model verification: empty descriptors");
    report["inference"]["descriptor"] = models.Profile().at("descriptor");
  }
  {
    Models models(root);
    const auto k = Tensor::Upload({1, 3072, 2}, keypoints.data(), keypoints.size());
    const auto d = Tensor::Upload({1, 3072, 256}, descriptors.Floats(), descriptors.Count());
    const std::vector<float> valid(size_t(count), 1.f);
    const auto mask = Tensor::Upload({1, 3072}, valid.data(), valid.size());
    const auto matched = models.Predict("matcher", {k, k, d, d, mask, mask});
    if (matched.size() != 2 || matched[0].type != 2 ||
        matched[0].shape != std::vector<int64_t>{1, 3072})
      throw std::runtime_error("Model verification: invalid matcher output");
    check(matched[1], {1, 3072});
    size_t identities = 0;
    for (size_t i = 0; i < 3072; ++i) {
      const auto index = matched[0].Index(i);
      const float confidence = matched[1].Floats()[i];
      if (index < -1 || (index >= 0 && (i >= size_t(count) || index != int64_t(i))) ||
          confidence < 0 || confidence > 1.00001f)
        throw std::runtime_error("Model verification: invalid identity match");
      identities += index >= 0;
    }
    if (!identities)
      throw std::runtime_error("Model verification: no identity matches");
    report["identity_matches"] = identities;
    report["inference"]["matcher"] = models.Profile().at("matcher");
  }
  {
    Models models(root);
    ImagePreprocessor pixels(models.Stream());
    pixels.Upload(rgb);
    const auto mean = manifest.at("segmentation_mean").get<std::array<float, 3>>();
    const auto deviation = manifest.at("segmentation_std").get<std::array<float, 3>>();
    const auto probabilities =
        models.Predict("segmentation", {pixels.Normalize(512, 512, mean, deviation)})[0];
    check(probabilities, {1, 2, 512, 512});
    for (size_t i = 0; i < 512 * 512; ++i) {
      const auto* values = probabilities.Floats();
      if (values[i] < 0 || values[i + 512 * 512] < 0 ||
          values[i] + values[i + 512 * 512] > 1.00001f)
        throw std::runtime_error("Model verification: invalid semantic probability");
    }
    report["inference"]["segmentation"] = models.Profile().at("segmentation");
  }
  return report;
}
Json MakeMasks(const Scene& scene, const fs::path& work, const fs::path& root) {
  const auto started = Clock::now();
  FrontendTiming timing;
  std::vector<Exposure> pending;
  for (const auto& record : scene.images) {
    const auto path = MaskPath(work, record.id);
    const auto& camera = scene.cameras[record.camera];
    if (fs::exists(path)) {
      const auto h = ImageHeader(path);
      if (h.at("width") != camera.width || h.at("height") != camera.height || h.at("mode") != "L")
        throw std::runtime_error("Invalid mask cache");
      ++timing.reused;
    } else
      pending.push_back(record);
  }
  if (pending.empty())
    return timing.Report(started);
  Models models(root);
  ImagePreprocessor pixels(models.Stream());
  const std::array<Fusion, 2> fusion{VirtualViews(scene.cameras[0]),
                                     VirtualViews(scene.cameras[1])};
  std::array<std::unique_ptr<MaskFusion>, 2> masks;
  for (int c = 0; c < 2; ++c)
    masks[c] = std::make_unique<MaskFusion>(fusion[c].views, fusion[c].weight,
                                            scene.cameras[c].Mask(), models.Stream());
  const auto mean = models.Manifest().at("segmentation_mean").get<std::array<float, 3>>(),
             deviation = models.Manifest().at("segmentation_std").get<std::array<float, 3>>();
  ImageReader reader(pending, work, &fusion);
  AsyncWriter writer;
  for (const auto& record : pending) {
    auto phase = Clock::now();
    auto image = reader.Next();
    timing.wait += Seconds(phase);
    timing.decode += image.decode;
    timing.preprocess += image.preprocess;
    auto& fusion_gpu = *masks[record.camera];
    phase = Clock::now();
    fusion_gpu.Begin();
    for (size_t view = 0; view < image.views.size(); ++view) {
      pixels.Upload(image.views[view]);
      const auto output =
          models.Predict("segmentation", {pixels.Normalize(512, 512, mean, deviation)}, true)[0];
      fusion_gpu.Add(output, view);
    }
    auto mask = fusion_gpu.Finish();
    timing.gpu_pipeline += Seconds(phase);
    const auto path = MaskPath(work, record.id);
    writer.Submit([path, mask = std::move(mask)] {
      const auto temporary = path.parent_path() / (path.stem().u8string() + ".writing.png");
      WriteImage(temporary, mask);
      PublishFile(temporary, path);
    });
    ++timing.computed;
    if (record.id % 16 == 0)
      Log("Masks " + std::to_string(record.id + 1) + "/" + std::to_string(scene.images.size()));
  }
  writer.Finish();
  timing.write = writer.seconds;
  return timing.Report(started, &models);
}
Json MakeFeatures(const Scene& scene, const fs::path& work, const fs::path& root) {
  const auto started = Clock::now();
  FrontendTiming timing;
  std::vector<Exposure> pending;
  for (const auto& record : scene.images) {
    const auto path = FeaturePath(work, record.id);
    if (fs::exists(path)) {
      const auto f = ReadFeatures(path);
      if (f.image != record.id)
        throw std::runtime_error("Feature identity differs");
      ++timing.reused;
    } else
      pending.push_back(record);
  }
  if (pending.empty())
    return timing.Report(started);
  Models models(root);
  ImagePreprocessor pixels(models.Stream());
  const auto expected = models.Manifest().at("detector_size").get<std::array<int, 2>>();
  ImageReader reader(pending, work);
  AsyncWriter writer;
  for (const auto& record : pending) {
    const auto& camera = scene.cameras[record.camera];
    const double scale = 1536. / std::max(camera.width, camera.height);
    const int width = int(scale * camera.width) / 8 * 8,
              height = int(scale * camera.height) / 8 * 8;
    if (expected != std::array<int, 2>{width, height})
      throw std::runtime_error("Detector engine resolution differs from camera input");
    auto phase = Clock::now();
    auto image = reader.Next();
    timing.wait += Seconds(phase);
    timing.decode += image.decode;
    timing.preprocess += image.preprocess;
    phase = Clock::now();
    pixels.Upload(image.rgb);
    const auto detected = models.Predict("detector", {pixels.Normalize(width, height)}, true)[0];
    const auto raw = DetectKeypoints(detected, width, height, 12288, models.Stream());
    timing.gpu_pipeline += Seconds(phase);
    phase = Clock::now();
    std::vector<std::array<float, 3>> candidates;
    const double radius = camera.Radius();
    for (const auto& point : raw) {
      const float x = (point[0] + 1) * camera.width / 2, y = (point[1] + 1) * camera.height / 2;
      const double dx = (x - camera.params[2]) / camera.params[0],
                   dy = (y - camera.params[3]) / camera.params[1];
      if (x >= 0 && y >= 0 && x < camera.width && y < camera.height &&
          dx * dx + dy * dy <= radius * radius && image.mask.at<uint8_t>(int(y), int(x)) == 255)
        candidates.push_back({x, y, point[2]});
    }
    const auto selected = Select(candidates, camera);
    if (selected.size() < 32)
      throw std::runtime_error("Too few static features: " + record.name);
    FeatureSet f;
    f.image = record.id;
    for (size_t i : selected) {
      const auto& p = candidates[i];
      f.pixels.push_back({p[0], p[1]});
      f.normalized.push_back({2 * p[0] / camera.width - 1, 2 * p[1] / camera.height - 1});
      f.scores.push_back(p[2]);
      const auto color = image.rgb.at<cv::Vec3b>(int(p[1]), int(p[0]));
      f.colors.push_back({color[0], color[1], color[2]});
    }
    timing.postprocess += Seconds(phase);
    phase = Clock::now();
    const auto description = models.Predict(
        "descriptor", {pixels.Normalize(784, 784),
                       Tensor::Float({1, int64_t(selected.size()), 2}, f.normalized[0].data(),
                                     selected.size() * 2)})[0];
    if (description.shape != std::vector<int64_t>{1, int64_t(selected.size()), 256})
      throw std::runtime_error("Unexpected descriptor output dimensions");
    timing.gpu_pipeline += Seconds(phase);
    phase = Clock::now();
    f.descriptors.assign(description.Floats(), description.Floats() + description.Count());
    std::array<double, 256> mean{};
    for (size_t k = 0; k < selected.size(); ++k)
      for (int d = 0; d < 256; ++d)
        mean[d] += f.descriptors[k * 256 + d];
    double norm = 0;
    for (double x : mean)
      norm += x * x;
    norm = std::max(1e-12, std::sqrt(norm));
    for (int d = 0; d < 256; ++d)
      f.retrieval[d] = float(mean[d] / norm);
    timing.postprocess += Seconds(phase);
    const auto path = FeaturePath(work, record.id);
    writer.Submit([path, f = std::move(f)] { WriteFeatures(path, f); });
    ++timing.computed;
    if (record.id % 16 == 0)
      Log("Features " + std::to_string(record.id + 1) + "/" + std::to_string(scene.images.size()));
  }
  writer.Finish();
  timing.write = writer.seconds;
  return timing.Report(started, &models);
}
Json MakeMatches(const Scene& scene, const fs::path& work, const fs::path& root,
                 const std::set<uint32_t>& repair) {
  const auto started = Clock::now();
  FrontendTiming timing;
  const uint32_t n = uint32_t(scene.images.size());
  std::vector<std::array<float, 256>> retrieval;
  for (uint32_t id = 0; id < n; ++id)
    retrieval.push_back(ReadFeatures(FeaturePath(work, id)).retrieval);
  std::set<std::pair<uint32_t, uint32_t>> pairs;
  auto add = [&](uint32_t a, uint32_t b) {
    if (a != b)
      pairs.emplace(std::min(a, b), std::max(a, b));
  };
  if (fs::exists(work / "cache/pairs.json")) {
    const auto previous = ReadJson(work / "cache/pairs.json");
    for (const auto& p : previous.at("pairs"))
      add(p[0], p[1]);
  }
  for (uint32_t id = 0; id < n; ++id) {
    if (!repair.empty() && !repair.count(id))
      continue;
    const uint32_t span = repair.empty() ? 12 : 48;
    for (uint32_t k = id > span ? id - span : 0; k < std::min(n, id + span + 1); ++k) {
      const uint32_t distance = id > k ? id - k : k - id;
      // Keep the closest exposures dense, then spread connections over the
      // remaining window. Repair only the requested images with dense edges.
      if (repair.empty() && distance > 4 && distance % 2 != 0)
        continue;
      add(id, k);
    }
    // Retrieve independently over multiple time distances, so near-duplicate
    // neighbors do not occupy every loop-closure slot.
    for (const auto range : std::vector<std::array<uint32_t, 2>>{{12, 48}, {48, 160}, {160, n}}) {
      std::vector<std::pair<double, uint32_t>> options;
      for (uint32_t j = 0; j < n; ++j) {
        const uint32_t distance = id > j ? id - j : j - id;
        if (distance < range[0] || distance >= range[1])
          continue;
        double similarity = 0;
        for (int d = 0; d < 256; ++d)
          similarity += retrieval[id][d] * retrieval[j][d];
        options.emplace_back(similarity, j);
      }
      std::sort(options.rbegin(), options.rend());
      std::vector<uint32_t> selected;
      for (size_t k = 0; k < std::min<size_t>(repair.empty() ? 4 : 12, options.size()); ++k) {
        const auto j = options[k].second;
        if (repair.empty()) {
          if (selected.size() == 3)
            break;
          const auto& candidate = scene.images[j];
          const bool duplicate = std::any_of(selected.begin(), selected.end(), [&](uint32_t p) {
            const auto& previous = scene.images[p];
            const double seconds =
                (candidate.week - previous.week) * 604800. + candidate.seconds - previous.seconds;
            return candidate.camera == previous.camera && std::abs(seconds) < 2.;
          });
          if (duplicate)
            continue;
        }
        selected.push_back(j);
        add(id, j);
      }
    }
  }
  std::vector<std::pair<uint32_t, uint32_t>> pending;
  for (const auto& [a, b] : pairs) {
    if (fs::exists(MatchPath(work, a, b)))
      ++timing.reused;
    else
      pending.emplace_back(a, b);
  }
  Json report;
  if (!pending.empty()) {
    Models models(root);
    FeatureCache cache(work);
    AsyncWriter writer;
    for (size_t index = 0; index < pending.size(); ++index) {
      const auto [a, b] = pending[index];
      const auto& fa = cache.Get(a);
      const auto& fb = cache.Get(b);
      for (size_t next = index + 1; next < std::min(pending.size(), index + 3); ++next) {
        cache.Prefetch(pending[next].first);
        cache.Prefetch(pending[next].second);
      }
      const auto na = int64_t(fa.value.pixels.size()), nb = int64_t(fb.value.pixels.size());
      constexpr int64_t capacity = 3072;
      auto phase = Clock::now();
      const auto output = models.Predict("matcher", {fa.keypoints, fb.keypoints, fa.descriptors,
                                                     fb.descriptors, fa.mask, fb.mask});
      timing.gpu_pipeline += Seconds(phase);
      phase = Clock::now();
      if (output.size() != 2 || output[0].Count() != size_t(capacity) ||
          output[1].Count() != size_t(capacity))
        throw std::runtime_error("Unexpected matcher outputs");
      MatchSet m{a, b, {}};
      for (uint32_t k = 0; k < uint32_t(na); ++k) {
        const auto j = output[0].Index(k);
        if (j >= 0) {
          if (j >= nb)
            throw std::runtime_error("Matcher returned invalid feature index");
          m.matches.push_back({k, uint32_t(j), output[1].Floats()[k]});
        }
      }
      timing.postprocess += Seconds(phase);
      const auto path = MatchPath(work, a, b);
      writer.Submit([path, m = std::move(m)] { WriteMatches(path, m); });
      ++timing.computed;
      if ((index + 1) % 500 == 0)
        Log("Matches " + std::to_string(index + 1) + "/" + std::to_string(pending.size()));
    }
    writer.Finish();
    timing.write = writer.seconds;
    timing.decode = cache.read_seconds;
    timing.wait = cache.wait_seconds;
    timing.preprocess = cache.upload_seconds;
    report = timing.Report(started, &models);
  } else
    report = timing.Report(started);
  Json list = Json::array();
  std::vector<double> degrees(n, 0);
  for (const auto& [a, b] : pairs) {
    list.push_back({a, b});
    ++degrees[a];
    ++degrees[b];
  }
  WriteJson(work / "cache/pairs.json", {{"version", 1}, {"pairs", list}});
  report["pairs"] = pairs.size();
  report["pair_degree"] = Statistics(degrees);
  report["pair_degree"]["minimum"] = *std::min_element(degrees.begin(), degrees.end());
  report["repair_images"] = repair;
  return report;
}
}  // namespace geosun

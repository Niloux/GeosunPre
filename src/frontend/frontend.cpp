#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "io/cache.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <list>
#include <numeric>
#include <set>
#include <cmath>
#include <cstring>

namespace geosun {
fs::path MaskPath(const fs::path& work, uint32_t id) {
  return work / "cache/masks" / (std::to_string(id) + ".png");
}
namespace {
struct View {
  cv::Mat source, x, y, weight;
};
struct Fusion {
  std::vector<View> views;
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
    View v{cv::Mat(512, 512, CV_32FC2), cv::Mat(h, w, CV_32FC1), cv::Mat(h, w, CV_32FC1),
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
    std::list<uint32_t>::iterator position;
  };
  std::map<uint32_t, Entry> entries;

 public:
  explicit FeatureCache(fs::path path) : work(std::move(path)) {}
  const FeatureSet& Get(uint32_t id) {
    auto it = entries.find(id);
    if (it != entries.end()) {
      order.splice(order.begin(), order, it->second.position);
      return it->second.value;
    }
    if (entries.size() >= 48) {
      entries.erase(order.back());
      order.pop_back();
    }
    order.push_front(id);
    auto entry = entries.emplace(id, Entry{ReadFeatures(FeaturePath(work, id)), order.begin()});
    return entry.first->second.value;
  }
};
}  // namespace
void MakeMasks(const Scene& scene, const fs::path& work, const fs::path& root) {
  Models models(root);
  std::array<Fusion, 2> fusion{VirtualViews(scene.cameras[0]), VirtualViews(scene.cameras[1])};
  const auto mean = models.Manifest().at("segmentation_mean").get<std::array<float, 3>>(),
             deviation = models.Manifest().at("segmentation_std").get<std::array<float, 3>>();
  for (const auto& record : scene.images) {
    const auto path = MaskPath(work, record.id);
    const auto& camera = scene.cameras[record.camera];
    if (fs::exists(path)) {
      const auto h = ImageHeader(path);
      if (h.at("width") != camera.width || h.at("height") != camera.height || h.at("mode") != "L")
        throw std::runtime_error("Invalid mask cache");
      continue;
    }
    auto image = ReadImage(record.source, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    const auto& geometry = fusion[record.camera];
    cv::Mat sum = cv::Mat::zeros(1200, 900, CV_32FC2);
    for (const auto& v : geometry.views) {
      cv::Mat sampled;
      cv::remap(image, sampled, v.source, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      const auto output =
          models.Predict("segmentation", {ImageTensor(sampled, mean, deviation)})[0];
      if (output.shape != std::vector<int64_t>{1, 2, 512, 512})
        throw std::runtime_error("Unexpected segmentation output");
      const auto* p = output.Floats();
      cv::Mat probability(512, 512, CV_32FC2);
      for (int y = 0; y < 512; ++y)
        for (int x = 0; x < 512; ++x)
          probability.at<cv::Vec2f>(y, x) = {p[y * 512 + x], p[512 * 512 + y * 512 + x]};
      cv::Mat projected;
      cv::remap(probability, projected, v.x, v.y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      for (int y = 0; y < 1200; ++y)
        for (int x = 0; x < 900; ++x)
          sum.at<cv::Vec2f>(y, x) += projected.at<cv::Vec2f>(y, x) * v.weight.at<float>(y, x);
    }
    cv::Mat sky(1200, 900, CV_8UC1), person(1200, 900, CV_8UC1);
    for (int y = 0; y < 1200; ++y)
      for (int x = 0; x < 900; ++x) {
        const auto p = sum.at<cv::Vec2f>(y, x) / std::max(geometry.weight.at<float>(y, x), 1e-8f);
        sky.at<uint8_t>(y, x) = p[0] > .90 ? 255 : 0;
        person.at<uint8_t>(y, x) = p[1] > .55 ? 255 : 0;
      }
    const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
    cv::erode(sky, sky, kernel);
    cv::dilate(person, person, kernel);
    cv::resize(sky, sky, {camera.width, camera.height}, 0, 0, cv::INTER_NEAREST);
    cv::resize(person, person, {camera.width, camera.height}, 0, 0, cv::INTER_NEAREST);
    auto mask = camera.Mask();
    for (int y = 0; y < camera.height; ++y)
      for (int x = 0; x < camera.width; ++x)
        if (mask.at<uint8_t>(y, x))
          mask.at<uint8_t>(y, x) = person.at<uint8_t>(y, x) ? 0 : sky.at<uint8_t>(y, x) ? 128 : 255;
    auto temporary = path.parent_path() / (path.stem().u8string() + ".writing.png");
    WriteImage(temporary, mask);
    PublishFile(temporary, path);
    if (record.id % 16 == 0)
      Log("Masks " + std::to_string(record.id + 1) + "/" + std::to_string(scene.images.size()));
  }
}
void MakeFeatures(const Scene& scene, const fs::path& work, const fs::path& root) {
  Models models(root);
  const auto expected = models.Manifest().at("detector_size").get<std::array<int, 2>>();
  for (const auto& record : scene.images) {
    const auto path = FeaturePath(work, record.id);
    if (fs::exists(path)) {
      const auto f = ReadFeatures(path);
      if (f.image != record.id)
        throw std::runtime_error("Feature identity differs");
      continue;
    }
    const auto& camera = scene.cameras[record.camera];
    const double scale = 1536. / std::max(camera.width, camera.height);
    const int width = int(scale * camera.width) / 8 * 8,
              height = int(scale * camera.height) / 8 * 8;
    if (expected != std::array<int, 2>{width, height})
      throw std::runtime_error("Detector engine resolution differs from camera input");
    auto image = ReadImage(record.source, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    const auto detected =
        models.Predict("detector", {ImageTensor(BicubicResize(image, width, height))})[0];
    const auto raw = DetectKeypoints(detected, width, height, 12288);
    const auto mask = ReadImage(MaskPath(work, record.id), cv::IMREAD_GRAYSCALE);
    std::vector<std::array<float, 3>> candidates;
    for (const auto& point : raw) {
      const float x = (point[0] + 1) * camera.width / 2, y = (point[1] + 1) * camera.height / 2;
      if (camera.Valid(x, y) && mask.at<uint8_t>(int(y), int(x)) == 255)
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
    }
    const auto description = models.Predict(
        "descriptor", {ImageTensor(BicubicResize(image, 784, 784)),
                       Tensor::Float({1, int64_t(selected.size()), 2}, f.normalized[0].data(),
                                     selected.size() * 2)})[0];
    if (description.shape != std::vector<int64_t>{1, int64_t(selected.size()), 256})
      throw std::runtime_error("Unexpected descriptor output dimensions");
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
    WriteFeatures(path, f);
    if (record.id % 16 == 0)
      Log("Features " + std::to_string(record.id + 1) + "/" + std::to_string(scene.images.size()));
  }
}
void MakeMatches(const Scene& scene, const fs::path& work, const fs::path& root,
                 const std::set<uint32_t>& repair) {
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
    for (uint32_t k = id > span ? id - span : 0; k < std::min(n, id + span + 1); ++k)
      add(id, k);
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
      for (size_t k = 0; k < std::min<size_t>(repair.empty() ? 4 : 12, options.size()); ++k)
        add(id, options[k].second);
    }
  }
  Models models(root);
  FeatureCache cache(work);
  size_t completed = 0;
  for (const auto& [a, b] : pairs) {
    const auto path = MatchPath(work, a, b);
    if (!fs::exists(path)) {
      const auto& fa = cache.Get(a);
      const auto& fb = cache.Get(b);
      const auto na = int64_t(fa.pixels.size()), nb = int64_t(fb.pixels.size());
      constexpr int64_t capacity = 3072;
      std::vector<float> ka(capacity * 2), kb(capacity * 2), da(capacity * 256), db(capacity * 256),
          ma(capacity), mb(capacity);
      std::memcpy(ka.data(), fa.normalized[0].data(), na * 2 * sizeof(float));
      std::memcpy(kb.data(), fb.normalized[0].data(), nb * 2 * sizeof(float));
      std::copy(fa.descriptors.begin(), fa.descriptors.end(), da.begin());
      std::copy(fb.descriptors.begin(), fb.descriptors.end(), db.begin());
      std::fill(ma.begin(), ma.begin() + na, 1.f);
      std::fill(mb.begin(), mb.begin() + nb, 1.f);
      const auto output =
          models.Predict("matcher", {Tensor::Float({1, capacity, 2}, ka.data(), ka.size()),
                                     Tensor::Float({1, capacity, 2}, kb.data(), kb.size()),
                                     Tensor::Float({1, capacity, 256}, da.data(), da.size()),
                                     Tensor::Float({1, capacity, 256}, db.data(), db.size()),
                                     Tensor::Float({1, capacity}, ma.data(), ma.size()),
                                     Tensor::Float({1, capacity}, mb.data(), mb.size())});
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
      WriteMatches(path, m);
    }
    if (++completed % 500 == 0)
      Log("Matches " + std::to_string(completed) + "/" + std::to_string(pairs.size()));
  }
  Json list = Json::array();
  for (const auto& [a, b] : pairs)
    list.push_back({a, b});
  WriteJson(work / "cache/pairs.json", {{"version", 1}, {"pairs", list}});
}
}  // namespace geosun

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
#include <condition_variable>
#include <mutex>
#include <thread>

namespace geosun {
fs::path MaskPath(const fs::path& work, uint32_t id) {
  return work / "cache/masks" / (std::to_string(id) + ".png");
}
namespace {
double ExposureDeltaSeconds(const Exposure& a, const Exposure& b) {
  return (a.week - b.week) * 604800. + a.seconds - b.seconds;
}
// A same-lens exposure passed in the opposite travel direction faces the other side: the rig has
// turned half a circle, so the other lens covers what this one saw. On 生产车间 half of these pairs
// shared no reconstructed point. Exposures with no clear motion are kept.
bool FacesAway(const Scene& scene, const std::vector<Eigen::Vector3d>& velocities, uint32_t a,
               uint32_t b) {
  constexpr double moving_m_s = .1;
  const auto &va = velocities[a], &vb = velocities[b];
  return scene.images[a].camera == scene.images[b].camera && va.norm() > moving_m_s &&
         vb.norm() > moving_m_s && va.dot(vb) < -.5 * va.norm() * vb.norm();
}
std::vector<uint32_t> NavigationRevisitCandidates(const Scene& scene,
                                                  const std::vector<Eigen::Vector3d>& positions,
                                                  const std::vector<Eigen::Vector3d>& velocities,
                                                  uint32_t image) {
  constexpr double same_radius_m = 5., cross_radius_m = 4., minimum_seconds = 10.;
  constexpr size_t maximum_candidates = 6, maximum_cross_candidates = 2;
  std::vector<std::pair<double, uint32_t>> same_nearby, cross_nearby;
  const auto& exposure = scene.images[image];
  for (uint32_t id = 0; id < scene.images.size(); ++id) {
    const auto& other = scene.images[id];
    const double seconds = ExposureDeltaSeconds(exposure, other);
    if (std::abs(seconds) <= minimum_seconds)
      continue;
    const double squared_distance = (positions[image] - positions[id]).squaredNorm();
    if (other.camera == exposure.camera) {
      if (squared_distance <= same_radius_m * same_radius_m &&
          !FacesAway(scene, velocities, image, id))
        same_nearby.emplace_back(squared_distance, id);
    } else if (squared_distance <= cross_radius_m * cross_radius_m) {
      cross_nearby.emplace_back(squared_distance, id);
    }
  }
  std::sort(same_nearby.begin(), same_nearby.end());
  std::sort(cross_nearby.begin(), cross_nearby.end());
  std::vector<uint32_t> candidates;
  auto select_diverse = [&](const std::vector<std::pair<double, uint32_t>>& nearby, size_t cap) {
    std::vector<uint32_t> chosen;
    auto pick_pass = [&](double min_dt, bool check_dist) {
      for (const auto& entry : nearby) {
        if (chosen.size() >= cap)
          break;
        const uint32_t id = entry.second;
        if (std::find(chosen.begin(), chosen.end(), id) != chosen.end())
          continue;
        const auto& candidate = scene.images[id];
        const bool duplicate = std::any_of(chosen.begin(), chosen.end(), [&](uint32_t prev_id) {
          const double dt = std::abs(ExposureDeltaSeconds(candidate, scene.images[prev_id]));
          return dt < min_dt ||
                 (check_dist && (positions[id] - positions[prev_id]).squaredNorm() < 1.);
        });
        if (!duplicate)
          chosen.push_back(id);
      }
    };
    pick_pass(12., false);
    pick_pass(2.5, true);
    candidates.insert(candidates.end(), chosen.begin(), chosen.end());
  };
  select_diverse(same_nearby, maximum_candidates - 1);
  if (candidates.size() < maximum_candidates)
    select_diverse(cross_nearby,
                   std::min(maximum_cross_candidates, maximum_candidates - candidates.size()));
  return candidates;
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
  cv::Mat rgb, mask, grounding;
  double decode = 0, preprocess = 0;
};
class ImageReader {
  const std::vector<Exposure>& images;
  const fs::path work;
  // Square side of the grounding input for the masks; 0 reads the masks instead.
  const int grounding;
  const std::array<cv::Rect, 2> grounding_rois;
  const std::array<cv::Size, 2> camera_sizes;
  size_t next = 0;
  std::deque<std::future<PreparedImage>> ready;
  // Four photos in flight keep the GPU fed with a few full-size photos in memory.
  void Fill() {
    while (ready.size() < 4 && next < images.size()) {
      const auto record = images[next++];
      ready.push_back(std::async(std::launch::async, [this, record] {
        PreparedImage image;
        image.id = record.id;
        auto started = Clock::now();
        const cv::Rect& roi = grounding_rois[record.camera];
        const bool use_half =
            grounding > 0 && roi.width >= 2 * grounding && roi.height >= 2 * grounding;
        image.rgb = ReadImage(record.source, use_half ? cv::IMREAD_REDUCED_COLOR_2 : cv::IMREAD_COLOR);
        image.decode = Seconds(started);
        started = Clock::now();
        cv::cvtColor(image.rgb, image.rgb, cv::COLOR_BGR2RGB);
        if (grounding) {
          cv::Rect scaled_roi = roi;
          const cv::Size cam_sz = camera_sizes[record.camera];
          if (!scaled_roi.empty() && cam_sz.width > 0 && cam_sz.height > 0 &&
              (image.rgb.cols != cam_sz.width || image.rgb.rows != cam_sz.height)) {
            const double sx = double(image.rgb.cols) / double(cam_sz.width);
            const double sy = double(image.rgb.rows) / double(cam_sz.height);
            const int x0 = std::clamp(int(std::lround(roi.x * sx)), 0, image.rgb.cols - 1);
            const int y0 = std::clamp(int(std::lround(roi.y * sy)), 0, image.rgb.rows - 1);
            const int rw = std::clamp(int(std::lround(roi.width * sx)), 1, image.rgb.cols - x0);
            const int rh = std::clamp(int(std::lround(roi.height * sy)), 1, image.rgb.rows - y0);
            scaled_roi = {x0, y0, rw, rh};
          }
          const cv::Mat source = scaled_roi.empty() ? image.rgb : image.rgb(scaled_roi);
          cv::resize(source, image.grounding, {grounding, grounding}, 0, 0, cv::INTER_AREA);
          image.rgb.release();
        } else
          image.mask = ReadImage(MaskPath(work, record.id), cv::IMREAD_GRAYSCALE);
        image.preprocess = Seconds(started);
        return image;
      }));
    }
  }

 public:
  ImageReader(const std::vector<Exposure>& records, fs::path path, int grounding_size = 0,
              std::array<cv::Rect, 2> rois = {}, std::array<cv::Size, 2> cam_sizes = {})
      : images(records),
        work(std::move(path)),
        grounding(grounding_size),
        grounding_rois(rois),
        camera_sizes(cam_sizes) {
    Fill();
  }
  PreparedImage Next() {
    auto image = ready.front().get();
    ready.pop_front();
    Fill();
    return image;
  }
};
class WriterPool {
  std::deque<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable can_push_, can_pop_;
  bool stop_ = false;
  std::exception_ptr error_;
  std::vector<std::thread> workers_;
  size_t max_queue_ = 8;

  void Run() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        can_pop_.wait(lock, [&] { return !queue_.empty() || stop_; });
        if (queue_.empty() && stop_)
          return;
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      can_push_.notify_one();
      const auto started = Clock::now();
      try {
        task();
        const double elapsed = Seconds(started);
        std::lock_guard<std::mutex> lock(mutex_);
        seconds += elapsed;
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_)
          error_ = std::current_exception();
        stop_ = true;
        can_push_.notify_all();
        can_pop_.notify_all();
        return;
      }
    }
  }

 public:
  double seconds = 0;

  explicit WriterPool(int threads = 4, size_t max_queue = 0) {
    const int n = std::clamp(threads, 1, 8);
    max_queue_ = max_queue ? max_queue : std::max<size_t>(8, size_t(n) * 2);
    workers_.reserve(size_t(n));
    for (int i = 0; i < n; ++i)
      workers_.emplace_back([this] { Run(); });
  }

  ~WriterPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    can_push_.notify_all();
    can_pop_.notify_all();
    for (auto& w : workers_)
      if (w.joinable())
        w.join();
  }

  void Submit(std::function<void()> write) {
    std::unique_lock<std::mutex> lock(mutex_);
    can_push_.wait(lock, [&] { return queue_.size() < max_queue_ || error_ != nullptr || stop_; });
    if (error_)
      std::rethrow_exception(error_);
    queue_.push_back(std::move(write));
    can_pop_.notify_one();
  }

  void Finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    can_pop_.notify_all();
    for (auto& w : workers_)
      if (w.joinable())
        w.join();
    if (error_)
      std::rethrow_exception(error_);
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
  static constexpr size_t kCapacity = 3072;
  static constexpr size_t kKeypointsBytes = kCapacity * 2 * sizeof(float);
  static constexpr size_t kDescriptorsBytes = kCapacity * 256 * sizeof(float);
  static constexpr size_t kMaskBytes = kCapacity * sizeof(float);
  const fs::path work;
  std::list<uint32_t> order;
  struct Entry {
    FeatureSet value;
    Tensor keypoints, descriptors, mask;
    std::list<uint32_t>::iterator position;
  };
  std::map<uint32_t, Entry> entries;
  std::vector<std::shared_ptr<DeviceBuffer>> free_keypoints, free_descriptors, free_masks;
  HostBuffer pinned;
  cudaStream_t upload_stream = nullptr;
  struct Loaded {
    FeatureSet value;
    double seconds;
  };
  std::map<uint32_t, std::future<Loaded>> pending;

 public:
  explicit FeatureCache(fs::path path)
      : work(std::move(path)), pinned(kKeypointsBytes + kDescriptorsBytes + kMaskBytes) {
    CudaCheck(cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking));
  }
  ~FeatureCache() {
    for (auto& job : pending)
      if (job.second.valid())
        job.second.wait();
    if (upload_stream) {
      cudaStreamSynchronize(upload_stream);
      cudaStreamDestroy(upload_stream);
    }
  }
  double read_seconds = 0, wait_seconds = 0, upload_seconds = 0;
  void Prefetch(uint32_t id) {
    if (entries.count(id) || pending.count(id) || pending.size() >= 8)
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
    if (entries.size() >= 96) {
      const auto evicted = entries.find(order.back());
      auto recycle = [](std::shared_ptr<DeviceBuffer>& buf,
                        std::vector<std::shared_ptr<DeviceBuffer>>& pool) {
        if (buf && buf.use_count() == 1)
          pool.push_back(std::move(buf));
      };
      recycle(evicted->second.keypoints.device, free_keypoints);
      recycle(evicted->second.descriptors.device, free_descriptors);
      recycle(evicted->second.mask.device, free_masks);
      entries.erase(evicted);
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
    auto acquire = [](std::vector<std::shared_ptr<DeviceBuffer>>& pool, size_t bytes) {
      if (!pool.empty()) {
        auto buf = std::move(pool.back());
        pool.pop_back();
        return buf;
      }
      return std::make_shared<DeviceBuffer>(bytes);
    };
    auto kp_dev = acquire(free_keypoints, kKeypointsBytes);
    auto desc_dev = acquire(free_descriptors, kDescriptorsBytes);
    auto mask_dev = acquire(free_masks, kMaskBytes);
    auto* kp_host = static_cast<float*>(pinned.data);
    auto* desc_host =
        reinterpret_cast<float*>(static_cast<uint8_t*>(pinned.data) + kKeypointsBytes);
    auto* mask_host = reinterpret_cast<float*>(static_cast<uint8_t*>(pinned.data) +
                                               kKeypointsBytes + kDescriptorsBytes);
    std::memcpy(kp_host, entry.value.normalized[0].data(), n * 2 * sizeof(float));
    if (n < kCapacity)
      std::memset(kp_host + n * 2, 0, (kCapacity - n) * 2 * sizeof(float));
    std::memcpy(desc_host, entry.value.descriptors.data(), n * 256 * sizeof(float));
    if (n < kCapacity)
      std::memset(desc_host + n * 256, 0, (kCapacity - n) * 256 * sizeof(float));
    std::fill_n(mask_host, n, 1.f);
    if (n < kCapacity)
      std::fill_n(mask_host + n, kCapacity - n, 0.f);
    CudaCheck(cudaMemcpyAsync(kp_dev->data, kp_host, kKeypointsBytes, cudaMemcpyHostToDevice,
                              upload_stream));
    CudaCheck(cudaMemcpyAsync(desc_dev->data, desc_host, kDescriptorsBytes, cudaMemcpyHostToDevice,
                              upload_stream));
    CudaCheck(cudaMemcpyAsync(mask_dev->data, mask_host, kMaskBytes, cudaMemcpyHostToDevice,
                              upload_stream));
    CudaCheck(cudaStreamSynchronize(upload_stream));
    entry.keypoints.shape = {1, int64_t(kCapacity), 2};
    entry.keypoints.type = 0;
    entry.keypoints.device = std::move(kp_dev);
    entry.descriptors.shape = {1, int64_t(kCapacity), 256};
    entry.descriptors.type = 0;
    entry.descriptors.device = std::move(desc_dev);
    entry.mask.shape = {1, int64_t(kCapacity)};
    entry.mask.type = 0;
    entry.mask.device = std::move(mask_dev);
    upload_seconds += Seconds(uploading);
    entry.position = order.begin();
    return entries.emplace(id, std::move(entry)).first->second;
  }
};
class ArchiveWriterQueue {
  MatchArchiveWriter writer_;
  std::deque<MatchSet> queue_;
  std::mutex mutex_;
  std::condition_variable can_push_, can_pop_;
  bool stop_ = false;
  std::exception_ptr error_;
  std::thread worker_;

 public:
  double seconds = 0;
  explicit ArchiveWriterQueue(const fs::path& path)
      : writer_(path), worker_([this] { Run(); }) {}
  ~ArchiveWriterQueue() {
    if (worker_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
      }
      can_pop_.notify_all();
      worker_.join();
    }
  }
  void Submit(MatchSet&& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    can_push_.wait(lock, [&] { return queue_.size() < 128 || error_ != nullptr; });
    if (error_)
      std::rethrow_exception(error_);
    queue_.push_back(std::move(item));
    can_pop_.notify_one();
  }
  void Finish() {
    if (!worker_.joinable())
      return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    can_pop_.notify_all();
    worker_.join();
    if (error_)
      std::rethrow_exception(error_);
  }

 private:
  void Run() {
    try {
      for (;;) {
        std::deque<MatchSet> batch;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          can_pop_.wait(lock, [&] { return !queue_.empty() || stop_; });
          if (queue_.empty() && stop_)
            break;
          batch.swap(queue_);
        }
        can_push_.notify_all();
        const auto started = Clock::now();
        for (const auto& m : batch)
          writer_.Append(m);
        seconds += Seconds(started);
      }
      const auto started = Clock::now();
      writer_.Flush();
      seconds += Seconds(started);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
      can_push_.notify_all();
    }
  }
};
// SAM3 grounding of the centered fisheye square ROI, prompts person, shadow, robot and sky.
constexpr float kGroundingScore = .5f;
constexpr float kRimPersonScore = .25f;
constexpr float kShadowScore = .25f;
constexpr int kRimBandCells = 10;
constexpr int kAccessoryCloseRadius = 16;
constexpr int kAccessoryBridgeDilateRadius = 5;
constexpr double kAccessoryMaxDomainShare = .08;
constexpr double kAccessoryMaxPersonRatio = .50;
constexpr double kPocketMaxDomainShare = .03;
constexpr double kPocketMinPeopleBorderShare = .35;
constexpr int kTransientMarginRadius = 2;
// A shadow is transient only within this many mask cells (about 8% of the photo side) of a
// person: static furniture and exhibit shadows are seen in every view and stay static.
constexpr int kShadowReach = 24;
// Oversized shadows (> 8% of the grounding grid, or > 3% and > 3x the touching person component
// area) are clipped to kShadowReach around people so one pedestrian stepping onto a building
// shadow does not mask an entire courtyard floor.
constexpr double kShadowMaxFullShare = .08;
constexpr double kShadowSmallShare = .03;
constexpr double kShadowMaxPersonAreaRatio = 3.;
struct GroundingCounts {
  size_t persons = 0, rim_persons = 0, robots = 0, accessory_bridges = 0, filled_pockets = 0,
         shadows = 0, clipped_shadows = 0, distant_shadows = 0, skies = 0, sky_images = 0;
};
int GroundingSize(const Json& manifest) {
  if (manifest.at("grounding_prompts") != Json{"person", "shadow", "robot", "sky"} ||
      manifest.at("models").at("grounding").at("outputs") !=
          Json{"scores", "boxes", "masks", "semantic"})
    throw std::runtime_error(
        "Masks need SAM3 grounding of person/shadow/robot/sky; prepare the models with "
        "run.ps1 prepare-models");
  return manifest.at("grounding_size");
}
// Square ROI centered on the principal point that keeps full circular fisheye context while
// stripping 4:3 pillarbox borders so SAM3 runs at 1:1 aspect ratio without horizontal squashing.
cv::Rect GroundingRoi(const Camera& camera, const cv::Mat& domain) {
  const cv::Rect box = cv::boundingRect(domain);
  if (box.empty())
    return {0, 0, camera.width, camera.height};
  const int need = std::max(box.width, box.height);
  const int side = std::min(std::max(std::min(camera.width, camera.height), need),
                            std::max(camera.width, camera.height));
  const int rw = std::min(camera.width, side);
  const int rh = std::min(camera.height, side);
  int x0 = std::clamp(int(std::lround(camera.params[2] - 0.5 * rw)), 0, camera.width - rw);
  int y0 = std::clamp(int(std::lround(camera.params[3] - 0.5 * rh)), 0, camera.height - rh);
  const int x_lo = std::max(0, box.x + box.width - rw), x_hi = std::min(box.x, camera.width - rw);
  if (x_lo <= x_hi)
    x0 = std::clamp(x0, x_lo, x_hi);
  const int y_lo = std::max(0, box.y + box.height - rh), y_hi = std::min(box.y, camera.height - rh);
  if (y_lo <= y_hi)
    y0 = std::clamp(y0, y_lo, y_hi);
  return {x0, y0, rw, rh};
}
struct GroundingResult {
  cv::Mat transient, sky;
};
// Coarse grounding-grid (size x size) 255 where people, their bridged wrist/hand accessories, and
// the shadows next to them are, and where the sky is. A person detection mostly covered by an at
// least as confident robot detection is a humanoid exhibit and stays static.
GroundingResult GroundingMaskCoarse(const std::vector<Tensor>& outputs,
                                    const cv::Mat& roi_domain_coarse, const cv::Mat& rim_zone,
                                    GroundingCounts& counts) {
  const int k = int(outputs[0].shape[1]), size = int(outputs[2].shape[3]);
  if (roi_domain_coarse.cols != size || roi_domain_coarse.rows != size ||
      rim_zone.cols != size || rim_zone.rows != size)
    throw std::runtime_error("Grounding domain grid differs from mask output size");
  const float *scores = outputs[0].Floats(), *logits = outputs[2].Floats();
  auto detections = [&](int prompt, float threshold) {
    std::vector<std::pair<float, cv::Mat>> found;
    // Scores arrive sorted in descending order.
    for (int q = 0; q < k && scores[prompt * k + q] >= threshold; ++q) {
      const cv::Mat logit(size, size, CV_32FC1,
                          const_cast<float*>(logits + (size_t(prompt) * k + q) * size * size));
      found.emplace_back(scores[prompt * k + q], logit > 0);
    }
    return found;
  };
  const auto persons = detections(0, kRimPersonScore), shadows = detections(1, kShadowScore),
             robots = detections(2, kRimPersonScore), skies = detections(3, kGroundingScore);
  cv::Mat people = cv::Mat::zeros(size, size, CV_8UC1);
  for (const auto& [score, mask] : persons) {
    const int area = cv::countNonZero(mask);
    if (!area)
      continue;
    bool robot = false;
    for (const auto& [robot_score, robot_mask] : robots)
      robot = robot || (robot_score >= score && 2 * cv::countNonZero(mask & robot_mask) >= area);
    if (robot) {
      ++counts.robots;
      continue;
    }
    if (score < kGroundingScore) {
      if (!cv::countNonZero(mask & rim_zone))
        continue;
      ++counts.rim_persons;
    }
    ++counts.persons;
    people |= mask;
  }
  const int people_area = cv::countNonZero(people);
  if (people_area > 0) {
    const int dom_area = std::max(1, cv::countNonZero(roi_domain_coarse));
    const cv::Mat close_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * kAccessoryCloseRadius + 1, 2 * kAccessoryCloseRadius + 1});
    cv::Mat closed;
    cv::morphologyEx(people, closed, cv::MORPH_CLOSE, close_kernel);
    const cv::Mat bridge = closed & ~people;
    if (cv::countNonZero(bridge) > 0) {
      cv::Mat bridge_labels, bridge_stats, bridge_centroids;
      const int bridge_count = cv::connectedComponentsWithStats(
          bridge, bridge_labels, bridge_stats, bridge_centroids, 8);
      const int max_bridge_area =
          std::min(int(std::lround(kAccessoryMaxDomainShare * dom_area)),
                   int(std::lround(kAccessoryMaxPersonRatio * people_area)));
      cv::Mat compact_bridge = cv::Mat::zeros(size, size, CV_8UC1);
      for (int label = 1; label < bridge_count; ++label) {
        if (bridge_stats.at<int>(label, cv::CC_STAT_AREA) <= max_bridge_area) {
          compact_bridge.setTo(255, bridge_labels == label);
          ++counts.accessory_bridges;
        }
      }
      if (cv::countNonZero(compact_bridge) > 0) {
        const cv::Mat bridge_dilate = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            {2 * kAccessoryBridgeDilateRadius + 1, 2 * kAccessoryBridgeDilateRadius + 1});
        cv::Mat expanded;
        cv::dilate(compact_bridge, expanded, bridge_dilate);
        people |= expanded;
      }
    }
    const cv::Mat free_zone = (people == 0) & (roi_domain_coarse > 0);
    cv::Mat free_labels, free_stats, free_centroids;
    const int free_count =
        cv::connectedComponentsWithStats(free_zone, free_labels, free_stats, free_centroids, 8);
    const int max_pocket_area = int(std::lround(kPocketMaxDomainShare * dom_area));
    const cv::Mat k3 = cv::Mat::ones(3, 3, CV_8UC1);
    for (int label = 1; label < free_count; ++label) {
      if (free_stats.at<int>(label, cv::CC_STAT_AREA) > max_pocket_area)
        continue;
      const cv::Mat comp = (free_labels == label);
      cv::Mat dilated;
      cv::dilate(comp, dilated, k3);
      const cv::Mat border = dilated & ~comp;
      const int border_total = cv::countNonZero(border);
      const int border_people = cv::countNonZero(border & people);
      if (border_total > 0 &&
          double(border_people) >= kPocketMinPeopleBorderShare * double(border_total)) {
        people |= comp;
        ++counts.filled_pockets;
      }
    }
  }
  cv::Mat transient = people.clone();
  if (!shadows.empty()) {
    cv::Mat people_labels, people_stats, people_centroids;
    const int people_count =
        cv::connectedComponentsWithStats(people, people_labels, people_stats, people_centroids, 8);
    if (people_count <= 1) {
      counts.distant_shadows += shadows.size();
    } else {
      const cv::Mat reach_kernel = cv::getStructuringElement(
          cv::MORPH_ELLIPSE, {2 * kShadowReach + 1, 2 * kShadowReach + 1});
      cv::Mat reach;
      cv::dilate(people, reach, reach_kernel);
      const int max_full_cells = int(std::lround(kShadowMaxFullShare * size * size));
      const int small_cells = int(std::lround(kShadowSmallShare * size * size));
      for (const auto& [score, mask] : shadows) {
        const cv::Mat near = mask & reach;
        if (!cv::countNonZero(near)) {
          ++counts.distant_shadows;
          continue;
        }
        const int shadow_area = cv::countNonZero(mask);
        bool oversized = shadow_area > max_full_cells;
        if (!oversized && shadow_area > small_cells) {
          cv::Mat mask_reach;
          cv::dilate(mask, mask_reach, reach_kernel);
          std::vector<uint8_t> hit(size_t(people_count), 0);
          for (int y = 0; y < size; ++y) {
            const uint8_t* m = mask_reach.ptr<uint8_t>(y);
            const int* l = people_labels.ptr<int>(y);
            for (int x = 0; x < size; ++x)
              if (m[x] && l[x] > 0)
                hit[size_t(l[x])] = 1;
          }
          int touching_person_area = 0;
          for (int label = 1; label < people_count; ++label)
            if (hit[size_t(label)])
              touching_person_area += people_stats.at<int>(label, cv::CC_STAT_AREA);
          oversized =
              double(shadow_area) > kShadowMaxPersonAreaRatio * double(touching_person_area);
        }
        ++counts.shadows;
        if (oversized) {
          ++counts.clipped_shadows;
          transient |= near;
        } else {
          transient |= mask;
        }
      }
    }
  }
  // Two cells of margin cover the coarse mask boundary, dark fringes, and soft shadow edges.
  const cv::Mat margin_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, {2 * kTransientMarginRadius + 1, 2 * kTransientMarginRadius + 1});
  cv::dilate(transient, transient, margin_kernel);
  // Sky is the union of its detections without margin: indoors a bright ceiling panel or a
  // printed sky occasionally qualifies and is accepted.
  cv::Mat sky = cv::Mat::zeros(size, size, CV_8UC1);
  for (const auto& [score, mask] : skies)
    sky |= mask;
  counts.skies += skies.size();
  counts.sky_images += !skies.empty();
  return {std::move(transient), std::move(sky)};
}
cv::Mat ComposeFullMask(const GroundingResult& coarse, const cv::Mat& domain, const cv::Rect& roi) {
  cv::Mat transient, sky;
  cv::resize(coarse.transient, transient, {roi.width, roi.height}, 0, 0, cv::INTER_LINEAR);
  cv::resize(coarse.sky, sky, {roi.width, roi.height}, 0, 0, cv::INTER_LINEAR);
  transient = transient >= 128;
  sky = sky >= 128;
  cv::Mat mask = domain.clone();
  cv::Mat sub = mask(roi);
  sub.setTo(128, sky & sub);
  sub.setTo(0, transient);
  return mask;
}
}  // namespace
Json CheckModels(const fs::path& root, const fs::path& image) {
  auto rgb = ReadImage(image, cv::IMREAD_COLOR);
  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  const auto manifest = ReadJson(root / "manifest.json");
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
    const auto size = models.DetectorSize(rgb.cols, rgb.rows);
    report["detector_size"] = size;
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
    if (models.MaxBatch("matcher") >= 2) {
      auto stack_buf = [&](const Tensor& src, size_t per_bytes, std::vector<int64_t> shape) {
        Tensor dst;
        dst.shape = std::move(shape);
        dst.type = src.type;
        dst.device = std::make_shared<DeviceBuffer>(2 * per_bytes);
        CudaCheck(cudaMemcpyAsync(dst.device->data, src.device->data, per_bytes,
                                  cudaMemcpyDeviceToDevice, models.Stream()));
        CudaCheck(cudaMemcpyAsync(static_cast<uint8_t*>(dst.device->data) + per_bytes,
                                  src.device->data, per_bytes, cudaMemcpyDeviceToDevice,
                                  models.Stream()));
        return dst;
      };
      const auto bk = stack_buf(k, 3072 * 2 * sizeof(float), {2, 3072, 2});
      const auto bd = stack_buf(d, 3072 * 256 * sizeof(float), {2, 3072, 256});
      const auto bm = stack_buf(mask, 3072 * sizeof(float), {2, 3072});
      const auto bmatched = models.Predict("matcher", {bk, bk, bd, bd, bm, bm});
      if (bmatched.size() != 2 || bmatched[0].type != 2 ||
          bmatched[0].shape != std::vector<int64_t>{2, 3072})
        throw std::runtime_error("Model verification: invalid batched matcher output");
      check(bmatched[1], {2, 3072});
      for (int b = 0; b < 2; ++b)
        for (size_t i = 0; i < 3072; ++i)
          if (bmatched[0].Index(size_t(b) * 3072 + i) != matched[0].Index(i))
            throw std::runtime_error("Model verification: batched matcher index mismatch");
    }
    report["identity_matches"] = identities;
    report["matcher_max_batch"] = models.MaxBatch("matcher");
    report["inference"]["matcher"] = models.Profile().at("matcher");
  }
  // MoGe-2 monocular geometry, used by supervision export.
  {
    Models models(root);
    ImagePreprocessor pixels(models.Stream());
    pixels.Upload(rgb);
    const int64_t size = manifest.at("geometry_size");
    const auto face_in = pixels.Normalize(int(size), int(size));
    const auto outputs = models.Predict("geometry", {face_in});
    if (outputs.size() != 4)
      throw std::runtime_error("Model verification: invalid geometry outputs");
    check(outputs[0], {1, size, size, 3});
    check(outputs[1], {1, size, size, 3});
    check(outputs[2], {1, size, size});
    check(outputs[3], {1});
    const auto *points = outputs[0].Floats(), *normals = outputs[1].Floats(),
               *mask = outputs[2].Floats();
    size_t valid = 0;
    for (int64_t i = 0; i < size * size; ++i) {
      const float* n = normals + 3 * i;
      if (mask[i] < 0 || mask[i] > 1 || std::abs(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1) > 2e-3f)
        throw std::runtime_error("Model verification: invalid geometry mask or normal");
      if (mask[i] > .5f) {
        if (points[3 * i + 2] <= 0)
          throw std::runtime_error("Model verification: geometry depth is not positive");
        ++valid;
      }
    }
    if (valid * 2 < size_t(size * size) || outputs[3].Floats()[0] <= 0)
      throw std::runtime_error("Model verification: geometry point map is mostly invalid");
    if (models.MaxBatch("geometry") >= 2) {
      const size_t face_bytes = size_t(3) * size * size * sizeof(float);
      Tensor b_in;
      b_in.shape = {2, 3, size, size};
      b_in.device = std::make_shared<DeviceBuffer>(2 * face_bytes);
      CudaCheck(cudaMemcpyAsync(b_in.device->data, face_in.device->data, face_bytes,
                                cudaMemcpyDeviceToDevice, models.Stream()));
      CudaCheck(cudaMemcpyAsync(static_cast<uint8_t*>(b_in.device->data) + face_bytes,
                                face_in.device->data, face_bytes, cudaMemcpyDeviceToDevice,
                                models.Stream()));
      const auto bout = models.Predict("geometry", {b_in});
      if (bout.size() != 4)
        throw std::runtime_error("Model verification: invalid batched geometry outputs");
      check(bout[0], {2, size, size, 3});
      check(bout[1], {2, size, size, 3});
      check(bout[2], {2, size, size});
      check(bout[3], {2});
      if (std::abs(bout[3].Floats()[0] - outputs[3].Floats()[0]) > 1e-2f ||
          std::abs(bout[3].Floats()[1] - outputs[3].Floats()[0]) > 1e-2f)
        throw std::runtime_error("Model verification: batched geometry scale mismatch");
    }
    report["geometry_valid_pixels"] = valid;
    report["geometry_max_batch"] = models.MaxBatch("geometry");
    report["inference"]["geometry"] = models.Profile().at("geometry");
  }
  // SAM3 grounding of people, shadows, robots and the sky, used by the masks.
  {
    Models models(root);
    ImagePreprocessor pixels(models.Stream());
    pixels.Upload(rgb);
    const int64_t size = manifest.at("grounding_size"),
                  prompts = int64_t(manifest.at("grounding_prompts").size()),
                  k = manifest.at("grounding_top_k"), cells = manifest.at("grounding_mask_size");
    const auto outputs =
        models.Predict("grounding", {pixels.Normalize(int(size), int(size), {.5f, .5f, .5f},
                                                      {.5f, .5f, .5f})});
    if (outputs.size() != 4)
      throw std::runtime_error("Model verification: invalid grounding outputs");
    check(outputs[0], {prompts, k});
    check(outputs[1], {prompts, k, 4});
    check(outputs[2], {prompts, k, cells, cells});
    check(outputs[3], {prompts, 1, cells, cells});
    Json detected = Json::object();
    for (int64_t p = 0; p < prompts; ++p) {
      const float* scores = outputs[0].Floats() + p * k;
      size_t confident = 0;
      for (int64_t q = 0; q < k; ++q) {
        if (scores[q] < 0 || scores[q] > 1 || (q && scores[q] > scores[q - 1]))
          throw std::runtime_error("Model verification: grounding scores out of range or order");
        confident += scores[q] >= kGroundingScore;
      }
      detected[manifest.at("grounding_prompts").at(p).get<std::string>()] = confident;
    }
    report["grounding_detections"] = detected;
    report["inference"]["grounding"] = models.Profile().at("grounding");
  }
  return report;
}
Json MakeMasks(const Scene& scene, const fs::path& work, const fs::path& root, int workers) {
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
  // SAM3 decides the sky, people and the shadows next to them; the rest of the domain is static.
  const int grounding_size = GroundingSize(models.Manifest());
  const int mask_size = models.Manifest().at("grounding_mask_size");
  const std::array<cv::Mat, 2> domains{scene.cameras[0].Mask(), scene.cameras[1].Mask()};
  const std::array<cv::Rect, 2> rois{GroundingRoi(scene.cameras[0], domains[0]),
                                     GroundingRoi(scene.cameras[1], domains[1])};
  const cv::Mat rim_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * kRimBandCells + 1, 2 * kRimBandCells + 1});
  std::array<cv::Mat, 2> roi_domains_coarse, rim_zones;
  for (int c = 0; c < 2; ++c) {
    cv::Mat resized;
    cv::resize(domains[c](rois[c]), resized, {mask_size, mask_size}, 0, 0, cv::INTER_AREA);
    roi_domains_coarse[c] = resized >= 128;
    cv::Mat inner;
    cv::erode(roi_domains_coarse[c], inner, rim_kernel, {-1, -1}, 1, cv::BORDER_CONSTANT,
              cv::Scalar(0));
    rim_zones[c] = (roi_domains_coarse[c] == 0) | ((roi_domains_coarse[c] > 0) & (inner == 0));
  }
  GroundingCounts counts;
  ImageReader reader(pending, work, grounding_size, rois,
                     {cv::Size(scene.cameras[0].width, scene.cameras[0].height),
                      cv::Size(scene.cameras[1].width, scene.cameras[1].height)});
  WriterPool writer(workers);
  for (const auto& record : pending) {
    auto phase = Clock::now();
    auto image = reader.Next();
    timing.wait += Seconds(phase);
    timing.decode += image.decode;
    timing.preprocess += image.preprocess;
    phase = Clock::now();
    pixels.Upload(image.grounding);
    const auto outputs = models.Predict(
        "grounding",
        {pixels.Normalize(grounding_size, grounding_size, {.5f, .5f, .5f}, {.5f, .5f, .5f})});
    timing.gpu_pipeline += Seconds(phase);
    phase = Clock::now();
    auto coarse = GroundingMaskCoarse(outputs, roi_domains_coarse[record.camera],
                                      rim_zones[record.camera], counts);
    timing.postprocess += Seconds(phase);
    const auto path = MaskPath(work, record.id);
    const cv::Mat* domain = &domains[record.camera];
    const cv::Rect roi = rois[record.camera];
    writer.Submit([path, coarse = std::move(coarse), domain, roi] {
      const cv::Mat mask = ComposeFullMask(coarse, *domain, roi);
      const auto temporary = path.parent_path() / (path.stem().u8string() + ".writing.png");
      WriteImage(temporary, mask, 1);
      PublishFile(temporary, path);
    });
    ++timing.computed;
    if (record.id % 16 == 0)
      Log("Masks " + std::to_string(record.id + 1) + "/" + std::to_string(scene.images.size()));
  }
  writer.Finish();
  timing.write = writer.seconds;
  auto report = timing.Report(started, &models);
  Json roi_json = Json::array();
  for (const auto& roi : rois)
    roi_json.push_back({roi.x, roi.y, roi.width, roi.height});
  report["grounding"] = {{"score_threshold", kGroundingScore},
                         {"rim_person_score_threshold", kRimPersonScore},
                         {"shadow_score_threshold", kShadowScore},
                         {"shadow_reach_cells", kShadowReach},
                         {"shadow_max_full_share", kShadowMaxFullShare},
                         {"shadow_small_share", kShadowSmallShare},
                         {"shadow_max_person_area_ratio", kShadowMaxPersonAreaRatio},
                         {"rois", roi_json},
                         {"persons", counts.persons},
                         {"rim_persons", counts.rim_persons},
                         {"persons_kept_as_robots", counts.robots},
                         {"accessory_bridges", counts.accessory_bridges},
                         {"filled_pockets", counts.filled_pockets},
                         {"shadows", counts.shadows},
                         {"shadows_clipped", counts.clipped_shadows},
                         {"shadows_far_from_persons", counts.distant_shadows},
                         {"skies", counts.skies},
                         {"images_with_sky", counts.sky_images}};
  return report;
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
  KeypointDetector keypoint_detector;
  ImageReader reader(pending, work);
  WriterPool writer;
  for (const auto& record : pending) {
    const auto& camera = scene.cameras[record.camera];
    const auto size = models.DetectorSize(camera.width, camera.height);
    const int width = size[0], height = size[1];
    auto phase = Clock::now();
    auto image = reader.Next();
    timing.wait += Seconds(phase);
    timing.decode += image.decode;
    timing.preprocess += image.preprocess;
    phase = Clock::now();
    pixels.Upload(image.rgb);
    const auto detected = models.Predict("detector", {pixels.Normalize(width, height)}, true)[0];
    const auto raw = keypoint_detector.Detect(detected, width, height, 12288, models.Stream());
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
namespace {
// Pairs to match: the cached ones, per-lens near exposures, closest cross-lens exposures,
// navigation revisits and descriptor retrieval over several exposure distances. Repair adds dense
// edges for the requested images only.
std::set<std::pair<uint32_t, uint32_t>> CandidatePairs(const Scene& scene, const fs::path& work,
                                                       const std::set<uint32_t>& repair,
                                                       size_t& navigation_added_pairs) {
  const uint32_t n = uint32_t(scene.images.size());
  Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor> retrieval_mat(n, 256);
  for (uint32_t id = 0; id < n; ++id) {
    const auto r = ReadFeatures(FeaturePath(work, id)).retrieval;
    std::copy_n(r.data(), 256, retrieval_mat.row(id).data());
  }
  std::array<std::vector<uint32_t>, 2> lens_images;
  std::vector<size_t> lens_rank(n, 0);
  for (uint32_t id = 0; id < n; ++id) {
    const uint32_t cam = scene.images[id].camera;
    lens_rank[id] = lens_images.at(cam).size();
    lens_images.at(cam).push_back(id);
  }
  std::set<std::pair<uint32_t, uint32_t>> pairs;
  const auto positions = NavigationPositions(scene);
  const auto velocities = NavigationVelocities(scene);
  navigation_added_pairs = 0;
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
    const uint32_t cam = scene.images[id].camera;
    // Same-lens temporal neighbors: interleaved L/R ordering halves a global index span, so step
    // along each lens independently. Keep short-window pairs even when backing out of a dead end.
    const auto& same = lens_images.at(cam);
    const size_t rank = lens_rank[id], same_span = repair.empty() ? 8 : 24;
    for (size_t idx = rank > same_span ? rank - same_span : 0;
         idx < std::min(same.size(), rank + same_span + 1); ++idx) {
      const size_t step = rank > idx ? rank - idx : idx - rank;
      if (step == 0)
        continue;
      if (repair.empty() &&
          ((step > 5 && step != 8) ||
           (positions[id] - positions[same[idx]]).squaredNorm() > 15. * 15.))
        continue;
      add(id, same[idx]);
    }
    // Cross-lens temporal neighbors by exposure timestamp, without assuming L/R synchronization.
    const auto& opposite = lens_images.at(1 - cam);
    const double max_dt = repair.empty() ? 2.5 : 12.0;
    std::vector<std::pair<double, uint32_t>> cross;
    for (uint32_t k : opposite) {
      const double abs_dt = std::abs(ExposureDeltaSeconds(scene.images[id], scene.images[k]));
      if (abs_dt > max_dt)
        continue;
      if (repair.empty() && (positions[id] - positions[k]).squaredNorm() > 15. * 15.)
        continue;
      cross.emplace_back(abs_dt, k);
    }
    std::sort(cross.begin(), cross.end());
    for (size_t i = 0; i < std::min<size_t>(repair.empty() ? 4 : 12, cross.size()); ++i)
      add(id, cross[i].second);
    // Revisited surfaces can have poor global descriptor similarity because
    // their scale and the surrounding scene changed. Navigation supplies
    // independent candidates; matching and ray-space verification still decide
    // whether the images actually overlap. Do not assume L/R synchronization.
    for (uint32_t candidate : NavigationRevisitCandidates(scene, positions, velocities, id)) {
      const auto before = pairs.size();
      add(id, candidate);
      navigation_added_pairs += pairs.size() - before;
    }
    const Eigen::VectorXf sims = retrieval_mat * retrieval_mat.row(id).transpose();
    if (repair.empty()) {
      std::vector<std::pair<double, uint32_t>> loop_options, mid_options;
      for (uint32_t j = 0; j < n; ++j) {
        const bool same_cam = cam == scene.images[j].camera;
        const size_t same_step =
            lens_rank[id] > lens_rank[j] ? lens_rank[id] - lens_rank[j] : lens_rank[j] - lens_rank[id];
        const double abs_dt = std::abs(ExposureDeltaSeconds(scene.images[id], scene.images[j]));
        if ((same_cam ? (same_step <= 5 || same_step == 8) : abs_dt <= 2.5) ||
            (abs_dt > 10. && FacesAway(scene, velocities, id, j)) ||
            (positions[id] - positions[j]).squaredNorm() > 20. * 20. ||
            pairs.count({std::min(id, j), std::max(id, j)}))
          continue;
        const double similarity = double(sims[j]);
        (abs_dt > 30. ? loop_options : mid_options).emplace_back(similarity, j);
      }
      std::sort(loop_options.rbegin(), loop_options.rend());
      std::sort(mid_options.rbegin(), mid_options.rend());
      std::vector<uint32_t> selected;
      auto pick_from = [&](const std::vector<std::pair<double, uint32_t>>& options,
                           size_t target_total) {
        for (size_t k = 0; k < std::min<size_t>(8, options.size()) && selected.size() < target_total;
             ++k) {
          const auto j = options[k].second;
          if (std::find(selected.begin(), selected.end(), j) != selected.end())
            continue;
          const auto& candidate = scene.images[j];
          const bool duplicate = std::any_of(selected.begin(), selected.end(), [&](uint32_t p) {
            const auto& previous = scene.images[p];
            const double seconds = ExposureDeltaSeconds(candidate, previous);
            return candidate.camera == previous.camera && std::abs(seconds) < 2.;
          });
          if (duplicate)
            continue;
          selected.push_back(j);
          add(id, j);
        }
      };
      pick_from(loop_options, 2);
      pick_from(mid_options, 3);
      pick_from(loop_options, 3);
    } else {
      for (const auto range : std::vector<std::array<size_t, 2>>{{6, 24}, {24, 80}, {80, n}}) {
        std::vector<std::pair<double, uint32_t>> options;
        for (uint32_t j = 0; j < n; ++j) {
          const bool same_cam = cam == scene.images[j].camera;
          const size_t same_step = lens_rank[id] > lens_rank[j] ? lens_rank[id] - lens_rank[j]
                                                                 : lens_rank[j] - lens_rank[id];
          const size_t distance = same_cam ? same_step : size_t(id > j ? id - j : j - id) / 2;
          const double abs_dt = std::abs(ExposureDeltaSeconds(scene.images[id], scene.images[j]));
          if (distance < range[0] || distance >= range[1] ||
              (abs_dt > 10. && FacesAway(scene, velocities, id, j)))
            continue;
          const double similarity = double(sims[j]);
          options.emplace_back(similarity, j);
        }
        std::sort(options.rbegin(), options.rend());
        for (size_t k = 0; k < std::min<size_t>(12, options.size()); ++k)
          add(id, options[k].second);
      }
    }
  }
  return pairs;
}
}  // namespace
Json MakeMatches(const Scene& scene, const fs::path& work, const fs::path& root,
                 const std::set<uint32_t>& repair) {
  const auto started = Clock::now();
  FrontendTiming timing;
  const uint32_t n = uint32_t(scene.images.size());
  size_t navigation_added_pairs;
  const auto pairs = CandidatePairs(scene, work, repair, navigation_added_pairs);
  const auto archive_path = MatchesPath(work);
  const auto existing = IndexMatchArchive(archive_path, true);
  std::vector<std::pair<uint32_t, uint32_t>> pending;
  for (const auto& [a, b] : pairs) {
    if (existing.count({a, b}))
      ++timing.reused;
    else
      pending.emplace_back(a, b);
  }
  Json report;
  if (!pending.empty()) {
    std::sort(pending.begin(), pending.end(), [](const auto& lhs, const auto& rhs) {
      constexpr uint32_t Tile = 32;
      const uint32_t la = lhs.first / Tile, ra = rhs.first / Tile;
      if (la != ra)
        return la < ra;
      const uint32_t lb = lhs.second / Tile, rb = rhs.second / Tile;
      if (lb != rb)
        return (la & 1u) ? (lb > rb) : (lb < rb);
      return lhs < rhs;
    });
    Models models(root);
    FeatureCache cache(work);
    ArchiveWriterQueue writer(archive_path);
    int max_batch = models.MaxBatch("matcher");
    if (max_batch > 4) {
      size_t free_bytes = 0, total_bytes = 0;
      CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
      if (free_bytes < size_t(5) * 1024 * 1024 * 1024)
        max_batch = 4;
    }
    max_batch = std::max(1, max_batch);
    constexpr int64_t capacity = 3072;
    constexpr size_t kKpBytes = size_t(capacity) * 2 * sizeof(float);
    constexpr size_t kDescBytes = size_t(capacity) * 256 * sizeof(float);
    constexpr size_t kMaskBytes = size_t(capacity) * sizeof(float);
    std::array<std::shared_ptr<DeviceBuffer>, 6> batch_bufs;
    if (max_batch > 1) {
      const size_t b = size_t(max_batch);
      batch_bufs[0] = std::make_shared<DeviceBuffer>(b * kKpBytes);
      batch_bufs[1] = std::make_shared<DeviceBuffer>(b * kKpBytes);
      batch_bufs[2] = std::make_shared<DeviceBuffer>(b * kDescBytes);
      batch_bufs[3] = std::make_shared<DeviceBuffer>(b * kDescBytes);
      batch_bufs[4] = std::make_shared<DeviceBuffer>(b * kMaskBytes);
      batch_bufs[5] = std::make_shared<DeviceBuffer>(b * kMaskBytes);
    }
    for (size_t index = 0; index < pending.size();) {
      const size_t batch = std::min<size_t>(size_t(max_batch), pending.size() - index);
      for (size_t next = index + batch; next < std::min(pending.size(), index + batch + 8); ++next) {
        cache.Prefetch(pending[next].first);
        cache.Prefetch(pending[next].second);
      }
      if (batch == 1) {
        const auto [a, b] = pending[index];
        const auto& fa = cache.Get(a);
        const auto& fb = cache.Get(b);
        const auto na = int64_t(fa.value.pixels.size()), nb = int64_t(fb.value.pixels.size());
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
        writer.Submit(std::move(m));
      } else {
        std::vector<size_t> na_list(batch), nb_list(batch);
        const auto stream = models.Stream();
        auto phase_copy = Clock::now();
        for (size_t bi = 0; bi < batch; ++bi) {
          const auto [a, b] = pending[index + bi];
          const auto& fa = cache.Get(a);
          const auto& fb = cache.Get(b);
          na_list[bi] = fa.value.pixels.size();
          nb_list[bi] = fb.value.pixels.size();
          auto copy_slot = [&](const std::shared_ptr<DeviceBuffer>& dst,
                               const std::shared_ptr<DeviceBuffer>& src, size_t bytes) {
            CudaCheck(cudaMemcpyAsync(static_cast<uint8_t*>(dst->data) + bi * bytes, src->data,
                                      bytes, cudaMemcpyDeviceToDevice, stream));
          };
          copy_slot(batch_bufs[0], fa.keypoints.device, kKpBytes);
          copy_slot(batch_bufs[1], fb.keypoints.device, kKpBytes);
          copy_slot(batch_bufs[2], fa.descriptors.device, kDescBytes);
          copy_slot(batch_bufs[3], fb.descriptors.device, kDescBytes);
          copy_slot(batch_bufs[4], fa.mask.device, kMaskBytes);
          copy_slot(batch_bufs[5], fb.mask.device, kMaskBytes);
        }
        timing.gpu_pipeline += Seconds(phase_copy);
        auto make_tensor = [&](int slot, std::vector<int64_t> shape) {
          Tensor t;
          t.shape = std::move(shape);
          t.type = 0;
          t.device = batch_bufs[slot];
          return t;
        };
        const int64_t bn = int64_t(batch);
        auto phase = Clock::now();
        const auto output = models.Predict(
            "matcher",
            {make_tensor(0, {bn, capacity, 2}), make_tensor(1, {bn, capacity, 2}),
             make_tensor(2, {bn, capacity, 256}), make_tensor(3, {bn, capacity, 256}),
             make_tensor(4, {bn, capacity}), make_tensor(5, {bn, capacity})});
        timing.gpu_pipeline += Seconds(phase);
        phase = Clock::now();
        if (output.size() != 2 || output[0].shape != std::vector<int64_t>{bn, capacity} ||
            output[1].shape != std::vector<int64_t>{bn, capacity})
          throw std::runtime_error("Unexpected batched matcher outputs");
        const float* confs = output[1].Floats();
        for (size_t bi = 0; bi < batch; ++bi) {
          const auto [a, b] = pending[index + bi];
          const size_t na = na_list[bi], nb = nb_list[bi];
          const size_t base = bi * size_t(capacity);
          MatchSet m{a, b, {}};
          for (uint32_t k = 0; k < uint32_t(na); ++k) {
            const auto j = output[0].Index(base + k);
            if (j >= 0) {
              if (j >= int64_t(nb))
                throw std::runtime_error("Matcher returned invalid feature index");
              m.matches.push_back({k, uint32_t(j), confs[base + k]});
            }
          }
          writer.Submit(std::move(m));
        }
        timing.postprocess += Seconds(phase);
      }
      const size_t prev_bucket = index / 500;
      index += batch;
      timing.computed += batch;
      if (index / 500 > prev_bucket)
        Log("Matches " + std::to_string(index) + "/" + std::to_string(pending.size()));
    }
    writer.Finish();
    timing.write = writer.seconds;
    timing.decode = cache.read_seconds;
    timing.wait = cache.wait_seconds;
    timing.preprocess = cache.upload_seconds;
    report = timing.Report(started, &models);
    report["matcher_max_batch"] = max_batch;
  } else {
    if (!fs::exists(archive_path))
      MatchArchiveWriter(archive_path).Flush();
    report = timing.Report(started);
  }
  Json list = Json::array();
  std::vector<double> degrees(n, 0);
  for (const auto& [a, b] : pairs) {
    list.push_back({a, b});
    ++degrees[a];
    ++degrees[b];
  }
  WriteJson(work / "cache/pairs.json", {{"pairs", list}});
  report["pairs"] = pairs.size();
  report["navigation_added_pairs"] = navigation_added_pairs;
  report["pair_degree"] = Statistics(degrees);
  report["pair_degree"]["minimum"] = *std::min_element(degrees.begin(), degrees.end());
  report["repair_images"] = repair;
  return report;
}
}  // namespace geosun

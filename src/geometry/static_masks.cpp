#include "geometry/static_masks.h"
#include "geometry/supervision.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <thread>

namespace geosun {
namespace {
constexpr int MapSize = 1024, MinArea = 16, MinSamples = 12;
// Masks of all images are kept at 1/ThumbScale of the photo to look up what each view masked.
constexpr int ThumbScale = 6;
// A candidate is static when the place it covers is flagged transient in a sizeable share of the
// views that see that place, over a long time: a picture, poster or seated person is flagged in
// every view that really sees it, a walking person only while passing, the operator only where the
// rig is. Visibility is geometric only (no occlusion test: views from the next room count), which
// is why the share of a true picture is about 0.4 and the minimum is low.
constexpr double MinRatio = .2, MinSpan = 40;
constexpr int MinSeen = 8;
// Views see the place from 0.8 m to MaxRange within this share of the lens's valid radius.
constexpr double MaxRange = 12, ValidShare = .8;
// A static candidate this close (fraction of the image width) to one that stays masked stays
// masked: a shadow darkens the floor in every view the same way, so it must follow its person.
constexpr double ShadowReach = .08;
// Sky verification: erode each 1024-grid sky component so LAS edge spread on trees or roofs does
// not enter thin outdoor gaps, then check whether the interior has dense LAS depth returns.
constexpr int SkyCoreErosion = 2;
constexpr double SkyLasShare = .5, SkyImageMinShare = .005, SkySceneMinImageShare = .05;
constexpr size_t SkySceneMinImages = 8;

struct Candidate {
  uint32_t image = 0;
  int label = 0, area = 0;
  double x = 0, y = 0;
  bool located = false;
  Eigen::Vector3d place = Eigen::Vector3d::Zero();
  size_t seen = 0, flagged = 0;
  double span = 0;
  std::string reason;
};
struct SkyComponent {
  int label = 0, area = 0, core_area = 0, core_las = 0;
  bool surface = false;
};
struct SkyImage {
  bool has_raw_sky = false;
  int valid_grid_pixels = 0;
  int open_sky_grid_pixels = 0;
  std::vector<SkyComponent> components;
};
struct Context {
  const Scene& scene;
  const Reconstruction& model;
  const ShutterMotion* motion = nullptr;
  const std::function<fs::path(uint32_t)> &source, &destination;
  std::array<cv::Mat, 2> domains;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<Eigen::Vector3d> translations, centers;
  std::vector<double> seconds;
  std::vector<cv::Mat> thumbs;  // transient share of each view, 1/ThumbScale
  std::vector<std::vector<Candidate>> candidates;
  std::array<int, 2> full_domain_grid_pixels{0, 0};
  std::vector<SkyImage> skies;
  bool keep_sky = true;
};
struct Labelled {
  cv::Mat mask, transient_map, labels, stats, centroids;
  int count = 0;
};
// Transient components of an image's mask, on the given grid.
Labelled Label(const Context& context, uint32_t id, cv::Size grid) {
  const auto& name = context.scene.images[id].name;
  const auto& camera = context.model.cameras[context.model.images[id].camera];
  Labelled out;
  out.mask = ReadImage(context.source(id), cv::IMREAD_GRAYSCALE);
  if (out.mask.cols != camera.width || out.mask.rows != camera.height)
    throw std::runtime_error("Mask size differs from its camera: " + name);
  const cv::Mat& domain = context.domains[context.model.images[id].camera];
  cv::resize((out.mask == 0) & domain, out.transient_map, grid, 0, 0, cv::INTER_NEAREST);
  out.count = cv::connectedComponentsWithStats(out.transient_map, out.labels, out.stats,
                                               out.centroids, 8);
  return out;
}
// Pass 1, per image: where each transient candidate is in the world (median LAS point of it).
void Collect(Context& context, uint32_t id, const cv::Mat& depth) {
  const auto& model = context.model;
  const auto& camera = model.cameras[model.images[id].camera];
  auto labelled = Label(context, id, depth.size());
  const double scale_x = double(camera.width) / depth.cols, scale_y = double(camera.height) / depth.rows;
  const cv::Mat& domain = context.domains[model.images[id].camera];
  cv::Mat transient;
  cv::resize((labelled.mask == 0) & domain, transient,
             {camera.width / ThumbScale, camera.height / ThumbScale}, 0, 0, cv::INTER_AREA);
  context.thumbs[id] = transient >= 80;
  std::vector<Candidate> found;
  for (int k = 1; k < labelled.count; ++k) {
    Candidate c;
    c.image = id;
    c.label = k;
    c.area = labelled.stats.at<int>(k, cv::CC_STAT_AREA);
    if (c.area < MinArea)
      continue;
    c.x = labelled.centroids.at<double>(k, 0);
    c.y = labelled.centroids.at<double>(k, 1);
    const cv::Rect box(labelled.stats.at<int>(k, cv::CC_STAT_LEFT),
                       labelled.stats.at<int>(k, cv::CC_STAT_TOP),
                       labelled.stats.at<int>(k, cv::CC_STAT_WIDTH),
                       labelled.stats.at<int>(k, cv::CC_STAT_HEIGHT));
    std::vector<cv::Point> pixels;
    cv::findNonZero(labelled.labels(box) == k, pixels);
    std::array<std::vector<double>, 3> coordinates;
    for (const auto& p : pixels) {
      const double d = depth.at<float>(p.y + box.y, p.x + box.x);
      if (d <= 0)
        continue;
      const Eigen::Vector2d pixel{(p.x + box.x + .5) * scale_x, (p.y + box.y + .5) * scale_y};
      const auto ray = camera.Ray(pixel);
      const Eigen::Vector3d moved = ray * d;
      const Eigen::Vector3d cam_point =
          context.motion ? context.motion->Reference(
                               id, model.images[id].camera,
                               context.motion->Fraction(camera, pixel), moved)
                         : moved;
      const Eigen::Vector3d world =
          context.rotations[id].transpose() * (cam_point - context.translations[id]);
      for (int a = 0; a < 3; ++a)
        coordinates[a].push_back(world[a]);
    }
    if (coordinates[0].size() < size_t(MinSamples) || coordinates[0].size() < pixels.size() * 3 / 10) {
      c.reason = "no_las";
    } else {
      for (int a = 0; a < 3; ++a) {
        auto& values = coordinates[a];
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        c.place[a] = values[values.size() / 2];
      }
      c.located = true;
    }
    found.push_back(std::move(c));
  }
  context.candidates[id] = std::move(found);
  const cv::Mat raw_sky = (labelled.mask == 128);
  if (cv::countNonZero(raw_sky) > 0) {
    SkyImage sky_info;
    sky_info.has_raw_sky = true;
    sky_info.valid_grid_pixels = context.full_domain_grid_pixels[model.images[id].camera];
    cv::Mat sky_down;
    cv::resize(raw_sky, sky_down, depth.size(), 0, 0, cv::INTER_AREA);
    cv::Mat sky_labels, sky_stats, sky_centroids;
    const int sky_count =
        cv::connectedComponentsWithStats(sky_down > 0, sky_labels, sky_stats, sky_centroids, 8);
    const cv::Mat sky_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * SkyCoreErosion + 1, 2 * SkyCoreErosion + 1});
    for (int k = 1; k < sky_count; ++k) {
      SkyComponent comp;
      comp.label = k;
      comp.area = sky_stats.at<int>(k, cv::CC_STAT_AREA);
      const cv::Rect box(sky_stats.at<int>(k, cv::CC_STAT_LEFT),
                         sky_stats.at<int>(k, cv::CC_STAT_TOP),
                         sky_stats.at<int>(k, cv::CC_STAT_WIDTH),
                         sky_stats.at<int>(k, cv::CC_STAT_HEIGHT));
      if (comp.area >= MinArea && box.width > 2 * SkyCoreErosion &&
          box.height > 2 * SkyCoreErosion) {
        const cv::Mat patch = (sky_labels(box) == k);
        cv::Mat core;
        cv::erode(patch, core, sky_kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
                  cv::Scalar(0));
        comp.core_area = cv::countNonZero(core);
        if (comp.core_area >= MinArea) {
          std::vector<cv::Point> core_pixels;
          cv::findNonZero(core, core_pixels);
          for (const auto& p : core_pixels)
            if (depth.at<float>(p.y + box.y, p.x + box.x) > 0.f)
              ++comp.core_las;
          if (comp.core_las >= MinSamples &&
              double(comp.core_las) >= SkyLasShare * double(comp.core_area))
            comp.surface = true;
        }
      }
      if (!comp.surface)
        sky_info.open_sky_grid_pixels += comp.area;
      sky_info.components.push_back(comp);
    }
    context.skies[id] = std::move(sky_info);
  }
}
// Pass 2: how many of the views that see a candidate's place flagged it, and over what time.
void Judge(Context& context, Candidate& c) {
  const auto& model = context.model;
  double first = context.seconds[c.image], last = first;
  for (uint32_t b = 0; b < model.images.size(); ++b) {
    if (b == c.image || !model.images[b].registered)
      continue;
    const Eigen::Vector3d seen = context.rotations[b] * c.place + context.translations[b];
    const auto& other = model.cameras[model.images[b].camera];
    if (seen.z() <= .3 || seen.norm() < .8 || seen.norm() > MaxRange)
      continue;
    const auto pixel = other.Project(seen);
    const double dx = (pixel.x() - other.params[2]) / other.params[0],
                 dy = (pixel.y() - other.params[3]) / other.params[1];
    if (dx * dx + dy * dy > std::pow(other.Radius(), 2) * ValidShare * ValidShare || pixel.x() < 0 ||
        pixel.y() < 0 || pixel.x() >= other.width || pixel.y() >= other.height)
      continue;
    ++c.seen;
    const cv::Mat& thumb = context.thumbs[b];
    const int tx = int(pixel.x() / ThumbScale), ty = int(pixel.y() / ThumbScale);
    bool flagged = false;
    for (int y = std::max(0, ty - 1); y <= std::min(thumb.rows - 1, ty + 1) && !flagged; ++y)
      for (int x = std::max(0, tx - 1); x <= std::min(thumb.cols - 1, tx + 1); ++x)
        if (thumb.at<uint8_t>(y, x)) {
          flagged = true;
          break;
        }
    if (flagged) {
      ++c.flagged;
      first = std::min(first, context.seconds[b]);
      last = std::max(last, context.seconds[b]);
    }
  }
  c.span = last - first;
  if (c.seen < size_t(MinSeen))
    c.reason = "few_views";
  else if (double(c.flagged) / double(c.seen) >= MinRatio && c.span >= MinSpan)
    c.reason = "static";
  else
    c.reason = "dynamic";
}
struct ApplyCounts {
  uint64_t person_unmasked = 0;
  uint64_t sky_cleared = 0;
};
// Pass 3, per image: static candidates become static in the mask, unless a candidate that stays
// masked is next to them; false-positive sky components and indoor skies become static too.
ApplyCounts Apply(Context& context, uint32_t id, cv::Size grid) {
  const auto& model = context.model;
  const auto& camera = model.cameras[model.images[id].camera];
  auto& found = context.candidates[id];
  auto labelled = Label(context, id, grid);
  ApplyCounts counts;
  const bool any = std::any_of(found.begin(), found.end(),
                               [](const Candidate& c) { return c.reason == "static"; });
  if (any) {
    cv::Mat stays = labelled.transient_map.clone();
    for (const auto& c : found)
      if (c.reason == "static")
        stays.setTo(0, labelled.labels == c.label);
    const int reach = int(ShadowReach * grid.width);
    cv::dilate(stays, stays,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * reach + 1, 2 * reach + 1}));
    cv::Mat restored = cv::Mat::zeros(grid, CV_8UC1);
    for (auto& c : found) {
      if (c.reason != "static")
        continue;
      const cv::Mat region = labelled.labels == c.label;
      if (cv::countNonZero(stays & region))
        c.reason = "static_near_masked";
      else
        restored.setTo(255, region);
    }
    if (cv::countNonZero(restored)) {
      cv::Mat region;
      cv::resize(restored, region, {camera.width, camera.height}, 0, 0, cv::INTER_NEAREST);
      cv::dilate(region, region, cv::Mat::ones(5, 5, CV_8UC1));
      const cv::Mat change = region & (labelled.mask == 0) & context.domains[model.images[id].camera];
      counts.person_unmasked = uint64_t(cv::countNonZero(change));
      labelled.mask.setTo(255, change);
    }
  }
  const auto& sky_info = context.skies[id];
  if (sky_info.has_raw_sky) {
    const cv::Mat raw_sky = (labelled.mask == 128);
    if (!context.keep_sky) {
      counts.sky_cleared = uint64_t(cv::countNonZero(raw_sky));
      labelled.mask.setTo(255, raw_sky);
    } else if (std::any_of(sky_info.components.begin(), sky_info.components.end(),
                           [](const SkyComponent& c) { return c.surface; })) {
      cv::Mat sky_down;
      cv::resize(raw_sky, sky_down, grid, 0, 0, cv::INTER_AREA);
      cv::Mat sky_labels, sky_stats, sky_centroids;
      cv::connectedComponentsWithStats(sky_down > 0, sky_labels, sky_stats, sky_centroids, 8);
      cv::Mat sky_restored = cv::Mat::zeros(grid, CV_8UC1);
      for (const auto& comp : sky_info.components)
        if (comp.surface)
          sky_restored.setTo(255, sky_labels == comp.label);
      cv::dilate(sky_restored, sky_restored, cv::Mat::ones(3, 3, CV_8UC1));
      cv::Mat full_sky_restored;
      cv::resize(sky_restored, full_sky_restored, {camera.width, camera.height}, 0, 0,
                 cv::INTER_NEAREST);
      const cv::Mat sky_change = full_sky_restored & raw_sky;
      counts.sky_cleared = uint64_t(cv::countNonZero(sky_change));
      labelled.mask.setTo(255, sky_change);
    }
  }
  const auto path = context.destination(id);
  fs::create_directories(path.parent_path());
  WriteImage(path, labelled.mask);
  return counts;
}
}  // namespace
Json StaticMaskRules() {
  return {{"flagged_share_minimum", MinRatio},
          {"span_seconds_minimum", MinSpan},
          {"views_seeing_minimum", MinSeen},
          {"max_range_m", MaxRange},
          {"shadow_reach", ShadowReach},
          {"sky_core_erosion_pixels", SkyCoreErosion},
          {"sky_las_share_minimum", SkyLasShare},
          {"sky_image_share_minimum", SkyImageMinShare},
          {"sky_scene_image_share_minimum", SkySceneMinImageShare},
          {"sky_scene_images_minimum", SkySceneMinImages}};
}
Json StaticMasks(const Scene& scene, const Reconstruction& model, const ShutterMotion* motion,
                 const LasHeader& header, LasSurfelCache& cache,
                 const std::function<fs::path(uint32_t)>& source,
                 const std::function<fs::path(uint32_t)>& destination) {
  const auto started = Clock::now();
  Context context{scene, model, motion, source, destination, {}, {}, {}, {}, {}, {}, {}, {0, 0},
                  {}, true};
  context.thumbs.resize(model.images.size());
  context.candidates.resize(model.images.size());
  context.skies.resize(model.images.size());
  for (int c = 0; c < 2; ++c) {
    cv::Mat domain = model.cameras[c].Mask();
    const auto grid_cam = MapCamera(model.cameras[c], MapSize);
    cv::Mat grid_domain;
    cv::resize(domain, grid_domain, {grid_cam.width, grid_cam.height}, 0, 0, cv::INTER_NEAREST);
    context.full_domain_grid_pixels[c] = cv::countNonZero(grid_domain);
    cv::erode(domain, domain, cv::getStructuringElement(cv::MORPH_ELLIPSE, {31, 31}));
    context.domains[c] = domain;
  }
  for (const auto& image : model.images) {
    context.rotations.push_back(image.pose.Rotation().toRotationMatrix());
    context.translations.push_back(image.pose.Translation());
    context.centers.push_back(image.pose.Center());
  }
  // Exposure times from the names: week-second-millisecond_frame-lens.
  for (const auto& image : scene.images) {
    const auto file = fs::u8path(image.name).filename().u8string();
    int week, second, millisecond;
    if (std::sscanf(file.c_str(), "%d-%d-%d_", &week, &second, &millisecond) != 3)
      throw std::runtime_error("Cannot read the exposure time from " + image.name);
    context.seconds.push_back(week * 604800. + second + millisecond / 1000.);
  }
  // Pass 1: every image's candidates and their places, while the LAS is projected.
  std::deque<std::future<void>> jobs;
  size_t images = 0;
  ProjectLasDepths(cache, header, model, motion, MapSize,
                   [&](uint32_t id, cv::Mat&& depth) {
                     if (jobs.size() >= 6) {
                       jobs.front().get();
                       jobs.pop_front();
                     }
                     jobs.push_back(std::async(std::launch::async,
                                               [&context, id, depth = std::move(depth)] {
                                                 Collect(context, id, depth);
                                               }));
                     if (++images % 64 == 0)
                       Log("Static masks: located " + std::to_string(images) + "/" +
                           std::to_string(model.Registered()));
                   });
  for (auto& job : jobs)
    job.get();
  // Pass 2: judge every candidate against all views.
  const int workers = std::clamp(int(std::thread::hardware_concurrency()), 1, 16);
  std::vector<std::pair<uint32_t, size_t>> work;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    for (size_t k = 0; k < context.candidates[i].size(); ++k)
      if (context.candidates[i][k].located)
        work.emplace_back(i, k);
  ParallelFor(work.size(), workers, 4, [&](size_t i) {
    Judge(context, context.candidates[work[i].first][work[i].second]);
  });
  // Scene-level sky decision before Pass 3.
  std::vector<uint32_t> written;
  for (uint32_t i = 0; i < model.images.size(); ++i)
    if (model.images[i].registered)
      written.push_back(i);
  size_t raw_sky_images = 0, open_sky_images = 0, surface_sky_components = 0;
  for (const auto id : written) {
    const auto& s = context.skies[id];
    if (!s.has_raw_sky)
      continue;
    ++raw_sky_images;
    for (const auto& comp : s.components)
      surface_sky_components += comp.surface ? 1 : 0;
    if (s.valid_grid_pixels > 0 &&
        double(s.open_sky_grid_pixels) >= SkyImageMinShare * double(s.valid_grid_pixels))
      ++open_sky_images;
  }
  const size_t min_open_images = std::min<size_t>(SkySceneMinImages, (written.size() + 1) / 2);
  context.keep_sky =
      open_sky_images >= min_open_images &&
      double(open_sky_images) >= SkySceneMinImageShare * double(written.size());
  // Pass 3: the new masks, on the same grid as pass 1.
  const auto map_camera = [&](uint32_t id) { return MapCamera(model.cameras[model.images[id].camera], MapSize); };
  std::vector<ApplyCounts> applied(model.images.size());
  ParallelFor(written.size(), workers, 1, [&](size_t i) {
    const auto grid = map_camera(written[i]);
    applied[written[i]] = Apply(context, written[i], {grid.width, grid.height});
  });
  Json rows = Json::array();
  std::map<std::string, size_t> reasons;
  uint64_t pixels = 0, sky_pixels = 0;
  size_t changed = 0, sky_changed = 0;
  for (const auto id : written) {
    pixels += applied[id].person_unmasked;
    changed += applied[id].person_unmasked > 0;
    sky_pixels += applied[id].sky_cleared;
    sky_changed += applied[id].sky_cleared > 0;
    for (const auto& c : context.candidates[id]) {
      ++reasons[c.reason];
      Json row = {{"image_index", id},
                  {"name", scene.images[id].name},
                  {"area", c.area},
                  {"x", c.x},
                  {"y", c.y},
                  {"reason", c.reason}};
      if (c.located)
        row["place"] = {c.place.x(), c.place.y(), c.place.z()};
      if (c.seen) {
        row["seen"] = c.seen;
        row["flagged"] = c.flagged;
        row["ratio"] = double(c.flagged) / double(c.seen);
        row["span_seconds"] = c.span;
      }
      rows.push_back(std::move(row));
    }
  }
  Log("Static masks: unmasked " + std::to_string(pixels) + " person pixels (" +
      std::to_string(changed) + " images), sky " +
      std::string(context.keep_sky ? "kept" : "cleared (indoor)") + ", cleared " +
      std::to_string(sky_pixels) + " sky pixels (" + std::to_string(sky_changed) + " images)");
  return {{"images", written.size()},
          {"images_changed", changed},
          {"components", rows.size()},
          {"unmasked_pixels", pixels},
          {"reasons", reasons},
          {"sky",
           {{"kept", context.keep_sky},
            {"images_with_raw_sky", raw_sky_images},
            {"images_with_open_sky", open_sky_images},
            {"surface_components", surface_sky_components},
            {"images_changed", sky_changed},
            {"cleared_pixels", sky_pixels}}},
          {"rules", StaticMaskRules()},
          {"seconds", Seconds(started)},
          {"component_rows", rows}};
}
}  // namespace geosun

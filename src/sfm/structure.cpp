#include "sfm/structure.h"
#include <numeric>
#include <set>

namespace geosun {
namespace {
Json StructureCoverage(const Reconstruction& before, const Reconstruction& after,
                       const std::vector<PlaneConstraint>& planes) {
  std::array<size_t, 2> previous{}, next{};
  for (size_t id = 0; id < before.images.size(); ++id) {
    previous[before.images[id].camera] += before.images[id].registered;
    next[after.images[id].camera] += after.images[id].registered;
  }
  size_t retained = 0;
  for (const auto& plane : planes)
    retained += after.points[plane.point].track.size() >= 3;
  std::vector<uint32_t> parent(after.images.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](uint32_t id) {
    while (parent[id] != id) {
      parent[id] = parent[parent[id]];
      id = parent[id];
    }
    return id;
  };
  for (const auto& point : after.points)
    for (size_t i = 1; i < point.track.size(); ++i)
      parent[root(point.track[i].image)] = root(point.track[0].image);
  std::set<uint32_t> components;
  for (uint32_t id = 0; id < after.images.size(); ++id)
    if (after.images[id].registered)
      components.insert(root(id));
  const bool camera_coverage = next[0] >= .98 * previous[0] && next[1] >= .98 * previous[1];
  return {{"checks",
           {{"registered_image_retention", after.Registered() >= .99 * before.Registered()},
            {"per_camera_image_retention", camera_coverage},
            {"image_graph_connected", components.size() == 1},
            {"las_supported_track_retention", retained >= .9 * planes.size()}}},
          {"registered_images_before", before.Registered()},
          {"registered_images_after", after.Registered()},
          {"camera_images_before", previous},
          {"camera_images_after", next},
          {"las_supported_tracks_before", planes.size()},
          {"las_supported_tracks_after", retained},
          {"limits",
           {{"minimum_image_retention", .99},
            {"minimum_per_camera_retention", .98},
            {"minimum_las_supported_track_retention", .9}}}};
}
}  // namespace
StructureCleanup CleanSparseStructure(const Scene& scene, const Reconstruction& source, int workers,
                                      const AlignmentLimits& limits) {
  const auto started = Clock::now();
  const auto source_quality = source.Quality(scene, limits.median_px, limits.p90_px);
  if (!source_quality.at("passed").get<bool>())
    throw std::runtime_error("Cleanup source reconstruction failed visual acceptance");
  StructureCleanup cleanup;
  cleanup.candidate = source;
  auto& report = cleanup.report;
  const auto planes = LasPlaneConstraints(scene, source, workers);
  if (planes.size() < 200) {
    report = {{"used", false},
              {"reason", "insufficient reliable LAS support for pruning"},
              {"surface_constraints", planes.size()},
              {"seconds", Seconds(started)}};
    return cleanup;
  }
  report = FilterSparseStructure(scene, cleanup.candidate, limits, workers);
  const auto quality = cleanup.candidate.Quality(scene, limits.median_px, limits.p90_px);
  report["coverage"] = StructureCoverage(source, cleanup.candidate, planes);
  std::vector<PlaneConstraint> retained_planes;
  for (const auto& plane : planes)
    if (!cleanup.candidate.points[plane.point].track.empty())
      retained_planes.push_back(plane);
  report["plane_before_m"] = PlaneErrors(source, retained_planes);
  report["plane_after_m"] = PlaneErrors(cleanup.candidate, retained_planes);
  auto checks = report.at("coverage").at("checks");
  checks["visual_acceptance"] = quality.at("passed");
  checks["full_reprojection"] =
      quality.at("reprojection_px").at("p90").is_number() &&
      quality.at("reprojection_px").at("p90").get<double>() <=
          source_quality.at("reprojection_px").at("p90").get<double>() * 1.05 + .05;
  checks["surface_fit"] = !retained_planes.empty() &&
                          report.at("plane_after_m").at("p90").get<double>() <=
                              report.at("plane_before_m").at("p90").get<double>() * 1.25 + .01;
  report["checks"] = checks;
  report["failed_checks"] = FailedChecks(checks);
  cleanup.accepted = report.at("failed_checks").empty();
  report["used"] = cleanup.accepted;
  report["reason"] =
      cleanup.accepted ? "accepted" : "pruning candidate failed coverage or geometry acceptance";
  report["candidate_quality"] = quality;
  report["surface_constraints"] = planes.size();
  report["seconds"] = Seconds(started);
  return cleanup;
}
}  // namespace geosun

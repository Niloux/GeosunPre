#pragma once
#include "geometry/model.h"
#include <set>

namespace geosun {
class RegistrationIncomplete : public std::runtime_error {
 public:
  std::vector<uint32_t> images;
  explicit RegistrationIncomplete(std::vector<uint32_t> ids)
      : std::runtime_error("SfM could not register all selected exposures"),
        images(std::move(ids)) {}
};
class CalibrationCoverageIncomplete : public std::runtime_error {
 public:
  std::vector<uint32_t> images;
  CalibrationCoverageIncomplete(std::vector<uint32_t> ids, std::string reason)
      : std::runtime_error(std::move(reason)), images(std::move(ids)) {}
};
std::set<uint64_t> CalibrationTracks(const Scene& scene, const Reconstruction& model, Json& report);
Reconstruction Reconstruct(const Scene& scene, const fs::path& work, int workers, Json& report,
                           bool stable_calibration = false, bool allow_partial = false,
                           int ba_workers = 0);
}  // namespace geosun

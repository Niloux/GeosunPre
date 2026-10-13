#pragma once
#include "geometry/model.h"

namespace geosun {
class RegistrationIncomplete : public std::runtime_error {
 public:
  std::vector<uint32_t> images;
  explicit RegistrationIncomplete(std::vector<uint32_t> ids)
      : std::runtime_error("SfM could not register all selected exposures"),
        images(std::move(ids)) {}
};
Reconstruction Reconstruct(const Scene& scene, const fs::path& work, int workers, Json& report,
                           bool stable_calibration = false, bool allow_partial = false);
}  // namespace geosun

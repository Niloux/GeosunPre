#pragma once
#include <Eigen/Core>
#include <functional>

#include "app/runtime.h"
#include "geometry/camera.h"
namespace geosun {

struct LasHeader {
  uint32_t count = 0, offset = 0;
  uint16_t record_size = 0;
  uint8_t format = 0;
  Eigen::Vector3d scale, origin;
  // Declared bounds of the file coordinates, and the bounds after frame in the working frame.
  Eigen::Vector3d file_lower, file_upper, lower, upper;
  // File coordinates to the working frame; identity unless the LAS was converted.
  Similarity frame;
  bool converted = false;
  Json metadata;
};
struct LasPoint {
  std::array<int32_t, 3> raw{};
  std::array<uint16_t, 3> rgb{};
  uint8_t classification = 0;
  double gps_time = 0;
};
bool KeepLasPoint(const LasPoint& point);
Eigen::Vector3d LasCoordinates(const LasHeader& header, const LasPoint& point);
LasHeader ReadLasHeader(const fs::path& path);
// Takes the LAS records into the working frame through a rigid (unit-scale) similarity.
void ConvertLasFrame(LasHeader& header, const Similarity& frame, const std::string& name);
void ScanLas(const fs::path& path, const LasHeader& header,
             const std::function<void(uint64_t, const std::vector<LasPoint>&)>& consume);
std::vector<Eigen::Vector3d> SampleLas(const fs::path& path, const LasHeader& header,
                                       Json& metadata, size_t maximum = 1500000);
Json ExportLasPly(const fs::path& las, const LasHeader& header, const fs::path& output,
                  double voxel = .05);
void VerifyInitializationPly(const fs::path& path, uint64_t points);
}  // namespace geosun

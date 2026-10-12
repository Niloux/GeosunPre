#pragma once
#include <Eigen/Core>
#include <functional>

#include "runtime.h"

struct LasHeader {
  uint32_t count = 0, offset = 0;
  uint16_t record_size = 0;
  uint8_t format = 0;
  Eigen::Vector3d scale, origin, lower, upper;
  Json metadata;
};
struct LasPoint {
  std::array<int32_t, 3> raw{};
  std::array<uint16_t, 3> rgb{};
  uint8_t classification = 0;
  double gps_time = 0;
};
LasHeader ReadLasHeader(const fs::path& path);
void ScanLas(
    const fs::path& path, const LasHeader& header,
    const std::function<void(uint64_t, const std::vector<LasPoint>&)>& consume);
std::vector<Eigen::Vector3d> SampleLas(const fs::path& path,
                                       const LasHeader& header, Json& metadata,
                                       size_t maximum = 1500000);
Json ExportLasPly(const fs::path& las, const fs::path& output,
                  double voxel = .05);
void VerifyInitializationPly(const fs::path& path, uint64_t points);
void RunLasPly(const Arguments& args);

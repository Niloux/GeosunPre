#define EIGEN_NO_CUDA
#include "geometry/transient.h"
#include "geometry/symmetric_eigen.cuh"
#include "frontend/inference.h"
#include "io/las.h"
#include <algorithm>
#include <cmath>

namespace geosun {
namespace {
constexpr int Version = 2;
// Carving grid: a ray counts a voxel when it crosses the plane fitted to the voxel's returns
// inside the voxel, up to a margin short of its own return. A ray grazing a surface on its way
// to a farther one stays on one side of that plane; counting every traversed voxel instead
// carved table tops, desk edges and ceiling slats seen at shallow angles.
constexpr float Voxel = .1f, Margin = .15f, MarginRelative = .05f;
// A voxel is transient when the rays crossing it outnumber its returns this many times; on the
// exhibition hall this separated the walking-person trails from walls, floor and exhibits.
constexpr float PassRatio = 4;
// Records whose GPS time falls inside the navigation trajectory, needed to trust the time base.
constexpr double MinTimedFraction = .9;
constexpr size_t PointChunk = 500000;
constexpr int MaxProbes = 64, MaxSteps = 1 << 14;
// Voxels with fewer returns have no plane, and any traversal counts.
constexpr unsigned MinPlaneReturns = 3;
constexpr unsigned long long EmptyKey = ~0ull;

// moments holds per voxel the sums of x, y, z, xx, xy, xz, yy, yz, zz of its returns in metres
// from the voxel corner; planes holds the fitted unit normal and offset in the same frame.
struct VoxelTable {
  unsigned long long* keys;
  unsigned* hits;
  unsigned* passes;
  float* moments;
  float4* planes;
  unsigned mask;
};
__device__ unsigned long long Key(int x, int y, int z) {
  return (unsigned long long)x << 42 | (unsigned long long)y << 21 | (unsigned long long)z;
}
__device__ bool Inside(int x, int y, int z) {
  return x >= 0 && y >= 0 && z >= 0 && x < (1 << 21) && y < (1 << 21) && z < (1 << 21);
}
__device__ unsigned Slot(unsigned long long key, unsigned mask) {
  return unsigned((key * 0x9E3779B97F4A7C15ull) >> 32) & mask;
}
__device__ int Find(const VoxelTable& t, unsigned long long key) {
  for (unsigned i = 0, s = Slot(key, t.mask); i < MaxProbes; ++i, s = (s + 1) & t.mask) {
    const auto k = t.keys[s];
    if (k == key)
      return int(s);
    if (k == EmptyKey)
      return -1;
  }
  return -1;
}
__device__ unsigned long long PointKey(float3 p) {
  return Key(int(floorf(p.x / Voxel)), int(floorf(p.y / Voxel)), int(floorf(p.z / Voxel)));
}
__device__ float3 Local(float3 p) {
  return make_float3(p.x - floorf(p.x / Voxel) * Voxel, p.y - floorf(p.y / Voxel) * Voxel,
                     p.z - floorf(p.z / Voxel) * Voxel);
}
__global__ void Insert(VoxelTable t, const float3* points, int count, int* overflow) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto key = PointKey(points[i]);
  for (unsigned n = 0, s = Slot(key, t.mask); n < MaxProbes; ++n, s = (s + 1) & t.mask) {
    const auto previous = atomicCAS(&t.keys[s], EmptyKey, key);
    if (previous == EmptyKey || previous == key) {
      atomicAdd(&t.hits[s], 1u);
      const float3 q = Local(points[i]);
      const float values[9] = {q.x,       q.y,       q.z,       q.x * q.x, q.x * q.y,
                               q.x * q.z, q.y * q.y, q.y * q.z, q.z * q.z};
      for (int k = 0; k < 9; ++k)
        atomicAdd(&t.moments[size_t(s) * 9 + k], values[k]);
      return;
    }
  }
  atomicExch(overflow, 1);
}
__global__ void FitPlanes(VoxelTable t) {
  const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s > t.mask || t.keys[s] == EmptyKey)
    return;
  const unsigned n = t.hits[s];
  if (n < MinPlaneReturns) {
    t.planes[s] = make_float4(0, 0, 0, 0);
    return;
  }
  const float* m = t.moments + size_t(s) * 9;
  const double cx = m[0] / n, cy = m[1] / n, cz = m[2] / n;
  double a[3][3] = {{m[3] / n - cx * cx, m[4] / n - cx * cy, m[5] / n - cx * cz},
                    {0, m[6] / n - cy * cy, m[7] / n - cy * cz},
                    {0, 0, m[8] / n - cz * cz}};
  a[1][0] = a[0][1];
  a[2][0] = a[0][2];
  a[2][1] = a[1][2];
  double values[3];
  double3 normal;
  SymmetricEigen(a, values, normal);
  t.planes[s] = make_float4(float(normal.x), float(normal.y), float(normal.z),
                            float(normal.x * cx + normal.y * cy + normal.z * cz));
}
// Amanatides-Woo traversal from the scanner to Margin short of the return.
__global__ void Carve(VoxelTable t, const float3* points, const float3* sensors, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float3 o = sensors[i], p = points[i];
  if (isnan(o.x))
    return;
  const float dx = p.x - o.x, dy = p.y - o.y, dz = p.z - o.z,
              length = sqrtf(dx * dx + dy * dy + dz * dz),
              end = (length - fmaxf(Margin, MarginRelative * length)) / Voxel;
  if (end <= 0)
    return;
  const float d[3] = {dx / length, dy / length, dz / length},
              a[3] = {o.x / Voxel, o.y / Voxel, o.z / Voxel};
  int c[3], step[3];
  float next[3], delta[3];
  for (int k = 0; k < 3; ++k) {
    c[k] = int(floorf(a[k]));
    step[k] = d[k] > 0 ? 1 : -1;
    delta[k] = d[k] != 0 ? fabsf(1 / d[k]) : INFINITY;
    next[k] = d[k] != 0 ? (d[k] > 0 ? c[k] + 1 - a[k] : a[k] - c[k]) * delta[k] : INFINITY;
  }
  float entry = 0;
  for (int n = 0; n < MaxSteps && entry < end; ++n) {
    const int k = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2);
    if (Inside(c[0], c[1], c[2])) {
      const int s = Find(t, Key(c[0], c[1], c[2]));
      if (s >= 0) {
        // Signed plane distances where the ray enters and leaves the voxel (or stops).
        const float4 plane = t.planes[s];
        auto side = [&](float u) {
          float distance = -plane.w;
          for (int j = 0; j < 3; ++j)
            distance += (&plane.x)[j] * (a[j] + d[j] * u - c[j]) * Voxel;
          return distance;
        };
        const bool flat = plane.x != 0 || plane.y != 0 || plane.z != 0;
        if (!flat || (side(entry) > 0) != (side(fminf(next[k], end)) > 0))
          atomicAdd(&t.passes[s], 1u);
      }
    }
    entry = next[k];
    next[k] += delta[k];
    c[k] += step[k];
  }
}
__global__ void Classify(VoxelTable t, const float3* points, int count, uint8_t* transient) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int s = Find(t, PointKey(points[i]));
  transient[i] = s >= 0 && float(t.passes[s]) > PassRatio * float(t.hits[s]);
}
__global__ void CountVoxels(VoxelTable t, unsigned long long* counts) {
  const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s > t.mask || t.keys[s] == EmptyKey)
    return;
  atomicAdd(&counts[0], 1ull);
  if (float(t.passes[s]) > PassRatio * float(t.hits[s]))
    atomicAdd(&counts[1], 1ull);
}
}  // namespace

Json TransientLasRequest() {
  return {{"version", Version},
          {"method", "scanner_ray_carving_voxel_plane_crossing"},
          {"min_plane_returns", MinPlaneReturns},
          {"sensor", "navigation position at the record GPS time"},
          {"voxel_m", Voxel},
          {"end_margin", {{"metres", Margin}, {"relative", MarginRelative}}},
          {"pass_ratio", PassRatio}};
}

TransientLas FindTransientLas(const Scene& scene) {
  const auto started = Clock::now();
  const auto header = ReadLasHeader(scene.las);
  TransientLas result;
  result.report = TransientLasRequest();
  if (header.format != 1 && header.format != 3) {
    result.report.update({{"used", false}, {"reason", "LAS point format has no GPS time"}});
    return result;
  }
  const auto track = NavigationTrack(scene);
  Eigen::Vector3d lower = header.lower;
  for (const auto& row : track)
    lower = lower.cwiseMin(row.second);
  lower -= Eigen::Vector3d::Constant(2 * Voxel);
  Eigen::Vector3d upper = header.upper;
  for (const auto& row : track)
    upper = upper.cwiseMax(row.second);
  if (((upper - lower) / Voxel).maxCoeff() >= double(1 << 21))
    throw std::runtime_error("LAS extent exceeds the transient carving grid");

  // Kept records of each chunk relative to lower, with the scanner position at their GPS time
  // (NaN outside the trajectory).
  std::vector<float3> points, sensors;
  std::vector<uint64_t> records;
  auto load = [&](uint64_t start, const std::vector<LasPoint>& chunk, bool rays) {
    points.clear();
    sensors.clear();
    records.clear();
    for (size_t i = 0; i < chunk.size(); ++i) {
      if (!KeepLasPoint(chunk[i]))
        continue;
      const Eigen::Vector3d p = LasCoordinates(header, chunk[i]) - lower;
      points.push_back(make_float3(float(p.x()), float(p.y()), float(p.z())));
      records.push_back(start + i);
      if (!rays)
        continue;
      const double time = chunk[i].gps_time;
      const auto next = std::upper_bound(
          track.begin(), track.end(), time,
          [](double t, const std::pair<double, Eigen::Vector3d>& row) { return t < row.first; });
      if (next == track.begin() || next == track.end()) {
        sensors.push_back(make_float3(NAN, NAN, NAN));
        continue;
      }
      const auto& [t0, a] = *(next - 1);
      const auto& [t1, b] = *next;
      const Eigen::Vector3d o = a + (b - a) * ((time - t0) / (t1 - t0)) - lower;
      sensors.push_back(make_float3(float(o.x()), float(o.y()), float(o.z())));
    }
  };
  DeviceBuffer device_points(PointChunk * sizeof(float3)),
      device_sensors(PointChunk * sizeof(float3)), flags(PointChunk), overflow(sizeof(int)),
      counts(2 * sizeof(unsigned long long));
  auto upload = [](DeviceBuffer& target, const std::vector<float3>& values) {
    CudaCheck(cudaMemcpy(target.data, values.data(), values.size() * sizeof(float3),
                         cudaMemcpyHostToDevice));
  };
  // Returns fill the table; it doubles until every voxel finds a slot within MaxProbes.
  std::unique_ptr<DeviceBuffer> keys, hits, passes, moments, planes;
  VoxelTable table{};
  uint64_t kept = 0;
  for (size_t slots = size_t(1) << 20;; slots *= 2) {
    if (slots < header.count / 8)
      continue;
    keys = std::make_unique<DeviceBuffer>(slots * sizeof(unsigned long long));
    hits = std::make_unique<DeviceBuffer>(slots * sizeof(unsigned));
    passes = std::make_unique<DeviceBuffer>(slots * sizeof(unsigned));
    moments = std::make_unique<DeviceBuffer>(slots * 9 * sizeof(float));
    planes = std::make_unique<DeviceBuffer>(slots * sizeof(float4));
    CudaCheck(cudaMemset(keys->data, 0xff, keys->bytes));
    CudaCheck(cudaMemset(moments->data, 0, moments->bytes));
    CudaCheck(cudaMemset(hits->data, 0, hits->bytes));
    CudaCheck(cudaMemset(passes->data, 0, passes->bytes));
    CudaCheck(cudaMemset(overflow.data, 0, overflow.bytes));
    table = {static_cast<unsigned long long*>(keys->data), static_cast<unsigned*>(hits->data),
             static_cast<unsigned*>(passes->data), static_cast<float*>(moments->data),
             static_cast<float4*>(planes->data), unsigned(slots - 1)};
    kept = 0;
    ScanLas(scene.las, header, [&](uint64_t start, const std::vector<LasPoint>& chunk) {
      load(start, chunk, false);
      kept += points.size();
      if (points.empty())
        return;
      upload(device_points, points);
      Insert<<<int(points.size() + 255) / 256, 256>>>(
          table, static_cast<const float3*>(device_points.data), int(points.size()),
          static_cast<int*>(overflow.data));
      CudaCheck(cudaGetLastError());
    });
    int full = 0;
    CudaCheck(cudaMemcpy(&full, overflow.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (!full)
      break;
    if (slots >= (size_t(1) << 31))
      throw std::runtime_error("LAS voxel table for transient carving is too large");
  }
  FitPlanes<<<unsigned((uint64_t(table.mask) + 256) / 256), 256>>>(table);
  CudaCheck(cudaGetLastError());
  uint64_t timed = 0;
  ScanLas(scene.las, header, [&](uint64_t start, const std::vector<LasPoint>& chunk) {
    load(start, chunk, true);
    if (points.empty())
      return;
    for (const auto& o : sensors)
      timed += !std::isnan(o.x);
    upload(device_points, points);
    upload(device_sensors, sensors);
    Carve<<<int(points.size() + 127) / 128, 128>>>(
        table, static_cast<const float3*>(device_points.data),
        static_cast<const float3*>(device_sensors.data), int(points.size()));
    CudaCheck(cudaGetLastError());
  });
  CudaCheck(cudaDeviceSynchronize());
  result.report.update({{"kept_records", kept}, {"records_with_navigation", timed}});
  if (!kept || double(timed) < MinTimedFraction * double(kept)) {
    result.report.update(
        {{"used", false}, {"reason", "LAS GPS time does not match the navigation trajectory"}});
    return result;
  }
  result.drop.assign(header.count, false);
  uint64_t dropped = 0;
  std::vector<uint8_t> host(PointChunk);
  ScanLas(scene.las, header, [&](uint64_t start, const std::vector<LasPoint>& chunk) {
    load(start, chunk, false);
    if (points.empty())
      return;
    upload(device_points, points);
    Classify<<<int(points.size() + 255) / 256, 256>>>(
        table, static_cast<const float3*>(device_points.data), int(points.size()),
        static_cast<uint8_t*>(flags.data));
    CudaCheck(cudaGetLastError());
    CudaCheck(cudaMemcpy(host.data(), flags.data, points.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < points.size(); ++i)
      if (host[i]) {
        result.drop[records[i]] = true;
        ++dropped;
      }
  });
  unsigned long long voxels[2] = {0, 0};
  CudaCheck(cudaMemset(counts.data, 0, counts.bytes));
  CountVoxels<<<unsigned((uint64_t(table.mask) + 256) / 256), 256>>>(
      table, static_cast<unsigned long long*>(counts.data));
  CudaCheck(cudaGetLastError());
  CudaCheck(cudaMemcpy(voxels, counts.data, sizeof(voxels), cudaMemcpyDeviceToHost));
  if (FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed during transient carving");
  result.report.update({{"used", true},
                        {"voxels", voxels[0]},
                        {"transient_voxels", voxels[1]},
                        {"transient_records", dropped},
                        {"seconds", Seconds(started)}});
  Log("Transient LAS: " + std::to_string(dropped) + " of " + std::to_string(kept) +
      " records in " + std::to_string(voxels[1]) + " voxels crossed by later scanner rays");
  return result;
}
}  // namespace geosun

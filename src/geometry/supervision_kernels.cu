#define EIGEN_NO_CUDA
#include "geometry/supervision_kernels.h"
#include "frontend/inference.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cmath>

namespace geosun::supervision_gpu {
namespace {
// The world normal is octahedral-coded as two 16-bit values in 1..65535; 0 means no normal.
__device__ unsigned EncodeNormal(float x, float y, float z) {
  const float s = fabsf(x) + fabsf(y) + fabsf(z);
  x /= s;
  y /= s;
  if (z < 0) {
    const float fx = (1 - fabsf(y)) * (x < 0 ? -1.f : 1.f),
                fy = (1 - fabsf(x)) * (y < 0 ? -1.f : 1.f);
    x = fx;
    y = fy;
  }
  const unsigned u = unsigned(lrintf((x * .5f + .5f) * 65534)) + 1,
                 v = unsigned(lrintf((y * .5f + .5f) * 65534)) + 1;
  return u << 16 | v;
}
__device__ float3 DecodeNormal(unsigned code) {
  float x = float((code >> 16) - 1) / 65534 * 2 - 1, y = float((code & 65535) - 1) / 65534 * 2 - 1;
  const float z = 1 - fabsf(x) - fabsf(y);
  if (z < 0) {
    const float fx = (1 - fabsf(y)) * (x < 0 ? -1.f : 1.f),
                fy = (1 - fabsf(x)) * (y < 0 ? -1.f : 1.f);
    x = fx;
    y = fy;
  }
  const float length = sqrtf(x * x + y * y + z * z);
  return make_float3(x / length, y / length, z / length);
}
__device__ void FisheyePixel(double x, double y, double z, const double* p, double& u, double& v) {
  const double r = hypot(x, y), theta = atan2(r, z), tt = theta * theta;
  const double radial = theta * (1 + tt * (p[4] + tt * (p[5] + tt * (p[6] + tt * p[7])))),
               scale = r > 1e-15 ? radial / r : 0;
  u = p[0] * x * scale + p[2];
  v = p[1] * y * scale + p[3];
}
__device__ void RodriguesRotate(double wx, double wy, double wz, double x, double y, double z,
                                double& ox, double& oy, double& oz) {
  const double t2 = wx * wx + wy * wy + wz * wz;
  const double theta = sqrt(t2);
  const double a = theta < 1e-4 ? 1.0 - t2 / 6.0 + t2 * t2 / 120.0 : sin(theta) / theta;
  const double b = theta < 1e-4 ? 0.5 - t2 / 24.0 + t2 * t2 / 720.0 : (1.0 - cos(theta)) / t2;
  const double kx = wy * z - wz * y;
  const double ky = wz * x - wx * z;
  const double kz = wx * y - wy * x;
  const double kkx = wy * kz - wz * ky;
  const double kky = wz * kx - wx * kz;
  const double kkz = wx * ky - wy * kx;
  ox = x + a * kx + b * kkx;
  oy = y + a * ky + b * kky;
  oz = z + a * kz + b * kkz;
}
// Keeps each pixel's nearest ray distance. A surfel covers the pixels whose rays meet its tangent
// disk; a point without a footprint covers only its projected pixel.
__global__ void ProjectNearestSurfels(const Surfel* points, int count, const Projection* cameras,
                                      const float3* rays, float* nearest, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto& c = cameras[blockIdx.y];
  const auto point = points[i];
  const auto raw = point.raw;
  const double x = raw.x * c.scales[0], y = raw.y * c.scales[1], z = raw.z * c.scales[2];
  double px = c.rotation[0] * x + c.rotation[1] * y + c.rotation[2] * z + c.translation[0],
         py = c.rotation[3] * x + c.rotation[4] * y + c.rotation[5] * z + c.translation[1],
         pz = c.rotation[6] * x + c.rotation[7] * y + c.rotation[8] * z + c.translation[2];
  if (pz <= 0)
    return;
  const double q2 = px * px + py * py + pz * pz;
  if (q2 > c.max_q2)
    return;
  const double qz_bound = pz + c.qz_slack;
  if (c.min_qz_slope > 0 && qz_bound * qz_bound < c.min_qz_slope * c.min_qz_slope * q2)
    return;
  double wx = 0, wy = 0, wz = 0;
  if (c.rolling) {
    // The pixel's readout line sets its camera; a few fixed-point steps settle the line.
    const double qx = px, qy = py, qz = pz;
    for (int k = 0; k < 3; ++k) {
      double u, v;
      FisheyePixel(px, py, pz, c.params, u, v);
      const double s = c.axis ? (v - .5 * height) / height : (u - .5 * width) / width;
      const double bx = qx - s * c.motion[3], by = qy - s * c.motion[4], bz = qz - s * c.motion[5];
      wx = s * c.motion[0];
      wy = s * c.motion[1];
      wz = s * c.motion[2];
      RodriguesRotate(wx, wy, wz, bx, by, bz, px, py, pz);
      if (pz <= 0)
        return;
    }
  }
  const double r = hypot(px, py), theta = atan2(r, pz), distance = sqrt(r * r + pz * pz);
  if (theta > MaxTheta || distance < .05 || distance > MaxDepth)
    return;
  const double tt = theta * theta;
  const auto* p = c.params;
  const double radial = theta * (1 + tt * (p[4] + tt * (p[5] + tt * (p[6] + tt * p[7])))),
               scale = r > 1e-15 ? radial / r : 0,
               u = p[0] * px * scale + p[2], v = p[1] * py * scale + p[3];
  if (!(u >= 0 && v >= 0 && u < width && v < height))
    return;
  // The normal turns with the pose and, for rolling shutter, the centre's readout rotation.
  float3 n = make_float3(0, 0, 0);
  if (point.normal) {
    const auto world = DecodeNormal(point.normal);
    double nx = c.rotation[0] * world.x + c.rotation[1] * world.y + c.rotation[2] * world.z,
           ny = c.rotation[3] * world.x + c.rotation[4] * world.y + c.rotation[5] * world.z,
           nz = c.rotation[6] * world.x + c.rotation[7] * world.y + c.rotation[8] * world.z;
    double rx = nx, ry = ny, rz = nz;
    if (c.rolling)
      RodriguesRotate(wx, wy, wz, nx, ny, nz, rx, ry, rz);
    const double length = sqrt(rx * rx + ry * ry + rz * rz),
                 facing = rx * px + ry * py + rz * pz > 0 ? -1 / length : 1 / length;
    n = make_float3(float(rx * facing), float(ry * facing), float(rz * facing));
  }
  const size_t image = size_t(blockIdx.y) * width * height;
  auto cover = [&](int pixel, float t) {
    atomicMin(reinterpret_cast<unsigned*>(nearest + image + pixel), __float_as_uint(t));
  };
  const int cu = int(u), cv = int(v);
  const float d = float(distance), radius = point.radius;
  if (!point.normal || radius <= 0) {
    cover(cv * width + cu, d);
    return;
  }
  // Fisheye magnification stays below 1.6 within MaxTheta, so the box holds the footprint.
  const int reach = min(MaxSplat, int(1.6 * p[0] * radius / d) + 1);
  const float3 q = make_float3(float(px), float(py), float(pz));
  const float plane = n.x * q.x + n.y * q.y + n.z * q.z;
  bool centre = false;
  for (int yy = max(0, cv - reach); yy <= min(height - 1, cv + reach); ++yy)
    for (int xx = max(0, cu - reach); xx <= min(width - 1, cu + reach); ++xx) {
      const auto ray = rays[yy * width + xx];
      const float den = n.x * ray.x + n.y * ray.y + n.z * ray.z;
      if (den > -.05f)
        continue;
      const float t = plane / den, ex = t * ray.x - q.x, ey = t * ray.y - q.y,
                  ez = t * ray.z - q.z, e2 = (ex * ex + ey * ey + ez * ez) / (radius * radius);
      if (e2 <= 1 && t >= .05f && t <= float(MaxDepth)) {
        cover(yy * width + xx, t);
        centre = centre || (xx == cu && yy == cv);
      }
    }
  // The point's own pixel is covered even when the tangent plane meets its ray obliquely.
  if (!centre)
    cover(cv * width + cu, d);
}
// Grid-cell key of a slab-relative position; each index fits 21 bits.
__device__ unsigned long long CellKey(int x, int y, int z) {
  return (unsigned long long)x << 42 | (unsigned long long)y << 21 | (unsigned long long)z;
}
__device__ int3 CellOf(float3 p) {
  return make_int3(int(floorf(p.x / NormalCell)), int(floorf(p.y / NormalCell)),
                   int(floorf(p.z / NormalCell)));
}
constexpr unsigned long long EmptyCell = ~0ull;
__device__ unsigned CellSlot(unsigned long long key, unsigned mask) {
  return unsigned((key * 0x9E3779B97F4A7C15ull) >> 32) & mask;
}
__global__ void CellKeys(const int3* raw, int count, int3 origin, float3 scale, float3* xyz,
                         unsigned long long* keys, unsigned* order) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto r = raw[i];
  const float3 p = make_float3(float(r.x - origin.x) * scale.x, float(r.y - origin.y) * scale.y,
                               float(r.z - origin.z) * scale.z);
  xyz[i] = p;
  const auto cell = CellOf(p);
  keys[i] = CellKey(cell.x, cell.y, cell.z);
  order[i] = i;
}
__global__ void Gather(const float3* xyz, const unsigned* order, float3* sorted, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    sorted[i] = xyz[order[i]];
}
__global__ void InsertCells(const unsigned long long* keys, const unsigned* starts,
                            const unsigned* counts, const int* runs, unsigned long long* table,
                            uint2* ranges, unsigned mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *runs)
    return;
  for (unsigned slot = CellSlot(keys[i], mask);; slot = (slot + 1) & mask)
    if (atomicCAS(table + slot, EmptyCell, keys[i]) == EmptyCell) {
      ranges[slot] = make_uint2(starts[i], counts[i]);
      return;
    }
}
struct CellGrid {
  const float3* xyz;
  const unsigned long long* table;
  const uint2* ranges;
  unsigned mask;
  __device__ uint2 Range(int x, int y, int z) const {
    if (x < 0 || y < 0 || z < 0 || x >= 1 << 21 || y >= 1 << 21 || z >= 1 << 21)
      return make_uint2(0, 0);
    const auto key = CellKey(x, y, z);
    for (unsigned slot = CellSlot(key, mask);; slot = (slot + 1) & mask) {
      const auto found = table[slot];
      if (found == key)
        return ranges[slot];
      if (found == EmptyCell)
        return make_uint2(0, 0);
    }
  }
};
// Smallest-eigenvalue eigenvector of a symmetric 3x3 matrix by cyclic Jacobi rotations;
// values returns the eigenvalues in ascending order.
__device__ void SymmetricEigen(double a[3][3], double values[3], double3& normal) {
  double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off < 1e-30)
      break;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (fabs(a[p][q]) < 1e-300)
          continue;
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]),
                     t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1)),
                     cs = 1 / sqrt(t * t + 1), sn = t * cs;
        for (int k = 0; k < 3; ++k) {
          const double kp = a[k][p], kq = a[k][q];
          a[k][p] = cs * kp - sn * kq;
          a[k][q] = sn * kp + cs * kq;
        }
        for (int k = 0; k < 3; ++k) {
          const double pk = a[p][k], qk = a[q][k];
          a[p][k] = cs * pk - sn * qk;
          a[q][k] = sn * pk + cs * qk;
        }
        for (int k = 0; k < 3; ++k) {
          const double kp = v[k][p], kq = v[k][q];
          v[k][p] = cs * kp - sn * kq;
          v[k][q] = sn * kp + cs * kq;
        }
      }
  }
  int index[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[index[j]][index[j]] < a[index[i]][index[i]]) {
        const int swap = index[i];
        index[i] = index[j];
        index[j] = swap;
      }
  for (int i = 0; i < 3; ++i)
    values[i] = a[index[i]][index[i]];
  normal = make_double3(v[0][index[0]], v[1][index[0]], v[2][index[0]]);
}
// K nearest neighbours of p within the reach, as squared distances and cell-order indices.
__device__ int Nearest(const CellGrid& grid, float3 p, float* best, unsigned* nearest) {
  const int3 cell = CellOf(p);
  const float reach = NormalRings * NormalCell;
  int found = 0;
  for (int ring = 0; ring <= NormalRings; ++ring) {
    for (int dz = -ring; dz <= ring; ++dz)
      for (int dy = -ring; dy <= ring; ++dy)
        for (int dx = -ring; dx <= ring; ++dx) {
          if (max(abs(dx), max(abs(dy), abs(dz))) != ring)
            continue;
          const auto range = grid.Range(cell.x + dx, cell.y + dy, cell.z + dz);
          for (unsigned k = range.x; k < range.x + range.y; ++k) {
            const float3 q = grid.xyz[k];
            const float ex = q.x - p.x, ey = q.y - p.y, ez = q.z - p.z,
                        d2 = ex * ex + ey * ey + ez * ez;
            if (d2 > reach * reach || (found == Neighbours && d2 >= best[Neighbours - 1]))
              continue;
            int j = found < Neighbours ? found++ : Neighbours - 1;
            for (; j > 0 && best[j - 1] > d2; --j) {
              best[j] = best[j - 1];
              nearest[j] = nearest[j - 1];
            }
            best[j] = d2;
            nearest[j] = k;
          }
        }
    // Points beyond the searched cube lie farther than ring cells from p.
    const float guaranteed = ring * NormalCell;
    if (found == Neighbours && best[Neighbours - 1] <= guaranteed * guaranteed)
      break;
  }
  return found;
}
// The K-th neighbour distance, or the reach when fewer are found, gives the local area per
// point; computed for every slab point in cell order, halo included.
__global__ void Spacings(CellGrid grid, int count, float* spacing) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= count)
    return;
  float best[Neighbours];
  unsigned nearest[Neighbours];
  const int found = Nearest(grid, grid.xyz[s], best, nearest);
  spacing[s] = found == Neighbours ? sqrtf(best[Neighbours - 1]) : NormalRings * NormalCell;
}
// One thread per slab point in cell order; only the slab's own points (original index below
// own) produce surfels, the halo only supplies neighbours.
__global__ void EstimateSurfels(CellGrid grid, const unsigned* order, const float* spacings,
                                int count, int own, const int3* raw, Surfel* surfels,
                                float* rms) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= count)
    return;
  const unsigned index = order[s];
  if (index >= unsigned(own))
    return;
  const float3 p = grid.xyz[s];
  const int3 cell = CellOf(p);
  const float reach = NormalRings * NormalCell;
  float best[Neighbours];
  unsigned nearest[Neighbours];
  const int found = Nearest(grid, p, best, nearest);
  Surfel out{raw[index], 0, 0};
  rms[index] = -1;
  const float spacing = spacings[s];
  if (found < MinNeighbours) {
    surfels[index] = out;
    return;
  }
  const float radius = fminf(reach, fmaxf(NormalRadius, spacing));
  const int rings = int(ceilf(radius / NormalCell));
  double3 normal = make_double3(0, 0, 1), centre = make_double3(0, 0, 0);
  double values[3] = {0, 0, 0};
  float band = 0;
  int total = 0, inliers = 0;
  for (int pass = 0; pass < 3; ++pass) {
    float sx = 0, sy = 0, sz = 0, sxx = 0, sxy = 0, sxz = 0, syy = 0, syz = 0, szz = 0;
    int used = 0;
    for (int dz = -rings; dz <= rings; ++dz)
      for (int dy = -rings; dy <= rings; ++dy)
        for (int dx = -rings; dx <= rings; ++dx) {
          const auto range = grid.Range(cell.x + dx, cell.y + dy, cell.z + dz);
          for (unsigned k = range.x; k < range.x + range.y; ++k) {
            const float3 q = grid.xyz[k];
            const float ex = q.x - p.x, ey = q.y - p.y, ez = q.z - p.z;
            if (ex * ex + ey * ey + ez * ez > radius * radius)
              continue;
            if (pass && fabs((ex - centre.x) * normal.x + (ey - centre.y) * normal.y +
                             (ez - centre.z) * normal.z) > band)
              continue;
            sx += ex;
            sy += ey;
            sz += ez;
            sxx += ex * ex;
            sxy += ex * ey;
            sxz += ex * ez;
            syy += ey * ey;
            syz += ey * ez;
            szz += ez * ez;
            ++used;
          }
        }
    if (!pass)
      total = used;
    inliers = used;
    if (used < MinNeighbours)
      break;
    const double m = 1. / used;
    centre = make_double3(sx * m, sy * m, sz * m);
    double a[3][3] = {{sxx * m - centre.x * centre.x, sxy * m - centre.x * centre.y,
                       sxz * m - centre.x * centre.z},
                      {0, syy * m - centre.y * centre.y, syz * m - centre.y * centre.z},
                      {0, 0, szz * m - centre.z * centre.z}};
    a[1][0] = a[0][1];
    a[2][0] = a[0][2];
    a[2][1] = a[1][2];
    SymmetricEigen(a, values, normal);
    // Trim to 2.5 sigma of the first fit, then settle on the fixed inlier band.
    band = pass ? PlaneBand : fmaxf(PlaneBand, 2.5f * sqrtf(float(fmax(values[0], 0.))));
  }
  const double sigma = sqrt(fmax(values[0], 0.)),
               offset = fabs(centre.x * normal.x + centre.y * normal.y + centre.z * normal.z);
  // A stray return ahead of a surface finds its neighbours far away on that surface, whose
  // own points are close together: compare with the median spacing of the neighbours.
  float around[Neighbours];
  int others = 0;
  for (int j = 0; j < found; ++j)
    if (nearest[j] != unsigned(s)) {
      int k = others++;
      for (; k > 0 && around[k - 1] > spacings[nearest[j]]; --k)
        around[k] = around[k - 1];
      around[k] = spacings[nearest[j]];
    }
  const float surface = around[others / 2];
  if (inliers >= MinNeighbours && inliers >= .6 * total && sigma <= PlaneRms &&
      values[1] >= .05 * values[2] && offset <= PlaneBand && spacing <= StraySpacing * surface) {
    // Only points with all K neighbours inside the reach get a footprint: a sparse
    // neighbourhood gives no spacing. Others, and points without a plane, stay single
    // projected pixels, so stray returns are dropped before culling.
    out.normal = EncodeNormal(float(normal.x), float(normal.y), float(normal.z));
    if (found == Neighbours)
      out.radius = fminf(MaxFootprint,
                         FootprintScale * fminf(spacing, surface) * rsqrtf(float(Neighbours)));
    rms[index] = float(sigma);
  }
  surfels[index] = out;
}
template <class Run>
void Cub(std::unique_ptr<DeviceBuffer>& temporary, Run run) {
  size_t bytes = 0;
  CudaCheck(run(nullptr, bytes));
  if (!temporary || bytes > temporary->bytes)
    temporary = std::make_unique<DeviceBuffer>(bytes);
  CudaCheck(run(temporary->data, bytes));
}
}  // namespace

// Surfels of the slab's first own points; the rest of points is the halo.
std::vector<Surfel> SlabSurfels(const std::vector<int3>& points, size_t own, int3 origin,
                                float3 scale, cudaStream_t stream,
                                SurfelStats& stats) {
  const int count = int(points.size()), blocks = (count + 255) / 256;
  DeviceBuffer raw(count * sizeof(int3)), xyz(count * sizeof(float3)),
      sorted(count * sizeof(float3)), keys(count * 8), keys_out(count * 8), order(count * 4),
      order_out(count * 4), unique(count * 8), counts(count * 4), starts(count * 4), runs(4),
      surfels(own * sizeof(Surfel)), rms(own * sizeof(float));
  std::unique_ptr<DeviceBuffer> temporary;
  CudaCheck(cudaMemcpyAsync(raw.data, points.data(), raw.bytes, cudaMemcpyHostToDevice, stream));
  CellKeys<<<blocks, 256, 0, stream>>>(
      static_cast<const int3*>(raw.data), count, origin,
      scale,
      static_cast<float3*>(xyz.data), static_cast<unsigned long long*>(keys.data),
      static_cast<unsigned*>(order.data));
  CudaCheck(cudaGetLastError());
  auto* k_in = static_cast<unsigned long long*>(keys.data);
  auto* k_out = static_cast<unsigned long long*>(keys_out.data);
  auto* o_in = static_cast<unsigned*>(order.data);
  auto* o_out = static_cast<unsigned*>(order_out.data);
  Cub(temporary, [&](void* t, size_t& b) {
    return cub::DeviceRadixSort::SortPairs(t, b, k_in, k_out, o_in, o_out, count, 0, 63, stream);
  });
  Gather<<<blocks, 256, 0, stream>>>(static_cast<const float3*>(xyz.data), o_out,
                                     static_cast<float3*>(sorted.data), count);
  CudaCheck(cudaGetLastError());
  auto* u = static_cast<unsigned long long*>(unique.data);
  auto* c = static_cast<unsigned*>(counts.data);
  auto* r = static_cast<int*>(runs.data);
  Cub(temporary, [&](void* t, size_t& b) {
    return cub::DeviceRunLengthEncode::Encode(t, b, k_out, u, c, r, count, stream);
  });
  int cells = 0;
  CudaCheck(cudaMemcpyAsync(&cells, r, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  Cub(temporary, [&](void* t, size_t& b) {
    return cub::DeviceScan::ExclusiveSum(t, b, c, static_cast<unsigned*>(starts.data), cells,
                                         stream);
  });
  unsigned slots = 1;
  while (slots < 2u * unsigned(cells))
    slots <<= 1;
  DeviceBuffer table(size_t(slots) * 8), ranges(size_t(slots) * sizeof(uint2));
  CudaCheck(cudaMemsetAsync(table.data, 0xff, table.bytes, stream));
  InsertCells<<<(cells + 255) / 256, 256, 0, stream>>>(
      u, static_cast<const unsigned*>(starts.data), c, r,
      static_cast<unsigned long long*>(table.data), static_cast<uint2*>(ranges.data), slots - 1);
  CudaCheck(cudaGetLastError());
  const CellGrid grid{static_cast<const float3*>(sorted.data),
                      static_cast<const unsigned long long*>(table.data),
                      static_cast<const uint2*>(ranges.data), slots - 1};
  DeviceBuffer spacing(count * sizeof(float));
  Spacings<<<(count + 127) / 128, 128, 0, stream>>>(grid, count,
                                                    static_cast<float*>(spacing.data));
  CudaCheck(cudaGetLastError());
  EstimateSurfels<<<(count + 127) / 128, 128, 0, stream>>>(grid, o_out,
                                                          static_cast<const float*>(spacing.data),
                                                          count, int(own),
                                             static_cast<const int3*>(raw.data),
                                             static_cast<Surfel*>(surfels.data),
                                             static_cast<float*>(rms.data));
  CudaCheck(cudaGetLastError());
  std::vector<Surfel> result(own);
  std::vector<float> sigma(own);
  CudaCheck(cudaMemcpyAsync(result.data(), surfels.data, surfels.bytes, cudaMemcpyDeviceToHost,
                            stream));
  CudaCheck(cudaMemcpyAsync(sigma.data(), rms.data, rms.bytes, cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < own; ++i) {
    stats.normals += result[i].normal != 0;
    if (i % 101 == 0 && sigma[i] >= 0) {
      stats.radius.push_back(result[i].radius);
      stats.rms.push_back(sigma[i]);
    }
  }
  return result;
}

void ProjectNearest(const Surfel* points, size_t count, size_t images, const Projection* cameras,
                    const float3* rays, float* nearest, int width, int height,
                    cudaStream_t stream) {
  const dim3 grid(unsigned((count + 255) / 256), unsigned(images));
  ProjectNearestSurfels<<<grid, 256, 0, stream>>>(points, int(count), cameras, rays, nearest,
                                                  width, height);
  CudaCheck(cudaGetLastError());
}

__global__ void ClearEmptyNearest(float* depths, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && !(depths[i] <= float(MaxDepth)))
    depths[i] = 0;
}

void ResolveNearest(float* depths, size_t count, cudaStream_t stream) {
  ClearEmptyNearest<<<unsigned((count + 255) / 256), 256, 0, stream>>>(depths, count);
  CudaCheck(cudaGetLastError());
}

namespace {
__global__ void SampleMogeFacesKernel(const uint8_t* prefiltered, int pre_w, int pre_h,
                                      const double* source_params, int src_w, int src_h,
                                      double source_radius, double sx, double sy,
                                      const double* to_camera, double3 omega, int rolling, int axis,
                                      int face_size, double focal, float* out_chw,
                                      uint8_t* out_face_valid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = blockIdx.y;
  const size_t face_pixels = size_t(face_size) * face_size;
  if (idx >= int(face_pixels))
    return;
  const int v = idx / face_size, u = idx % face_size;
  const double rx0 = (u + 0.5 - face_size * 0.5) / focal;
  const double ry0 = (v + 0.5 - face_size * 0.5) / focal;
  const double* tc = to_camera + size_t(f) * 9;
  double rx = tc[0] * rx0 + tc[1] * ry0 + tc[2];
  double ry = tc[3] * rx0 + tc[4] * ry0 + tc[5];
  double rz = tc[6] * rx0 + tc[7] * ry0 + tc[8];
  const double rlen = sqrt(rx * rx + ry * ry + rz * rz);
  rx /= rlen;
  ry /= rlen;
  rz /= rlen;
  double px = rx, py = ry, pz = rz;
  if (rolling) {
    const double axis_size = axis ? double(src_h) : double(src_w);
    for (int k = 0; k < 3; ++k) {
      if (pz <= 0.0)
        break;
      double u_src = 0.0, v_src = 0.0;
      FisheyePixel(px, py, pz, source_params, u_src, v_src);
      double s = ((axis ? v_src : u_src) - 0.5 * axis_size) / axis_size;
      s = fmax(-0.5, fmin(0.5, s));
      const double wx = s * omega.x, wy = s * omega.y, wz = s * omega.z;
      RodriguesRotate(wx, wy, wz, rx, ry, rz, px, py, pz);
    }
  }
  bool valid = pz > 0.0;
  double u_src = 0.0, v_src = 0.0;
  if (valid) {
    const double r = hypot(px, py);
    const double theta = atan2(r, pz);
    if (theta > MaxTheta)
      valid = false;
    else {
      FisheyePixel(px, py, pz, source_params, u_src, v_src);
      const double dx = (u_src - source_params[2]) / source_params[0];
      const double dy = (v_src - source_params[3]) / source_params[1];
      if (!(u_src >= 0.0 && v_src >= 0.0 && u_src < double(src_w) && v_src < double(src_h) &&
            dx * dx + dy * dy <= source_radius * source_radius))
        valid = false;
    }
  }
  const size_t out_base = size_t(f) * 3 * face_pixels + size_t(idx);
  if (!valid) {
    out_face_valid[size_t(f) * face_pixels + size_t(idx)] = 0;
    out_chw[out_base] = 0.f;
    out_chw[out_base + face_pixels] = 0.f;
    out_chw[out_base + 2 * face_pixels] = 0.f;
    return;
  }
  const double fx = fmax(0.0, fmin(double(pre_w - 1), u_src * sx - 0.5));
  const double fy = fmax(0.0, fmin(double(pre_h - 1), v_src * sy - 0.5));
  const int x0 = int(floor(fx)), y0 = int(floor(fy));
  const int x1 = min(x0 + 1, pre_w - 1), y1 = min(y0 + 1, pre_h - 1);
  const float tx = float(fx - x0), ty = float(fy - y0);
  const float w00 = (1.f - tx) * (1.f - ty), w10 = tx * (1.f - ty), w01 = (1.f - tx) * ty,
              w11 = tx * ty;
  const size_t i00 = (size_t(y0) * pre_w + x0) * 3;
  const size_t i10 = (size_t(y0) * pre_w + x1) * 3;
  const size_t i01 = (size_t(y1) * pre_w + x0) * 3;
  const size_t i11 = (size_t(y1) * pre_w + x1) * 3;
  for (int c = 0; c < 3; ++c) {
    const float val = w00 * prefiltered[i00 + c] + w10 * prefiltered[i10 + c] +
                      w01 * prefiltered[i01 + c] + w11 * prefiltered[i11 + c];
    out_chw[out_base + size_t(c) * face_pixels] = val * (1.f / 255.f);
  }
  out_face_valid[size_t(f) * face_pixels + size_t(idx)] = 255;
}

__global__ void SampleMogeShiftGridKernel(const float* points, const float* mask,
                                          const uint8_t* face_valid, int size, float4* samples) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= 64 * 64)
    return;
  const int f = blockIdx.y;
  const int i = k / 64, j = k % 64;
  const int row = i * size / 64, column = j * size / 64;
  const size_t face_pixels = size_t(size) * size;
  const size_t idx = size_t(f) * face_pixels + size_t(row) * size + column;
  const float pz = points[3 * idx + 2];
  samples[size_t(f) * 4096 + k] =
      (face_valid[idx] && mask[idx] > .5f && isfinite(pz))
          ? make_float4(points[3 * idx], points[3 * idx + 1], pz, 1.f)
          : make_float4(0.f, 0.f, 0.f, 0.f);
}

__global__ void BuildMogeFaceEstimatesKernel(const float* points, const float* normals,
                                             const float* mask, const uint8_t* face_valid,
                                             const double* shifts, const uint8_t* shift_ok,
                                             const float* metric_scales, const double* to_camera,
                                             int size, double focal, float* out_distance,
                                             float* out_normal, uint32_t* valid_counts) {
  __shared__ unsigned int block_valid[256];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int f = blockIdx.y;
  const size_t face_pixels = size_t(size) * size;
  unsigned int is_valid = 0;
  if (idx < int(face_pixels)) {
    const size_t p_idx = size_t(f) * face_pixels + size_t(idx);
    const float pz = points[3 * p_idx + 2];
    const double nx0 = normals[3 * p_idx], ny0 = normals[3 * p_idx + 1],
                 nz0 = normals[3 * p_idx + 2];
    const double n0_len = sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0);
    const bool valid_px = face_valid[p_idx] && mask[p_idx] > .5f && isfinite(pz) &&
                          isfinite(n0_len) && n0_len > 1e-6;
    const double depth = double(pz) + shifts[f];
    if (shift_ok[f] && valid_px && depth > 0.0) {
      const int v = idx / size, u = idx % size;
      const double rx = (u + 0.5 - size * 0.5) / focal;
      const double ry = (v + 0.5 - size * 0.5) / focal;
      const double ray_norm = sqrt(rx * rx + ry * ry + 1.0);
      const double* tc = to_camera + size_t(f) * 9;
      double nx = (tc[0] * nx0 + tc[1] * ny0 + tc[2] * nz0) / n0_len;
      double ny = (tc[3] * nx0 + tc[4] * ny0 + tc[5] * nz0) / n0_len;
      double nz = (tc[6] * nx0 + tc[7] * ny0 + tc[8] * nz0) / n0_len;
      const double cx = tc[0] * rx + tc[1] * ry + tc[2];
      const double cy = tc[3] * rx + tc[4] * ry + tc[5];
      const double cz = tc[6] * rx + tc[7] * ry + tc[8];
      if (nx * cx + ny * cy + nz * cz > 0.0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }
      out_distance[p_idx] = float(depth * ray_norm * double(metric_scales[f]));
      out_normal[3 * p_idx] = float(nx);
      out_normal[3 * p_idx + 1] = float(ny);
      out_normal[3 * p_idx + 2] = float(nz);
      is_valid = 1;
    } else {
      out_distance[p_idx] = 0.f;
      out_normal[3 * p_idx] = 0.f;
      out_normal[3 * p_idx + 1] = 0.f;
      out_normal[3 * p_idx + 2] = 0.f;
    }
  }
  block_valid[tid] = is_valid;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      block_valid[tid] += block_valid[tid + stride];
    __syncthreads();
  }
  if (tid == 0 && block_valid[0] > 0)
    atomicAdd(valid_counts + f, block_valid[0]);
}
}  // namespace

void SampleMogeFaces(const uint8_t* prefiltered, int pre_w, int pre_h, const double* source_params,
                     int src_w, int src_h, double source_radius, double sx, double sy,
                     const double* to_camera, double3 omega, int rolling, int axis, int faces,
                     int face_size, double focal, float* out_chw, uint8_t* out_face_valid,
                     cudaStream_t stream) {
  if (faces <= 0)
    return;
  const dim3 grid(unsigned((size_t(face_size) * face_size + 255) / 256), unsigned(faces));
  SampleMogeFacesKernel<<<grid, 256, 0, stream>>>(
      prefiltered, pre_w, pre_h, source_params, src_w, src_h, source_radius, sx, sy, to_camera,
      omega, rolling, axis, face_size, focal, out_chw, out_face_valid);
  CudaCheck(cudaGetLastError());
}

void SampleMogeShiftGrid(const float* points, const float* mask, const uint8_t* face_valid,
                         int faces, int size, float4* samples, cudaStream_t stream) {
  if (faces <= 0)
    return;
  const dim3 grid(unsigned((64 * 64 + 255) / 256), unsigned(faces));
  SampleMogeShiftGridKernel<<<grid, 256, 0, stream>>>(points, mask, face_valid, size, samples);
  CudaCheck(cudaGetLastError());
}

void BuildMogeFaceEstimates(const float* points, const float* normals, const float* mask,
                            const uint8_t* face_valid, const double* shifts,
                            const uint8_t* shift_ok, const float* metric_scales,
                            const double* to_camera, int faces, int size, double focal,
                            float* out_distance, float* out_normal, uint32_t* valid_counts,
                            cudaStream_t stream) {
  if (faces <= 0)
    return;
  const dim3 grid(unsigned((size_t(size) * size + 255) / 256), unsigned(faces));
  BuildMogeFaceEstimatesKernel<<<grid, 256, 0, stream>>>(
      points, normals, mask, face_valid, shifts, shift_ok, metric_scales, to_camera, size, focal,
      out_distance, out_normal, valid_counts);
  CudaCheck(cudaGetLastError());
}
}  // namespace geosun::supervision_gpu

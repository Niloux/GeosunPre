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
__device__ float BlendBand(float nearest) { return .01f + .01f * nearest; }
// Pass Nearest keeps each pixel's nearest ray distance; pass Accumulate adds every surfel
// within the band of it to the pixel's blend. A surfel covers the pixels whose rays meet its
// tangent disk, weighted by 1 - (offset / radius)^2; a point without a footprint covers only
// its projected pixel.
template <bool Accumulate>
__global__ void ProjectSurfels(const Surfel* points, int count, const Projection* cameras,
                               const float3* rays, float* nearest, Blend* blends, int width,
                               int height) {
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
      px = bx + wy * bz - wz * by;
      py = by + wz * bx - wx * bz;
      pz = bz + wx * by - wy * bx;
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
    const double rx = nx + wy * nz - wz * ny, ry = ny + wz * nx - wx * nz,
                 rz = nz + wx * ny - wy * nx;
    const double length = sqrt(rx * rx + ry * ry + rz * rz),
                 facing = rx * px + ry * py + rz * pz > 0 ? -1 / length : 1 / length;
    n = make_float3(float(rx * facing), float(ry * facing), float(rz * facing));
  }
  const size_t image = size_t(blockIdx.y) * width * height;
  auto cover = [&](int pixel, float t, float weight) {
    if (!Accumulate) {
      atomicMin(reinterpret_cast<unsigned*>(nearest + image + pixel), __float_as_uint(t));
      return;
    }
    const float front = nearest[image + pixel];
    if (t > front + BlendBand(front))
      return;
    auto& b = blends[image + pixel];
    atomicAdd(&b.weight, weight);
    atomicAdd(&b.depth, weight * t);
    if (point.normal) {
      atomicAdd(&b.nx, weight * n.x);
      atomicAdd(&b.ny, weight * n.y);
      atomicAdd(&b.nz, weight * n.z);
      atomicAdd(&b.normals, weight);
    }
  };
  const int cu = int(u), cv = int(v);
  const float d = float(distance), radius = point.radius;
  if (!point.normal || radius <= 0) {
    cover(cv * width + cu, d, 1);
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
        cover(yy * width + xx, t, fmaxf(.05f, 1 - e2));
        centre = centre || (xx == cu && yy == cv);
      }
    }
  // The point's own pixel is covered even when the tangent plane meets its ray obliquely.
  if (!centre)
    cover(cv * width + cu, d, 1);
}
// Blended depth everywhere a surfel landed; the blended normal only where normal-bearing
// surfels carry most of the weight and agree (unit-vector mean length MinNormalAgreement).
__global__ void Resolve(const Blend* blends, float* depth, unsigned* normals, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const auto b = blends[i];
  depth[i] = 0;
  normals[i] = 0;
  if (b.weight <= 0)
    return;
  const float d = b.depth / b.weight;
  depth[i] = d <= float(MaxDepth) ? d : 0;
  const float length = sqrtf(b.nx * b.nx + b.ny * b.ny + b.nz * b.nz);
  if (b.normals >= .5f * b.weight && length >= MinNormalAgreement * b.normals)
    normals[i] = EncodeNormal(b.nx / length, b.ny / length, b.nz / length);
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
__device__ float At(const float* depth, int x, int y, int w, int h) {
  if (x < 0 || y < 0 || x >= w || y >= h)
    return 0;
  const float d = depth[y * w + x];
  return d > 0 && d <= MaxDepth ? d : 0;
}
__global__ void Cull(const float* raw, float* clean, const uint8_t* mask, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float nearest = float(MaxDepth), d = At(raw, x, y, w, h);
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const float v = At(raw, xx, yy, w, h);
      if (v > 0 && mask[yy * w + xx])
        nearest = fminf(nearest, v);
    }
  clean[i] = mask[i] && d <= nearest * 1.03f + .02f ? d : 0;
}
__global__ void Fill(const float* input, float* output, const uint8_t* mask, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float sum = 0, lo = float(MaxDepth), hi = 0;
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (!dx && !dy)
        continue;
      const float v = At(input, x + dx, y + dy, w, h);
      if (v > 0) {
        ++count;
        sum += v;
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
    }
  const float d = At(input, x, y, w, h);
  output[i] = d;
  if (mask[i] && !d && count >= 6 && hi - lo <= .02f * lo + .01f)
    output[i] = sum / count;
}
// Cull leaves no depth jump inside a 5x5 window, so occlusions are found on the supported
// projection before culling: a background pixel a clear step behind a supported foreground
// pixel within two pixels is a step seed, which also forms the background-side margin; its
// depth is kept for the foreground side. The step exceeds the depth ramp across two pixels
// of oblique floors, ceilings and side walls, so those surfaces do not seed. Pixels next to
// the semantic mask limit are marked separately.
__global__ void Edges(const float* depth, const uint8_t* mask, uint8_t* limits, float* seeds,
                      int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  float nearest = float(MaxDepth);
  bool boundary = !mask[i];
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const bool adjacent = abs(dx) <= 1 && abs(dy) <= 1;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
        boundary = boundary || adjacent;
        continue;
      }
      boundary = boundary || (adjacent && !mask[yy * w + xx]);
      const float v = At(depth, xx, yy, w, h);
      if (v > 0 && mask[yy * w + xx])
        nearest = fminf(nearest, v);
    }
  limits[i] = boundary;
  const float d = At(depth, x, y, w, h);
  seeds[i] = mask[i] && d > nearest * 1.1f + .05f ? d : 0;
}
// LAS foreground silhouettes spread by about OcclusionSpread beyond the photographed ones
// (lateral range noise and overlapping scans win the nearest-depth test), so a foreground
// pixel is dropped within that spread, in pixels at its depth, of a step seed behind it.
__global__ void Margins(const float* depth, const float* projected, const float* seeds,
                        const uint8_t* limits, uint8_t* edges, int w, int h, float focal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  edges[i] = limits[i] || seeds[i] > 0;
  const int x = i % w, y = i / w;
  float d = At(depth, x, y, w, h);
  if (!d)
    d = At(projected, x, y, w, h);
  if (edges[i] || !d)
    return;
  const int radius = min(MaxOcclusionMargin, int(ceilf(OcclusionSpread * focal / d)));
  const float behind = d * 1.1f + .05f;
  for (int yy = max(0, y - radius); yy <= min(h - 1, y + radius); ++yy)
    for (int xx = max(0, x - radius); xx <= min(w - 1, x + radius); ++xx)
      if (seeds[yy * w + xx] > behind) {
        edges[i] = 1;
        return;
      }
}
__global__ void Dilate(const uint8_t* input, uint8_t* output, int w, int h, int radius,
                         bool horizontal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  uint8_t bad = 0;
  for (int d = -radius; d <= radius; ++d) {
    const int x = i % w + (horizontal ? d : 0), y = i / w + (horizontal ? 0 : d);
    if (x >= 0 && y >= 0 && x < w && y < h)
      bad |= input[y * w + x];
  }
  output[i] = bad;
}
__device__ bool DepthSupported(const float* depth, int x, int y, int w, int h, double center) {
  int count = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const float v = At(depth, x + dx, y + dy, w, h);
      if (v > 0 && fabs(v - center) <= .02 * center + .01 && ++count >= 4)
        return true;
    }
  return false;
}
// Projected pixels without local depth support are dropped before culling, so an isolated
// outlier neither culls the surface around it nor seeds a boundary margin.
__global__ void Isolate(const float* input, float* output, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  const float d = At(input, x, y, w, h);
  output[i] = d && DepthSupported(input, x, y, w, h, d) ? d : 0;
}
// Depth stays valid off the occlusion margins with local support; the blended surfel normal
// is kept only where the pixel holds its blended depth (not a filled one).
__global__ void Encode(const float* depth, const float* projected, const unsigned* codes,
                       const float3* rays, const uint8_t* edges, uint16_t* millimetres,
                       uchar3* normals, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  millimetres[i] = 0;
  normals[i] = make_uchar3(0, 0, 0);
  const int x = i % w, y = i / w;
  const float d = At(depth, x, y, w, h);
  const auto ray = rays[i];
  if (edges[i] || !d || (ray.x == 0 && ray.y == 0 && ray.z == 0) ||
      !DepthSupported(depth, x, y, w, h, d))
    return;
  millimetres[i] = uint16_t(max(1, min(65535, int(d * 1000. + .5))));
  if (!codes[i] || projected[i] != d)
    return;
  const auto n = DecodeNormal(codes[i]);
  if (-(n.x * ray.x + n.y * ray.y + n.z * ray.z) < MinIncidence)
    return;
  normals[i] = make_uchar3(uint8_t(127.5f + 127.5f * n.z + .5f), uint8_t(127.5f + 127.5f * n.y + .5f),
                           uint8_t(127.5f + 127.5f * n.x + .5f));
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

void Project(const Surfel* points, size_t count, size_t images, const Projection* cameras,
             const float3* rays, float* nearest, Blend* blends, int width, int height,
             bool accumulate, cudaStream_t stream) {
  const dim3 grid(unsigned((count + 255) / 256), unsigned(images));
  if (accumulate)
    ProjectSurfels<true><<<grid, 256, 0, stream>>>(points, int(count), cameras, rays,
                                                          nearest, blends, width, height);
  else
    ProjectSurfels<false><<<grid, 256, 0, stream>>>(points, int(count), cameras, rays,
                                                           nearest, blends, width, height);
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

void EncodeMaps(const Blend* blends, int w, int h, int boundary, float focal,
                const uint8_t* keep, const float3* rays, float* raw_depth, unsigned* codes,
                float* filtered, float* scratch, uint8_t* limit, uint8_t* dilated,
                float* seeds, uint8_t* edge, uint16_t* millimetres, uchar3* normals,
                cudaStream_t stream) {
  const int size = w * h, blocks = (size + 255) / 256;
  Resolve<<<blocks, 256, 0, stream>>>(blends, raw_depth, codes, size);
  Isolate<<<blocks, 256, 0, stream>>>(raw_depth, scratch, w, h);
  Cull<<<blocks, 256, 0, stream>>>(scratch, filtered, keep, w, h);
  Edges<<<blocks, 256, 0, stream>>>(scratch, keep, limit, seeds, w, h);
  for (int pass = 0; pass < 2; ++pass) {
    Fill<<<blocks, 256, 0, stream>>>(filtered, scratch, keep, w, h);
    std::swap(filtered, scratch);
  }
  Dilate<<<blocks, 256, 0, stream>>>(limit, dilated, w, h, boundary, true);
  Dilate<<<blocks, 256, 0, stream>>>(dilated, limit, w, h, boundary, false);
  Margins<<<blocks, 256, 0, stream>>>(filtered, raw_depth, seeds, limit, edge, w, h, focal);
  Encode<<<blocks, 256, 0, stream>>>(filtered, raw_depth, codes, rays, edge, millimetres,
                                     normals, w, h);
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
      px = rx + wy * rz - wz * ry;
      py = ry + wz * rx - wx * rz;
      pz = rz + wx * ry - wy * rx;
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
  const int i = k / 64, j = k % 64;
  const int row = i * size / 64, column = j * size / 64;
  const size_t idx = size_t(row) * size + column;
  const float pz = points[3 * idx + 2];
  if (face_valid[idx] && mask[idx] > .5f && isfinite(pz))
    samples[k] = make_float4(points[3 * idx], points[3 * idx + 1], pz, 1.f);
  else
    samples[k] = make_float4(0.f, 0.f, 0.f, 0.f);
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
    const bool valid_px = face_valid[p_idx] && mask[p_idx] > .5f && isfinite(pz);
    const double depth = double(pz) + shifts[f];
    if (shift_ok[f] && valid_px && depth > 0.0) {
      const int v = idx / size, u = idx % size;
      const double rx = (u + 0.5 - size * 0.5) / focal;
      const double ry = (v + 0.5 - size * 0.5) / focal;
      const double ray_norm = sqrt(rx * rx + ry * ry + 1.0);
      const double* tc = to_camera + size_t(f) * 9;
      const double nx0 = normals[3 * p_idx], ny0 = normals[3 * p_idx + 1],
                   nz0 = normals[3 * p_idx + 2];
      double nx = tc[0] * nx0 + tc[1] * ny0 + tc[2] * nz0;
      double ny = tc[3] * nx0 + tc[4] * ny0 + tc[5] * nz0;
      double nz = tc[6] * nx0 + tc[7] * ny0 + tc[8] * nz0;
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

__device__ void CheckConflictDirection(const uint16_t* da, const uint16_t* db,
                                       const ConflictView& va, const ConflictView& vb,
                                       const double3* rays_a, int cell, int lattice_step,
                                       double view_limit, double tolerance, uint8_t* own,
                                       uint8_t* other, unsigned long long& tested,
                                       unsigned long long& conflicting) {
  const int cx = cell % va.cw, cy = cell / va.cw;
  const int x = cx * lattice_step, y = cy * lattice_step;
  const uint16_t millimetres = da[size_t(y) * va.width + x];
  const double3 ray = rays_a[cell];
  if (!millimetres || (ray.x == 0.0 && ray.y == 0.0 && ray.z == 0.0))
    return;
  const double dist_a = double(millimetres) / 1000.0;
  const double sx = dist_a * ray.x, sy = dist_a * ray.y, sz = dist_a * ray.z;
  double rx = sx, ry = sy, rz = sz;
  if (va.rolling) {
    const double size_a = va.axis ? va.height : va.width;
    const double coord_a = va.axis ? (y + 0.5) : (x + 0.5);
    const double s = (coord_a - 0.5 * size_a) / size_a;
    const double wx = s * va.motion_rot[0], wy = s * va.motion_rot[1], wz = s * va.motion_rot[2];
    const double w2 = wx * wx + wy * wy + wz * wz;
    const double dot_wm = wx * sx + wy * sy + wz * sz;
    const double cxw = wy * sz - wz * sy, cyw = wz * sx - wx * sz, czw = wx * sy - wy * sx;
    const double inv = 1.0 / (1.0 + w2);
    rx = (sx - cxw + dot_wm * wx) * inv + s * va.motion_trans[0];
    ry = (sy - cyw + dot_wm * wy) * inv + s * va.motion_trans[1];
    rz = (sz - czw + dot_wm * wz) * inv + s * va.motion_trans[2];
  }
  const double dx = rx - va.translation[0], dy = ry - va.translation[1], dz = rz - va.translation[2];
  const double wx_p = va.rotation[0] * dx + va.rotation[3] * dy + va.rotation[6] * dz;
  const double wy_p = va.rotation[1] * dx + va.rotation[4] * dy + va.rotation[7] * dz;
  const double wz_p = va.rotation[2] * dx + va.rotation[5] * dy + va.rotation[8] * dz;
  const double tx = vb.rotation[0] * wx_p + vb.rotation[1] * wy_p + vb.rotation[2] * wz_p +
                    vb.translation[0];
  const double ty = vb.rotation[3] * wx_p + vb.rotation[4] * wy_p + vb.rotation[5] * wz_p +
                    vb.translation[1];
  const double tz = vb.rotation[6] * wx_p + vb.rotation[7] * wy_p + vb.rotation[8] * wz_p +
                    vb.translation[2];
  double mx = tx, my = ty, mz = tz, u_px = 0.0, v_px = 0.0, s_b = 0.0;
  const int steps = vb.rolling ? 4 : 1;
  const double size_b = vb.axis ? vb.height : vb.width;
  for (int k = 0; k < steps; ++k) {
    if (vb.rolling) {
      const double bx = tx - s_b * vb.motion_trans[0];
      const double by = ty - s_b * vb.motion_trans[1];
      const double bz = tz - s_b * vb.motion_trans[2];
      const double wrx = s_b * vb.motion_rot[0], wry = s_b * vb.motion_rot[1],
                   wrz = s_b * vb.motion_rot[2];
      mx = bx + wry * bz - wrz * by;
      my = by + wrz * bx - wrx * bz;
      mz = bz + wrx * by - wry * bx;
    }
    const double norm = sqrt(mx * mx + my * my + mz * mz);
    if (mz < view_limit * norm)
      return;
    FisheyePixel(mx, my, mz, vb.params, u_px, v_px);
    if (vb.rolling)
      s_b = ((vb.axis ? v_px : u_px) - 0.5 * size_b) / size_b;
  }
  const int u = int(floor(u_px)), v = int(floor(v_px));
  if (u < 1 || v < 1 || u >= vb.width - 1 || v >= vb.height - 1)
    return;
  uint16_t nearest = 65535;
  for (int k = 0; k < 9 && nearest; ++k) {
    const uint16_t val = db[size_t(v + k / 3 - 1) * vb.width + (u + k % 3 - 1)];
    nearest = val ? (val < nearest ? val : nearest) : uint16_t(0);
  }
  if (!nearest)
    return;
  ++tested;
  const double metres = sqrt(mx * mx + my * my + mz * mz);
  if (metres * 1000.0 * tolerance < double(nearest)) {
    ++conflicting;
    own[cell] = 1;
    const int cell_b = (v / lattice_step) * vb.cw + (u / lattice_step);
    other[cell_b] = 1;
  }
}

__global__ void CheckConflictPairsKernel(const uint16_t* depths_base, size_t depth_stride,
                                         const ConflictView* views, const double3* rays_cam0,
                                         const double3* rays_cam1, const ConflictPair* pairs,
                                         int max_cells, int lattice_step, double view_limit,
                                         double tolerance, uint8_t* pair_own, uint8_t* pair_other,
                                         unsigned long long* stats) {
  __shared__ unsigned long long s_tested[256], s_conflicting[256];
  const int tid = threadIdx.x;
  const int cell = blockIdx.x * blockDim.x + tid;
  const int pair_idx = blockIdx.y;
  unsigned long long local_tested = 0, local_conflicting = 0;
  if (cell < max_cells) {
    const auto pair = pairs[pair_idx];
    const auto& va = views[pair.id_a];
    const auto& vb = views[pair.id_b];
    const uint16_t* da = depths_base + size_t(pair.slot_a) * depth_stride;
    const uint16_t* db = depths_base + size_t(pair.slot_b) * depth_stride;
    const double3* ra = va.camera == 0 ? rays_cam0 : rays_cam1;
    const double3* rb = vb.camera == 0 ? rays_cam0 : rays_cam1;
    uint8_t* own = pair_own + size_t(pair_idx) * max_cells;
    uint8_t* other = pair_other + size_t(pair_idx) * max_cells;
    if (cell < va.cw * va.ch)
      CheckConflictDirection(da, db, va, vb, ra, cell, lattice_step, view_limit, tolerance, own,
                             other, local_tested, local_conflicting);
    if (cell < vb.cw * vb.ch)
      CheckConflictDirection(db, da, vb, va, rb, cell, lattice_step, view_limit, tolerance, other,
                             own, local_tested, local_conflicting);
  }
  s_tested[tid] = local_tested;
  s_conflicting[tid] = local_conflicting;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_tested[tid] += s_tested[tid + stride];
      s_conflicting[tid] += s_conflicting[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (s_tested[0] > 0)
      atomicAdd(stats, s_tested[0]);
    if (s_conflicting[0] > 0)
      atomicAdd(stats + 1, s_conflicting[0]);
  }
}

__global__ void AccumulateConflictVotesKernel(const ConflictPair* pairs, int max_cells,
                                              const uint8_t* pair_own, const uint8_t* pair_other,
                                              unsigned int* tile_conflicts) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= max_cells)
    return;
  const auto pair = pairs[blockIdx.y];
  const size_t offset = size_t(blockIdx.y) * max_cells + cell;
  if (pair_own[offset])
    atomicAdd(tile_conflicts + size_t(pair.slot_a) * max_cells + cell, 1u);
  if (pair_other[offset])
    atomicAdd(tile_conflicts + size_t(pair.slot_b) * max_cells + cell, 1u);
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
                         int size, float4* samples, cudaStream_t stream) {
  SampleMogeShiftGridKernel<<<(64 * 64 + 255) / 256, 256, 0, stream>>>(points, mask, face_valid,
                                                                        size, samples);
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

void CheckConflictPairs(const uint16_t* depths_base, size_t depth_stride, const ConflictView* views,
                        const double3* rays_cam0, const double3* rays_cam1,
                        const ConflictPair* pairs, int num_pairs, int max_cells, int lattice_step,
                        double view_limit, double tolerance, uint8_t* pair_own,
                        uint8_t* pair_other, unsigned int* tile_conflicts, unsigned long long* stats,
                        cudaStream_t stream) {
  if (num_pairs <= 0 || max_cells <= 0)
    return;
  const dim3 grid(unsigned((max_cells + 255) / 256), unsigned(num_pairs));
  CheckConflictPairsKernel<<<grid, 256, 0, stream>>>(
      depths_base, depth_stride, views, rays_cam0, rays_cam1, pairs, max_cells, lattice_step,
      view_limit, tolerance, pair_own, pair_other, stats);
  CudaCheck(cudaGetLastError());
  AccumulateConflictVotesKernel<<<grid, 256, 0, stream>>>(pairs, max_cells, pair_own, pair_other,
                                                          tile_conflicts);
  CudaCheck(cudaGetLastError());
}
}  // namespace geosun::supervision_gpu

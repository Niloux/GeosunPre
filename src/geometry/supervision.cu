#define EIGEN_NO_CUDA
#include "geometry/supervision.h"
#include "frontend/frontend.h"
#include "frontend/inference.h"
#include "geometry/monocular.h"
#include "io/las.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <future>
#include <numeric>

namespace geosun {
namespace {
constexpr int Version = 8;
// Monocular supervision (source moge) is published as version 10; LAS maps keep version 8.
constexpr int MonocularVersion = 10;
constexpr const char* MonocularSource = "moge2_faces_las_scaled";
constexpr double MaxDepth = 65.535, MinIncidence = .3;
// Half of FieldOfViewDegrees in radians, kept as the published constant.
constexpr double MaxTheta = 1.396263401595464;
constexpr size_t DepthBudget = 1024 * 1024 * 1024;
constexpr size_t PointChunk = 500000;
// Mask decoding, PNG encoding and read-back run beside the GPU with this many images in flight.
constexpr size_t Lookahead = 6;
// LAS surfels: K nearest neighbours within the hashed-grid reach set the point spacing and the
// footprint; the tangent plane is a trimmed PCA over a ball of at least NormalRadius.
constexpr int Neighbours = 16, MinNeighbours = 10, NormalRings = 2;
constexpr float NormalCell = .03f, NormalRadius = .03f, PlaneBand = .02f, PlaneRms = .01f,
                FootprintScale = 2.f, MaxFootprint = .05f, StraySpacing = 2.f;
// Footprints reach at most this many map pixels from the projected centre.
constexpr int MaxSplat = 8;
constexpr float MinNormalAgreement = .95f;
// Metric spread of LAS foreground silhouettes beyond the photographed ones, measured on the
// indoor scene; the foreground-side occlusion margin covers it, up to MaxOcclusionMargin.
constexpr float OcclusionSpread = .01f;
constexpr int MaxOcclusionMargin = 24;
// Device bytes per point of a normal-estimation slab, all buffers at their worst-case size.
constexpr size_t SlabPointBytes = 192;
constexpr size_t MaxSlabPoints = 64 * 1000 * 1000;
constexpr int SlabBins = 1 << 16;
// The world normal is octahedral-coded as two 16-bit values in 1..65535; 0 means no normal.
__host__ __device__ unsigned EncodeNormal(float x, float y, float z) {
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
// raw holds the LAS integer coordinates; normal is the world-frame code (0: none) and radius
// the tangent footprint disk radius in metres (0: centre pixel only).
struct Surfel {
  int3 raw;
  unsigned normal;
  float radius;
};
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() {
    cudaStreamSynchronize(value);
    cudaStreamDestroy(value);
  }
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { CudaCheck(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
  ~Event() { cudaEventDestroy(value); }
};
struct PinnedPoints {
  Surfel* data = nullptr;
  PinnedPoints() {
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), PointChunk * sizeof(Surfel)));
  }
  ~PinnedPoints() { cudaFreeHost(data); }
};
struct PointTransfer {
  PinnedPoints host;
  DeviceBuffer device;
  Event copied, consumed;
  bool used = false;
  explicit PointTransfer(bool streaming) : device(streaming ? PointChunk * sizeof(Surfel) : 0) {}
};
// motion holds the readout rotation and the readout translation (readout time times velocity)
// of a rolling-shutter image, both in the reference camera frame; see ShutterMotion.
struct Projection {
  double rotation[9], translation[3], params[8], scales[3], motion[6];
  int rolling, axis;
};
__device__ void FisheyePixel(double x, double y, double z, const double* p, double& u, double& v) {
  const double r = hypot(x, y), theta = atan2(r, z), tt = theta * theta;
  const double radial = theta * (1 + tt * (p[4] + tt * (p[5] + tt * (p[6] + tt * p[7])))),
               scale = r > 1e-15 ? radial / r : 0;
  u = p[0] * x * scale + p[2];
  v = p[1] * y * scale + p[3];
}
// Per-pixel sums of the surfels within BlendBand of the nearest one: weights, weighted ray
// distances, and the weighted normals with their own weight.
struct Blend {
  float weight, depth, nx, ny, nz, normals;
};
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
struct SurfelStats {
  uint64_t normals = 0;
  std::vector<float> radius, rms;
};
// Surfels of the slab's first own points; the rest of points is the halo.
std::vector<Surfel> SlabSurfels(const std::vector<int3>& points, size_t own, int3 origin,
                                const Eigen::Vector3d& scale, cudaStream_t stream,
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
      make_float3(float(scale.x()), float(scale.y()), float(scale.z())),
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
class LasPointCache {
  std::unique_ptr<DeviceBuffer> resident_;
  std::array<std::unique_ptr<PointTransfer>, 2> transfers_;
  // Drain both streams before releasing any buffers, including constructor failure.
  Stream compute_, upload_;
  fs::path file_;
  uint64_t count_ = 0, excluded_ = 0, uploaded_bytes_ = 0;
  size_t budget_ = 0, batches_ = 0, slabs_ = 0;
  double prepare_seconds_ = 0, surfel_seconds_ = 0;
  SurfelStats stats_;

 public:
  LasPointCache(const fs::path& las, const LasHeader& header, const fs::path& output,
                int cache_mb) {
    const auto started = Clock::now();
    // Kept coordinates go to a compact file with a histogram along the longest axis, which
    // splits normal estimation into slabs that fit the device together with their halo.
    const Eigen::Vector3d extent = header.upper - header.lower;
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (extent[k] > extent[axis])
        axis = k;
    const double lower = std::floor((header.lower[axis] - header.origin[axis]) / header.scale[axis]),
                 upper = std::ceil((header.upper[axis] - header.origin[axis]) / header.scale[axis]),
                 bin_width = std::max(1., (upper - lower + 1) / SlabBins);
    std::vector<uint64_t> histogram(SlabBins);
    auto bin = [&](int raw) {
      return std::clamp(int(std::floor((raw - lower) / bin_width)), 0, SlabBins - 1);
    };
    const auto coordinates = output / "las-coordinates.cache";
    {
      std::ofstream packed(coordinates, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot create LAS coordinate cache");
      std::vector<int3> kept;
      ScanLas(las, header, [&](uint64_t, const std::vector<LasPoint>& chunk) {
        kept.clear();
        for (const auto& point : chunk) {
          if (KeepLasPoint(point)) {
            kept.push_back(make_int3(point.raw[0], point.raw[1], point.raw[2]));
            ++histogram[bin(point.raw[axis])];
          } else
            ++excluded_;
        }
        packed.write(reinterpret_cast<const char*>(kept.data()),
                     std::streamsize(kept.size() * sizeof(int3)));
        if (!packed)
          throw std::runtime_error("Cannot write LAS coordinate cache");
        count_ += kept.size();
      });
      packed.close();
      if (!packed)
        throw std::runtime_error("Incomplete LAS coordinate cache write");
    }
    size_t free, total;
    CudaCheck(cudaMemGetInfo(&free, &total));
    budget_ = std::min(size_t(cache_mb) * 1024 * 1024, free / 4);
    const size_t capacity = std::min(MaxSlabPoints, free / 2 / SlabPointBytes),
                 own_capacity = capacity * 3 / 4;
    // Each slab is a run of histogram bins; its points own the surfels they produce.
    std::vector<std::pair<int, int>> slabs;
    uint64_t filled = 0;
    for (int b = 0, first = 0; b < SlabBins; ++b) {
      if (histogram[b] > own_capacity)
        throw std::runtime_error("LAS normal slab bin exceeds device capacity");
      if (filled + histogram[b] > own_capacity) {
        slabs.emplace_back(first, b);
        first = b;
        filled = 0;
      }
      filled += histogram[b];
      if (b == SlabBins - 1)
        slabs.emplace_back(first, SlabBins);
    }
    if (size_t(count_) * sizeof(Surfel) <= budget_)
      resident_ = std::make_unique<DeviceBuffer>(size_t(count_) * sizeof(Surfel));
    std::ofstream surfels;
    if (!resident_) {
      file_ = output / "las-surfels.cache";
      surfels.open(file_, std::ios::binary);
      if (!surfels)
        throw std::runtime_error("Cannot create LAS surfel cache");
    }
    prepare_seconds_ = Seconds(started);
    const auto surfel_started = Clock::now();
    // Halo points hold every neighbour of an own point together with that neighbour's own
    // neighbours, whose spacing the stray test reads.
    const double halo = (2 * NormalRings + 1) * NormalCell / header.scale[axis];
    uint64_t stored = 0;
    std::vector<int3> chunk(PointChunk), own, rest;
    for (const auto& [first, last] : slabs) {
      const double lo = lower + first * bin_width,
                   hi = last == SlabBins ? std::numeric_limits<double>::infinity()
                                         : lower + last * bin_width;
      own.clear();
      rest.clear();
      int3 origin = make_int3(INT_MAX, INT_MAX, INT_MAX);
      std::ifstream packed(coordinates, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS coordinate cache");
      for (uint64_t read = 0; read < count_; read += PointChunk) {
        const size_t n = size_t(std::min<uint64_t>(PointChunk, count_ - read));
        if (!packed.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(n * sizeof(int3))))
          throw std::runtime_error("Truncated LAS coordinate cache");
        for (size_t i = 0; i < n; ++i) {
          const auto& p = chunk[i];
          const int a = axis == 0 ? p.x : axis == 1 ? p.y : p.z, b = bin(a);
          const bool mine = b >= first && b < last;
          if (!mine && (a < lo - halo || a >= hi + halo))
            continue;
          (mine ? own : rest).push_back(p);
          origin = make_int3(std::min(origin.x, p.x), std::min(origin.y, p.y),
                             std::min(origin.z, p.z));
        }
        if (own.size() + rest.size() > capacity)
          throw std::runtime_error("LAS normal slab and halo exceed device capacity");
      }
      if (own.empty())
        continue;
      const size_t mine = own.size();
      own.insert(own.end(), rest.begin(), rest.end());
      const auto result = SlabSurfels(own, mine, origin, header.scale, compute_.value, stats_);
      if (resident_)
        CudaCheck(cudaMemcpy(static_cast<Surfel*>(resident_->data) + stored, result.data(),
                             result.size() * sizeof(Surfel), cudaMemcpyHostToDevice));
      else {
        surfels.write(reinterpret_cast<const char*>(result.data()),
                      std::streamsize(result.size() * sizeof(Surfel)));
        if (!surfels)
          throw std::runtime_error("Cannot write LAS surfel cache");
      }
      stored += result.size();
      ++slabs_;
    }
    fs::remove(coordinates);
    if (stored != count_)
      throw std::runtime_error("LAS normal slabs do not cover every kept point");
    if (!resident_) {
      surfels.close();
      if (!surfels)
        throw std::runtime_error("Incomplete LAS surfel cache write");
    }
    for (auto& slot : transfers_)
      slot = std::make_unique<PointTransfer>(!resident_);
    surfel_seconds_ = Seconds(surfel_started);
    Log("LAS surfels: " + std::to_string(count_) + " points, " + std::to_string(stats_.normals) +
        " with normals, " + std::to_string(slabs_) + " slabs, " +
        (resident_ ? "GPU cache" : "compact file with double-buffered upload"));
  }
  cudaStream_t StreamHandle() const { return compute_.value; }
  uint64_t Count() const { return count_; }
  uint64_t Excluded() const { return excluded_; }
  void Project(const Projection* cameras, const float3* rays, float* nearest, Blend* blends,
               int width, int height, size_t images) {
    for (const bool accumulate : {false, true})
      Project(cameras, rays, nearest, blends, width, height, images, accumulate);
    ++batches_;
  }
  void Project(const Projection* cameras, const float3* rays, float* nearest, Blend* blends,
               int width, int height, size_t images, bool accumulate) {
    std::ifstream packed;
    if (!resident_) {
      packed.open(file_, std::ios::binary);
      if (!packed)
        throw std::runtime_error("Cannot read LAS surfel cache");
    }
    size_t index = 0;
    for (uint64_t first = 0; first < count_; first += PointChunk) {
      const size_t count = size_t(std::min<uint64_t>(PointChunk, count_ - first));
      const Surfel* points;
      PointTransfer* transfer = nullptr;
      if (resident_)
        points = static_cast<const Surfel*>(resident_->data) + first;
      else {
        transfer = transfers_[index++ % transfers_.size()].get();
        if (transfer->used)
          CudaCheck(cudaEventSynchronize(transfer->copied.value));
        const size_t bytes = count * sizeof(Surfel);
        if (!packed.read(reinterpret_cast<char*>(transfer->host.data), std::streamsize(bytes)))
          throw std::runtime_error("Truncated LAS surfel cache");
        if (transfer->used)
          CudaCheck(cudaStreamWaitEvent(upload_.value, transfer->consumed.value, 0));
        CudaCheck(cudaMemcpyAsync(transfer->device.data, transfer->host.data, bytes,
                                  cudaMemcpyHostToDevice, upload_.value));
        CudaCheck(cudaEventRecord(transfer->copied.value, upload_.value));
        CudaCheck(cudaStreamWaitEvent(compute_.value, transfer->copied.value, 0));
        points = static_cast<const Surfel*>(transfer->device.data);
        uploaded_bytes_ += bytes;
      }
      const dim3 grid(unsigned((count + 255) / 256), unsigned(images));
      if (accumulate)
        ProjectSurfels<true><<<grid, 256, 0, compute_.value>>>(points, int(count), cameras, rays,
                                                              nearest, blends, width, height);
      else
        ProjectSurfels<false><<<grid, 256, 0, compute_.value>>>(points, int(count), cameras, rays,
                                                               nearest, blends, width, height);
      CudaCheck(cudaGetLastError());
      if (transfer) {
        CudaCheck(cudaEventRecord(transfer->consumed.value, compute_.value));
        transfer->used = true;
      }
    }
  }
  void Finish() {
    CudaCheck(cudaStreamSynchronize(compute_.value));
    CudaCheck(cudaStreamSynchronize(upload_.value));
    if (!file_.empty())
      fs::remove(file_);
  }
  Json Surfels() const {
    auto median = [](std::vector<float> values) {
      if (values.empty())
        return 0.;
      std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
      return double(values[values.size() / 2]);
    };
    return {{"estimator", "knn_spacing_trimmed_pca_tangent_plane"},
            {"neighbours", Neighbours},
            {"minimum_neighbours", MinNeighbours},
            {"grid_cell_m", NormalCell},
            {"search_reach_m", NormalRings * NormalCell},
            {"minimum_plane_radius_m", NormalRadius},
            {"inlier_band_m", PlaneBand},
            {"plane_rms_limit_m", PlaneRms},
            {"minimum_inlier_fraction", .6},
            {"footprint", "tangent disk of 2 x min(kth-neighbour distance, neighbour median) / sqrt(k), at most 0.05 m, planar points with k neighbours within the reach only"},
            {"maximum_spacing_to_neighbour_median_spacing", StraySpacing},
            {"maximum_splat_pixels", MaxSplat},
            {"slabs", slabs_},
            {"points_with_normal", stats_.normals},
            {"median_footprint_m", median(stats_.radius)},
            {"median_plane_rms_m", median(stats_.rms)},
            {"seconds", surfel_seconds_}};
  }
  Json Profile() const {
    return {{"storage", resident_ ? "gpu" : "file"},
            {"las_scans", 1},
            {"budget_bytes", budget_},
            {"surfel_bytes", count_ * sizeof(Surfel)},
            {"gpu_cache_bytes", resident_ ? resident_->bytes : 0},
            {"transfer_buffer_bytes", resident_ ? 0 : 2 * PointChunk * sizeof(Surfel)},
            {"host_to_device_bytes", uploaded_bytes_},
            {"projection_batches", batches_},
            {"prepare_seconds", prepare_seconds_}};
  }
};
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
fs::path ExtraMask(const SupervisionOptions& options, const std::string& name) {
  return options.mask_dir / fs::u8path(name + ".png");
}
fs::path StaticMaskPath(const fs::path& masks, uint32_t image) {
  return masks / (std::to_string(image) + ".png");
}
cv::Mat StaticMask(const Scene& scene, const fs::path& masks, uint32_t image, const Camera& camera,
                   const SupervisionOptions& options) {
  auto mask = ReadImage(StaticMaskPath(masks, image), cv::IMREAD_GRAYSCALE);
  mask = mask == 255;
  if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, scene.images[image].name))) {
    auto extra = ReadImage(ExtraMask(options, scene.images[image].name), cv::IMREAD_GRAYSCALE);
    if (extra.size() != mask.size())
      throw std::runtime_error("Geometry exclusion mask must match its original image");
    mask &= extra == 255;
  }
  if (options.crop) {
    const auto& w = (*options.crop)[scene.images[image].camera];
    if (w.x + w.width > mask.cols || w.y + w.height > mask.rows)
      throw std::runtime_error("Crop window lies outside the static mask");
    mask = mask(cv::Rect(w.x, w.y, w.width, w.height)).clone();
  }
  // A map pixel is valid only when its whole footprint is. The invalid indicator is resampled,
  // so a fully valid footprint sums exact zeros; resampling the valid one instead lets float
  // area weights fall short of 1 on fixed columns of non-integer scales.
  cv::Mat invalid, resized;
  cv::Mat(mask == 0).convertTo(invalid, CV_32F, 1. / 255);
  cv::resize(invalid, resized, {camera.width, camera.height}, 0, 0, cv::INTER_AREA);
  mask = resized == 0.f;
  mask /= 255;
  return mask;
}
std::pair<uint64_t, uint64_t> Counts(const cv::Mat& depth, const cv::Mat& normal,
                                     bool normal_needs_depth) {
  static const auto squares = [] {
    std::array<double, 256> values;
    for (int v = 0; v < 256; ++v)
      values[v] = std::pow(v / 127.5 - 1, 2);
    return values;
  }();
  uint64_t depths = 0, normals = 0;
  for (int y = 0; y < depth.rows; ++y) {
    const auto* d = depth.ptr<uint16_t>(y);
    const auto* n = normal.ptr<uint8_t>(y);
    for (int x = 0; x < depth.cols; ++x, n += 3) {
      const bool has_depth = d[x] != 0;
      const bool has_normal = n[0] || n[1] || n[2];
      if (has_normal) {
        if (normal_needs_depth && !has_depth)
          throw std::runtime_error("Normal has no corresponding valid LAS depth");
        const double norm = squares[n[0]] + squares[n[1]] + squares[n[2]];
        if (std::abs(norm - 1) > .025)
          throw std::runtime_error("Nonunit encoded LAS normal");
      }
      depths += has_depth;
      normals += has_normal;
    }
  }
  return {depths, normals};
}
// The geometry engine a monocular request was made with.
Json GeometryIdentity(const fs::path& root) {
  const auto manifest = ReadJson(root / "manifest.json");
  if (!manifest.at("models").contains("geometry"))
    throw std::runtime_error(
        "Supervision models have no geometry engine; prepare a version 3 model package");
  return {{"models", root.u8string()},
          {"manifest", FileMetadata(root / "manifest.json")},
          {"engine", FileMetadata(root / "engines/geometry.engine")},
          {"sha256", manifest.at("models").at("geometry").at("sha256")},
          {"source", manifest.at("geometry_source")},
          {"face_size", manifest.at("geometry_size")},
          {"tokens", manifest.at("geometry_tokens")}};
}
}  // namespace
Camera MapCamera(const Camera& source, int max_size) {
  auto camera = source;
  const double scale = std::min(1., double(max_size) / std::max(source.width, source.height));
  camera.width = std::max(1, int(std::lround(source.width * scale)));
  camera.height = std::max(1, int(std::lround(source.height * scale)));
  for (int i = 0; i < 4; ++i)
    camera.params[i] *= i % 2 ? double(camera.height) / source.height
                              : double(camera.width) / source.width;
  return camera;
}
SupervisionOptions ParseSupervisionOptions(const Arguments& args) {
  SupervisionOptions options;
  if (args.values.count("--las-supervision") && args.values.count("--no-las-supervision"))
    throw std::runtime_error("--las-supervision conflicts with --no-las-supervision");
  options.enabled = args.Get("--no-las-supervision") != "true";
  options.max_size = args.Integer("--supervision-max-size", 1600);
  options.boundary = args.Integer("--supervision-boundary", 6);
  options.cache_mb = args.Integer("--supervision-cache-mb", 2048);
  if (!args.Get("--supervision-mask-dir").empty())
    options.mask_dir = args.Path("--supervision-mask-dir");
  if (options.max_size < 128 || options.max_size > 4096 || options.boundary > 32)
    throw std::runtime_error("Supervision max size must be 128-4096 and boundary 1-32 pixels");
  if (options.cache_mb > 4096)
    throw std::runtime_error("Supervision cache budget must be 1-4096 MiB");
  if (!options.enabled &&
      (args.values.count("--supervision-max-size") || args.values.count("--supervision-boundary") ||
       args.values.count("--supervision-cache-mb") || !options.mask_dir.empty() ||
       args.values.count("--supervision-source") || args.values.count("--supervision-models")))
    throw std::runtime_error("Supervision options cannot be used with --no-las-supervision");
  if (!options.mask_dir.empty() && !fs::is_directory(options.mask_dir))
    throw std::runtime_error("Geometry exclusion mask directory does not exist");
  options.source = args.Get("--supervision-source", "las");
  if (options.source != "las" && options.source != "moge")
    throw std::runtime_error("Supervision source must be las or moge");
  if (!args.Get("--supervision-models").empty()) {
    if (options.source != "moge")
      throw std::runtime_error("--supervision-models requires --supervision-source moge");
    options.geometry_models = args.Path("--supervision-models");
  } else if (options.source == "moge") {
    options.geometry_models = DefaultModels();
  }
  return options;
}
Json SupervisionRequest(const Scene& scene, const Reconstruction& model, const fs::path& masks_dir,
                        const SupervisionOptions& options, bool rolling) {
  if (!options.enabled)
    return nullptr;
  if (FileMetadata(scene.las) != scene.sources.at("las"))
    throw std::runtime_error("LAS changed since reconstruction; choose a new work directory");
  Json masks = Json::object(), static_masks = Json::object();
  for (const auto& image : scene.images) {
    if (!model.images[image.id].registered)
      continue;
    static_masks[image.name] = FileMetadata(StaticMaskPath(masks_dir, image.id));
    if (!options.mask_dir.empty() && fs::exists(ExtraMask(options, image.name)))
      masks[image.name] = FileMetadata(ExtraMask(options, image.name));
  }
  const bool monocular = options.source == "moge";
  Json request = {{"version", monocular ? MonocularVersion : Version},
                  {"max_size", options.max_size},
                  {"boundary_pixels", options.boundary},
                  {"las_file", FileMetadata(scene.las)},
                  {"exclusion_mask_dir", options.mask_dir.u8string()},
                  {"static_masks", static_masks},
                  {"exclusion_masks", masks}};
  if (rolling)
    request["readout_normals_version"] = 2;
  if (options.crop)
    request["crop_windows"] = {(*options.crop)[0].ToJson(), (*options.crop)[1].ToJson()};
  if (monocular) {
    request["source"] = "moge";
    request["geometry"] = GeometryIdentity(options.geometry_models);
  }
  return request;
}
Json ExportLasSupervision(const Scene& scene, const Reconstruction& model,
                          const fs::path& masks_dir, const fs::path& output,
                          const SupervisionOptions& options, const ShutterMotion* motion) {
  const auto started = Clock::now();
  const auto header = ReadLasHeader(scene.las);
  const auto request = SupervisionRequest(scene, model, masks_dir, options, motion != nullptr);
  std::vector<Json> rows(model.images.size());
  const bool monocular = options.source == "moge";
  std::unique_ptr<Models> geometry;
  std::unique_ptr<MonocularEstimator> estimator;
  if (monocular) {
    geometry = std::make_unique<Models>(options.geometry_models);
    estimator = std::make_unique<MonocularEstimator>(*geometry);
  }
  Json lenses = Json::array();
  LasPointCache points(scene.las, header, output, options.cache_mb);
  const auto stream = points.StreamHandle();
  uint64_t depth_pixels = 0, normal_pixels = 0;
  size_t done = 0;
  struct Written {
    uint32_t id;
    Json row;
  };
  std::deque<std::future<Written>> writes;
  auto finish = [&] {
    auto written = writes.front().get();
    writes.pop_front();
    depth_pixels += written.row.at("depth_pixels").get<uint64_t>();
    normal_pixels += written.row.at("normal_pixels").get<uint64_t>();
    rows[written.id] = std::move(written.row);
    ++done;
    if (done % 16 == 0 || done == model.Registered())
      Log("LAS supervision: " + std::to_string(done) + "/" + std::to_string(model.Registered()) +
          " images, " + std::to_string(Seconds(started)) + " s");
  };
  struct LoadedMask {
    cv::Mat pixels;
    int count;
    std::vector<cv::Mat> faces;  // MoGe faces of the photo for monocular supervision
  };
  // Lens plans outlive their camera's loop: the monocular fits run with the pending writes.
  std::array<std::optional<MonocularLens>, 2> lens_plans;
  for (uint32_t cid = 0; cid < model.cameras.size(); ++cid) {
    std::vector<uint32_t> images;
    for (uint32_t i = 0; i < model.images.size(); ++i)
      if (model.images[i].registered && model.images[i].camera == cid)
        images.push_back(i);
    if (images.empty())
      continue;
    const auto camera = MapCamera(model.cameras[cid], options.max_size);
    const int w = camera.width, h = camera.height, size = w * h, blocks = (size + 255) / 256;
    auto& lens = lens_plans.at(cid);
    if (monocular) {
      lens = PlanMonocularLens(model.cameras[cid], camera, estimator->FaceSize());
      lenses.push_back({{"camera", cid}, {"plan", lens->ToJson()}});
    }
    const size_t batch = std::min<size_t>(
        32, std::max<size_t>(1, DepthBudget / (size * (sizeof(float) + sizeof(Blend)))));
    std::vector<float3> ray_host(size);
    const double valid_radius = camera.Radius();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double dx = (x + .5 - camera.params[2]) / camera.params[0],
                     dy = (y + .5 - camera.params[3]) / camera.params[1];
        const Eigen::Vector3d ray = dx * dx + dy * dy <= valid_radius * valid_radius
                                       ? camera.Ray({x + .5, y + .5}) : Eigen::Vector3d::Zero();
        ray_host[y * w + x] = make_float3(float(ray.x()), float(ray.y()), float(ray.z()));
      }
    DeviceBuffer nearest(batch * size * sizeof(float)), blends(batch * size * sizeof(Blend)),
        cameras(batch * sizeof(Projection)), rays(size * sizeof(float3)),
        projected(size * sizeof(float)), codes(size * sizeof(unsigned)),
        clean(size * sizeof(float)), filled(size * sizeof(float)),
        seeds(size * sizeof(float)), mask(size), limit(size), dilated(size), edge(size),
        millimetres(size * sizeof(uint16_t)), normals(size * sizeof(uchar3));
    CudaCheck(
        cudaMemcpyAsync(rays.data, ray_host.data(), rays.bytes, cudaMemcpyHostToDevice, stream));
    // Masks for the following images decode while the GPU projects the current batch.
    std::deque<std::future<LoadedMask>> masks;
    size_t requested = 0;
    auto load_masks = [&] {
      for (; masks.size() < Lookahead && requested < images.size(); ++requested)
        masks.push_back(std::async(std::launch::async, [&, id = images[requested]] {
          auto pixels = StaticMask(scene, masks_dir, id, camera, options);
          const int count = cv::countNonZero(pixels);
          LoadedMask loaded{std::move(pixels), count, {}};
          if (lens) {
            auto photo = ReadImage(scene.images[id].source, cv::IMREAD_COLOR);
            if (options.crop) {
              const auto& window = (*options.crop)[cid];
              photo = photo(cv::Rect(window.x, window.y, window.width, window.height)).clone();
            }
            cv::cvtColor(photo, photo, cv::COLOR_BGR2RGB);
            loaded.faces = MonocularEstimator::Faces(*lens, photo);
          }
          return loaded;
        }));
    };
    for (size_t first = 0; first < images.size(); first += batch) {
      load_masks();
      const size_t count = std::min(batch, images.size() - first);
      std::vector<Projection> projection(count);
      for (size_t j = 0; j < count; ++j) {
        auto& p = projection[j];
        const auto& pose = model.images[images[first + j]].pose;
        const Eigen::Matrix3d rotation = pose.Rotation().toRotationMatrix();
        const Eigen::Vector3d translation = rotation * header.origin + pose.Translation();
        for (int r = 0; r < 3; ++r) {
          p.translation[r] = translation[r];
          p.scales[r] = header.scale[r];
          for (int c = 0; c < 3; ++c)
            p.rotation[r * 3 + c] = rotation(r, c);
        }
        std::copy(camera.params.begin(), camera.params.end(), p.params);
        if (motion) {
          const uint32_t id = images[first + j];
          const Eigen::Vector3d moved = motion->readout[cid] * motion->velocity[id];
          for (int k = 0; k < 3; ++k) {
            p.motion[k] = motion->rotation[id][k];
            p.motion[3 + k] = moved[k];
          }
          p.rolling = 1;
          p.axis = motion->axis;
        }
      }
      CudaCheck(cudaMemcpyAsync(cameras.data, projection.data(), count * sizeof(Projection),
                                cudaMemcpyHostToDevice, stream));
      CudaCheck(cudaMemsetAsync(nearest.data, 0x7f, count * size * sizeof(float), stream));
      CudaCheck(cudaMemsetAsync(blends.data, 0, count * size * sizeof(Blend), stream));
      points.Project(static_cast<const Projection*>(cameras.data),
                     static_cast<const float3*>(rays.data), static_cast<float*>(nearest.data),
                     static_cast<Blend*>(blends.data), w, h, count);
      for (size_t j = 0; j < count; ++j) {
        const uint32_t id = images[first + j];
        const auto static_mask = masks.front().get();
        masks.pop_front();
        load_masks();
        CudaCheck(cudaMemcpyAsync(mask.data, static_mask.pixels.data, size, cudaMemcpyHostToDevice,
                                  stream));
        auto* raw_depth = static_cast<float*>(projected.data);
        Resolve<<<blocks, 256, 0, stream>>>(static_cast<const Blend*>(blends.data) + j * size,
                                            raw_depth, static_cast<unsigned*>(codes.data), size);
        auto* filtered = static_cast<float*>(clean.data);
        auto* scratch = static_cast<float*>(filled.data);
        const auto* keep = static_cast<const uint8_t*>(mask.data);
        Isolate<<<blocks, 256, 0, stream>>>(raw_depth, scratch, w, h);
        Cull<<<blocks, 256, 0, stream>>>(scratch, filtered, keep, w, h);
        Edges<<<blocks, 256, 0, stream>>>(scratch, keep, static_cast<uint8_t*>(limit.data),
                                          static_cast<float*>(seeds.data), w, h);
        for (int pass = 0; pass < 2; ++pass) {
          Fill<<<blocks, 256, 0, stream>>>(filtered, scratch, keep, w, h);
          std::swap(filtered, scratch);
        }
        Dilate<<<blocks, 256, 0, stream>>>(static_cast<const uint8_t*>(limit.data),
                                           static_cast<uint8_t*>(dilated.data), w, h,
                                           options.boundary, true);
        Dilate<<<blocks, 256, 0, stream>>>(static_cast<const uint8_t*>(dilated.data),
                                           static_cast<uint8_t*>(limit.data), w, h,
                                           options.boundary, false);
        Margins<<<blocks, 256, 0, stream>>>(filtered, raw_depth, static_cast<const float*>(seeds.data),
                                            static_cast<const uint8_t*>(limit.data),
                                            static_cast<uint8_t*>(edge.data), w, h,
                                            float(camera.params[0]));
        Encode<<<blocks, 256, 0, stream>>>(
            filtered, raw_depth, static_cast<const unsigned*>(codes.data),
            static_cast<const float3*>(rays.data), static_cast<const uint8_t*>(edge.data),
            static_cast<uint16_t*>(millimetres.data), static_cast<uchar3*>(normals.data), w, h);
        CudaCheck(cudaGetLastError());
        cv::Mat depth(h, w, CV_16UC1), normal(h, w, CV_8UC3);
        CudaCheck(cudaMemcpyAsync(depth.data, millimetres.data, millimetres.bytes,
                                  cudaMemcpyDeviceToHost, stream));
        CudaCheck(cudaMemcpyAsync(normal.data, normals.data, normals.bytes, cudaMemcpyDeviceToHost,
                                  stream));
        cv::Mat semantic;
        if (monocular) {
          semantic.create(h, w, CV_8UC1);
          CudaCheck(cudaMemcpyAsync(semantic.data, limit.data, limit.bytes, cudaMemcpyDeviceToHost,
                                    stream));
        }
        CudaCheck(cudaStreamSynchronize(stream));
        // MoGe faces replace the maps; the LAS depth only scales them and gates their depth. The
        // GPU estimates the faces here; the fit and encoding run with the write on the CPU.
        std::vector<FaceEstimate> estimates;
        cv::Mat allowed;
        if (monocular) {
          allowed = static_mask.pixels.clone();
          allowed.setTo(0, semantic);
          estimates = estimator->Estimate(*lens, static_mask.faces);
        }
        const MonocularLens* plan = monocular ? &*lens : nullptr;
        const auto& name = scene.images[id].name;
        const auto dp = output / "depths" / fs::u8path(name + ".png"),
                   np = output / "normals" / fs::u8path(name + ".png");
        fs::create_directories(dp.parent_path());
        fs::create_directories(np.parent_path());
        if (writes.size() >= Lookahead)
          finish();
        writes.push_back(std::async(std::launch::async, [=, estimates = std::move(estimates),
                                                         static_pixels = static_mask.count]() mutable {
          Json monocular_row;
          if (plan) {
            const auto fit = FitFaceScales(*plan, estimates, depth, normal);
            const auto maps = GatherFaces(*plan, estimates, [&](const FaceSample& sample) {
              return fit.LogScale(sample, plan->size);
            });
            cv::Mat monocular_depth, monocular_normal;
            monocular_row = EncodeMonocular(*plan, maps, depth, normal, allowed, fit.accepted,
                                            MinIncidence, monocular_depth, monocular_normal);
            Json faces = Json::array();
            for (const auto& estimate : estimates)
              faces.push_back(estimate.ToJson());
            monocular_row["fit"] = fit.ToJson();
            monocular_row["faces"] = faces;
            depth = monocular_depth;
            normal = monocular_normal;
          }
          const auto valid = Counts(depth, normal, !monocular);
          WriteImage(dp, depth);
          WriteImage(np, normal);
          Json row = {{"image_index", id}, {"name", name}, {"width", w}, {"height", h},
                      {"depth_size", fs::file_size(dp)}, {"normal_size", fs::file_size(np)},
                      {"static_pixels", static_pixels}, {"depth_pixels", valid.first},
                      {"normal_pixels", valid.second}};
          if (monocular)
            row["monocular"] = monocular_row;
          return Written{id, std::move(row)};
        }));
      }
    }
  }
  while (!writes.empty())
    finish();
  if (!depth_pixels || !normal_pixels)
    throw std::runtime_error("LAS supervision has no reliable depth or normal pixels");
  Json images = Json::array();
  for (const auto& row : rows)
    if (!row.is_null())
      images.push_back(row);
  if (request != SupervisionRequest(scene, model, masks_dir, options, motion != nullptr))
    throw std::runtime_error("LAS or geometry exclusion masks changed during supervision export");
  points.Finish();
  Json summary = {{"version", Version},
          {"request", request},
          {"source", "valid original LAS records"},
          {"projection", motion ? "rolling_shutter_readout_line" : "global_shutter"},
          {"accepted_records", points.Count()},
          {"excluded_records", points.Excluded()},
          {"point_cache", points.Profile()},
          {"depth_format", "uint16_png_mm"},
          {"depth_type", "ray_distance"},
          {"invalid_depth", 0},
          {"normal_format", "rgb8_png"},
          {"normal_frame", "opencv_camera_facing_camera"},
          {"normal_encoding", "round(127.5 * (n + 1))"},
          {"invalid_normal_rgb", {0, 0, 0}},
          {"depth_validity", "isolated_outlier_removal_occlusion_boundary_and_local_depth_support"},
          {"depth_support_pixels", 4},
          {"occlusion_step", "background beyond 1.1 x foreground + 0.05 m within 5x5"},
          {"occlusion_margin", {{"background_pixels", 2},
                                {"foreground_spread_m", OcclusionSpread},
                                {"foreground_pixels", "ceil(spread x fx / depth), at most 24"}}},
          {"semantic_margin_pixels", options.boundary},
          {"depth_rendering", "las_surfel_tangent_disks_blended_within_0.01_m_plus_1_percent_of_nearest"},
          {"normal_blend_minimum_agreement", MinNormalAgreement},
          {"normal_estimator", motion ? "las_knn_pca_surfel_readout_line" : "las_knn_pca_surfel"},
          {"surfels", points.Surfels()},
          {"minimum_abs_normal_dot_ray", MinIncidence},
          {"depth_pixels", depth_pixels},
          {"normal_pixels", normal_pixels},
          {"images", images},
          {"seconds", Seconds(started)}};
  if (monocular) {
    summary["version"] = MonocularVersion;
    summary["source"] = MonocularSource;
    summary["anchor_source"] = "valid original LAS records";
    summary["geometry_model"] = request.at("geometry");
    summary["lenses"] = lenses;
    summary["depth_rendering"] =
        "moge2 faces scaled per face to the LAS depth, blended in log distance";
    summary["depth_validity"] =
        "accepted face fit, static mask off the semantic margin, no blended depth step, "
        "agreement with trusted LAS depth";
    summary["normal_estimator"] = "moge2_vitl_faces";
    summary["normal_validity"] =
        "static mask off the semantic margin, no blended depth step, incidence";
    summary["monocular"] = {{"fit", "per-face 12 x 12 bilinear log-scale field, Cauchy IRLS in log "
                                    "distance, scale 0.03, inliers within 0.05, overlap weight 0.25, "
                                    "node first difference 0.1 x mean anchors per node, lattice 2 px"},
                            {"trusted_las", "LAS depth whose surfel normal is within 15 degrees of the "
                                            "MoGe normal; only it is fitted and vetoes MoGe depth"},
                            {"accept", "at least 2000 trusted LAS anchors, half of them inliers"},
                            {"depth_step_log", .04},
                            {"las_agreement", "within max(3%, 0.03 m)"},
                            {"normals_need_depth", false},
                            {"inference", geometry->Profile()}};
  }
  return summary;
}
void CheckLasSupervision(const fs::path& output, const Reconstruction& model,
                         const Json& photos, const Json& summary) {
  const bool monocular = summary.value("source", std::string()) == MonocularSource;
  if (summary.at("version") < 1 ||
      summary.at("version") > (monocular ? MonocularVersion : Version) ||
      summary.at("depth_type") != "ray_distance" ||
      summary.at("depth_format") != "uint16_png_mm" || summary.at("normal_format") != "rgb8_png" ||
      summary.at("normal_frame") != "opencv_camera_facing_camera" ||
      summary.at("images").size() != photos.size())
    throw std::runtime_error("Unexpected LAS supervision contract");
  uint64_t depth_total = 0, normal_total = 0;
  std::deque<std::future<std::pair<uint64_t, uint64_t>>> checks;
  auto finish = [&] {
    const auto counts = checks.front().get();
    checks.pop_front();
    depth_total += counts.first;
    normal_total += counts.second;
  };
  for (size_t i = 0; i < photos.size(); ++i) {
    const auto& image = photos[i];
    const auto& row = summary.at("images")[i];
    if (row.at("image_index") != image.at("image_index") || row.at("name") != image.at("name"))
      throw std::runtime_error("LAS supervision image identity differs");
    const auto camera = MapCamera(model.cameras[image.at("camera").get<size_t>()],
                                  summary.at("request").at("max_size"));
    const auto rel = fs::u8path(image.at("name").get<std::string>() + ".png");
    const auto dp = output / "depths" / rel, np = output / "normals" / rel;
    if (fs::file_size(dp) != row.at("depth_size").get<uint64_t>() ||
        fs::file_size(np) != row.at("normal_size").get<uint64_t>())
      throw std::runtime_error("LAS supervision file size differs");
    if (row.at("width") != camera.width || row.at("height") != camera.height)
      throw std::runtime_error("LAS supervision map dimensions or pixel types differ");
    if (checks.size() >= Lookahead)
      finish();
    checks.push_back(std::async(std::launch::async, [=, depths = row.at("depth_pixels").get<uint64_t>(),
                                                     normals = row.at("normal_pixels").get<uint64_t>()] {
      const auto depth = ReadImage(dp, cv::IMREAD_UNCHANGED),
                 normal = ReadImage(np, cv::IMREAD_UNCHANGED);
      if (depth.type() != CV_16UC1 || normal.type() != CV_8UC3 || depth.size() != normal.size() ||
          depth.cols != camera.width || depth.rows != camera.height)
        throw std::runtime_error("LAS supervision map dimensions or pixel types differ");
      const auto counts = Counts(depth, normal, !monocular);
      if (counts.first != depths || counts.second != normals)
        throw std::runtime_error("LAS supervision valid pixel counts differ");
      return counts;
    }));
  }
  while (!checks.empty())
    finish();
  if (!depth_total || !normal_total || depth_total != summary.at("depth_pixels").get<uint64_t>() ||
      normal_total != summary.at("normal_pixels").get<uint64_t>())
    throw std::runtime_error("LAS supervision totals differ");
}
}  // namespace geosun

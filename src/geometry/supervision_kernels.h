#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace geosun::supervision_gpu {
constexpr double MinIncidence = .3, MaxDepth = 65.535;
// Half of FieldOfViewDegrees in radians, kept as the published constant.
constexpr double MaxTheta = 1.396263401595464;
// Foreground occlusion margin covers the measured spread of LAS silhouettes.
constexpr float OcclusionSpread = .01f;
constexpr int MaxOcclusionMargin = 24;
// LAS surfels: K nearest neighbours within the hashed-grid reach set the point spacing and the
// footprint; the tangent plane is a trimmed PCA over a ball of at least NormalRadius.
constexpr int Neighbours = 16, MinNeighbours = 10, NormalRings = 2;
constexpr float NormalCell = .03f, NormalRadius = .03f, PlaneBand = .02f, PlaneRms = .01f,
                FootprintScale = 2.f, MaxFootprint = .05f, StraySpacing = 2.f;
// Footprints reach at most this many map pixels from the projected centre.
constexpr int MaxSplat = 8;
constexpr float MinNormalAgreement = .95f;
// raw holds the LAS integer coordinates; normal is the world-frame code (0: none) and radius
// the tangent footprint disk radius in metres (0: centre pixel only).
struct Surfel {
  int3 raw;
  unsigned normal;
  float radius;
};
// motion holds the readout rotation and the readout translation (readout time times velocity)
// of a rolling-shutter image, both in the reference camera frame; see ShutterMotion.
// max_q2, min_qz_slope and qz_slack bound points before rolling-shutter iteration with no
// false negatives.
struct Projection {
  double rotation[9], translation[3], params[8], scales[3], motion[6], max_q2, min_qz_slope,
      qz_slack;
  int rolling, axis;
};
// Per-pixel sums of the surfels within BlendBand of the nearest one: weights, weighted ray
// distances, and the weighted normals with their own weight.
struct Blend {
  float weight, depth, nx, ny, nz, normals;
};
struct SurfelStats {
  uint64_t normals = 0;
  std::vector<float> radius, rms;
};

// Surfels of the slab's first own points; the rest of points is the halo.
std::vector<Surfel> SlabSurfels(const std::vector<int3>& points, size_t own, int3 origin,
                                float3 scale, cudaStream_t stream, SurfelStats& stats);
void Project(const Surfel* points, size_t count, size_t images, const Projection* cameras,
             const float3* rays, float* nearest, Blend* blends, int width, int height,
             bool accumulate, cudaStream_t stream);
void ResolveNearest(float* depths, size_t count, cudaStream_t stream);
void EncodeMaps(const Blend* blends, int w, int h, int boundary, float focal, const uint8_t* keep,
                const float3* rays, float* raw_depth, unsigned* codes, float* filtered,
                float* scratch, uint8_t* limit, uint8_t* dilated, float* seeds, uint8_t* edge,
                uint16_t* millimetres, uchar3* normals, cudaStream_t stream);
}  // namespace geosun::supervision_gpu

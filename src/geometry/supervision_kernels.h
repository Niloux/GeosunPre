#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace geosun::supervision_gpu {
constexpr double MaxDepth = 65.535;
// Half of FieldOfViewDegrees in radians, kept as the published constant.
constexpr double MaxTheta = 1.396263401595464;
// LAS surfels: K nearest neighbours within the hashed-grid reach set the point spacing and the
// footprint; the tangent plane is a trimmed PCA over a ball of at least NormalRadius.
constexpr int Neighbours = 16, MinNeighbours = 10, NormalRings = 2;
constexpr float NormalCell = .03f, NormalRadius = .03f, PlaneBand = .02f, PlaneRms = .01f,
                FootprintScale = 2.f, MaxFootprint = .05f, StraySpacing = 2.f;
// Footprints reach at most this many map pixels from the projected centre.
constexpr int MaxSplat = 8;
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
struct SurfelStats {
  uint64_t normals = 0;
  std::vector<float> radius, rms;
};

// Surfels of the slab's first own points; the rest of points is the halo.
std::vector<Surfel> SlabSurfels(const std::vector<int3>& points, size_t own, int3 origin,
                                float3 scale, cudaStream_t stream, SurfelStats& stats);
void ProjectNearest(const Surfel* points, size_t count, size_t images, const Projection* cameras,
                    const float3* rays, float* nearest, int width, int height, cudaStream_t stream);
void ResolveNearest(float* depths, size_t count, cudaStream_t stream);
void SampleMogeFaces(const uint8_t* prefiltered, int pre_w, int pre_h, const double* source_params,
                     int src_w, int src_h, double source_radius, double sx, double sy,
                     const double* to_camera, double3 omega, int rolling, int axis, int faces,
                     int face_size, double focal, float* out_chw, uint8_t* out_face_valid,
                     cudaStream_t stream);
void SampleMogeShiftGrid(const float* points, const float* mask, const uint8_t* face_valid,
                         int faces, int size, float4* samples, cudaStream_t stream);
void SolveMogeShifts(const float4* samples, int faces, int size, double* shifts, uint8_t* shift_ok,
                     cudaStream_t stream);
void BuildMogeFaceEstimates(const float* points, const float* normals, const float* mask,
                            const uint8_t* face_valid, const double* shifts,
                            const uint8_t* shift_ok, const float* metric_scales,
                            const double* to_camera, int faces, int size, double focal,
                            float* out_distance, float* out_normal, uint32_t* valid_counts,
                            cudaStream_t stream);
}  // namespace geosun::supervision_gpu

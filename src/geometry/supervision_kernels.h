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

struct ConflictView {
  double rotation[9], translation[3], params[8], motion_rot[3], motion_trans[3];
  int rolling, axis, camera, width, height, cw, ch;
};
struct ConflictPair {
  uint32_t pair_index;
  uint16_t slot_a, slot_b;
  uint32_t id_a, id_b;
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
void SampleMogeFaces(const uint8_t* prefiltered, int pre_w, int pre_h, const double* source_params,
                     int src_w, int src_h, double source_radius, double sx, double sy,
                     const double* to_camera, double3 omega, int rolling, int axis, int faces,
                     int face_size, double focal, float* out_chw, uint8_t* out_face_valid,
                     cudaStream_t stream);
void SampleMogeShiftGrid(const float* points, const float* mask, const uint8_t* face_valid,
                         int size, float4* samples, cudaStream_t stream);
void BuildMogeFaceEstimates(const float* points, const float* normals, const float* mask,
                            const uint8_t* face_valid, const double* shifts,
                            const uint8_t* shift_ok, const float* metric_scales,
                            const double* to_camera, int faces, int size, double focal,
                            float* out_distance, float* out_normal, uint32_t* valid_counts,
                            cudaStream_t stream);
void CheckConflictPairs(const uint16_t* depths_base, size_t depth_stride, const ConflictView* views,
                        const double3* rays_cam0, const double3* rays_cam1,
                        const ConflictPair* pairs, int num_pairs, int max_cells, int lattice_step,
                        double view_limit, double tolerance, uint8_t* pair_own,
                        uint8_t* pair_other, unsigned int* tile_conflicts, unsigned long long* stats,
                        cudaStream_t stream);
}  // namespace geosun::supervision_gpu

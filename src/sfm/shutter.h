#pragma once
#include "alignment/alignment.h"

namespace geosun {
struct ShutterCorrection {
  // Accepted: cameras, points and every observation in the image the reference (middle-row) pose
  // would have recorded, so later stages and COLMAP use the ordinary global-shutter projection.
  Reconstruction candidate;
  ShutterMotion motion;
  Json report;
  bool accepted = false;
};
// Estimates per-image readout rotation and per-lens readout time jointly with cameras, points and
// the shared intrinsics, the readout translation following the navigation velocity. Held-out
// tracks, the LAS surface fit and agreement between the readout time and the camera rotation rate
// decide acceptance.
ShutterCorrection CorrectRollingShutter(const Scene& scene, const Reconstruction& source,
                                        int workers, const AlignmentLimits& limits,
                                        int ba_workers);
Json MotionToJson(const ShutterMotion& motion, const Reconstruction& model);
ShutterMotion MotionFromJson(const Json& value, size_t images);
// Maps the observations of a corrected model back into the original rolling-shutter photos and
// sets each point error to its rolling-shutter reprojection.
Reconstruction OriginalPhotoObservations(const Reconstruction& corrected,
                                         const ShutterMotion& motion, int workers);
// Mean rolling-shutter reprojection error of a point over its track.
double ShutterPointError(const Reconstruction& model, const ShutterMotion& motion,
                         const ModelPoint& point);
// rolling_shutter.json of a delivery with original photos, keyed by COLMAP camera and image IDs.
Json DeliveryShutterJson(const Scene& scene, const Reconstruction& model,
                         const ShutterMotion& motion);
ShutterMotion DeliveryShutterFromJson(const Json& value, const Scene& scene,
                                      const Reconstruction& model);
}  // namespace geosun

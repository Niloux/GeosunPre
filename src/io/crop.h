#pragma once
#include "geometry/model.h"

namespace geosun {
// One CropWindow per lens that keeps every non-zero pixel of the static masks of its registered
// photos (cache masks named by image index). Its left and top edges sit on the photos' JPEG MCU
// grid, mcu being the largest MCU edge, so the photos crop without re-encoding. Along the readout
// axis it is centred on the image, so the readout fraction only scales: a pixel's fraction in the
// window is s' = s N / N', with N and N' the sizes along that axis.
std::array<CropWindow, 2> ValidDomainCrop(const Scene& scene, const Reconstruction& model,
                                          const fs::path& masks, int readout_axis, int mcu,
                                          int workers);
// The scene's cameras (the initial calibration, used for angle statistics) cut to the windows.
Scene CropScene(const Scene& scene, const std::array<CropWindow, 2>& windows);
// Cameras and observations moved into the windows; point errors are recomputed with the cropped
// projection (rolling shutter when motion is given).
Reconstruction CropModel(const Reconstruction& model, const std::array<CropWindow, 2>& windows,
                         const ShutterMotion* motion);
// The same readout motion expressed in readout fractions of the cropped images: rotation and
// readout time scale by N' / N, velocity is unchanged.
ShutterMotion CropMotion(const ShutterMotion& motion, const Reconstruction& uncropped,
                         const std::array<CropWindow, 2>& windows);
// Largest JPEG MCU edge in pixels, from the frame header.
int JpegMcuSize(const fs::path& path);
// Lossless crop of DCT coefficients; the window must start on the MCU grid. Markers such as EXIF
// are copied, so orientation and exposure tags survive.
void CropJpeg(const fs::path& source, const fs::path& destination, const CropWindow& window);
// Exact crop of a single-channel PNG mask.
void CropMask(const fs::path& source, const fs::path& destination, const CropWindow& window);
}  // namespace geosun

#include "frontend/inference.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <math_constants.h>

namespace geosun {
namespace {
__global__ void Normalize(const uint8_t* image, float* out, int count, float3 mean,
                          float3 deviation) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  out[i] = (image[3 * i] / 255.f - mean.x) / deviation.x;
  out[count + i] = (image[3 * i + 1] / 255.f - mean.y) / deviation.y;
  out[2 * count + i] = (image[3 * i + 2] / 255.f - mean.z) / deviation.z;
}
__global__ void Candidates(const float* logits, float* score, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  const int x = i % w, y = i / w;
  const float value = logits[i];
  bool maximum = true;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && logits[yy * w + xx] > value)
        maximum = false;
    }
  score[i] = maximum ? value : -CUDART_INF_F;
}
}  // namespace
Tensor ImageTensor(const cv::Mat& rgb, const std::array<float, 3>& mean,
                   const std::array<float, 3>& deviation) {
  if (rgb.type() != CV_8UC3 || !rgb.isContinuous())
    throw std::runtime_error("Expected contiguous RGB8 image");
  const size_t count = size_t(rgb.rows) * rgb.cols;
  DeviceBuffer source(count * 3);
  CudaCheck(cudaMemcpy(source.data, rgb.data, count * 3, cudaMemcpyHostToDevice));
  Tensor t;
  t.shape = {1, 3, rgb.rows, rgb.cols};
  t.device = std::make_shared<DeviceBuffer>(count * 3 * sizeof(float));
  Normalize<<<unsigned((count + 255) / 256), 256>>>(
      static_cast<const uint8_t*>(source.data), static_cast<float*>(t.device->data), int(count),
      make_float3(mean[0], mean[1], mean[2]),
      make_float3(deviation[0], deviation[1], deviation[2]));
  CudaCheck(cudaGetLastError());
  CudaCheck(cudaDeviceSynchronize());
  return t;
}
std::vector<std::array<float, 3>> DetectKeypoints(const Tensor& logits, int w, int h,
                                                  size_t maximum) {
  if (logits.Count() != size_t(w) * h)
    throw std::runtime_error("Unexpected detector output dimensions");
  DeviceBuffer source(logits.bytes.size()), scores(logits.bytes.size());
  CudaCheck(
      cudaMemcpy(source.data, logits.bytes.data(), logits.bytes.size(), cudaMemcpyHostToDevice));
  Candidates<<<(w * h + 255) / 256, 256>>>(static_cast<const float*>(source.data),
                                           static_cast<float*>(scores.data), w, h);
  CudaCheck(cudaGetLastError());
  std::vector<float> values(size_t(w) * h);
  CudaCheck(cudaMemcpy(values.data(), scores.data, logits.bytes.size(), cudaMemcpyDeviceToHost));
  std::vector<uint32_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  maximum = std::min(maximum, indices.size());
  const auto cmp = [&](uint32_t a, uint32_t b) {
    return values[a] > values[b] || (values[a] == values[b] && a < b);
  };
  if (maximum < indices.size())
    std::nth_element(indices.begin(), indices.begin() + maximum, indices.end(), cmp);
  indices.resize(maximum);
  std::sort(indices.begin(), indices.end(), cmp);
  const auto* data = logits.Floats();
  const double highest = *std::max_element(data, data + values.size());
  double total = 0;
  for (size_t i = 0; i < values.size(); ++i)
    total += std::exp(double(data[i]) - highest);
  std::vector<std::array<float, 3>> out;
  for (uint32_t id : indices) {
    if (!std::isfinite(values[id]))
      break;
    const int x = int(id % w), y = int(id / w);
    double local_max = -std::numeric_limits<double>::infinity();
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int xx = x + dx, yy = y + dy;
        const double v = xx >= 0 && yy >= 0 && xx < w && yy < h ? data[yy * w + xx] : 0;
        local_max = std::max(local_max, v / .5);
      }
    double mass = 0, ox = 0, oy = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int xx = x + dx, yy = y + dy;
        const double v = xx >= 0 && yy >= 0 && xx < w && yy < h ? data[yy * w + xx] : 0;
        const double p = std::exp(v / .5 - local_max);
        mass += p;
        ox += dx * p;
        oy += dy * p;
      }
    out.push_back({float(2 * (x + .5 + ox / mass) / w - 1), float(2 * (y + .5 + oy / mass) / h - 1),
                   float(std::exp(double(data[id]) - highest) / total)});
  }
  return out;
}
}  // namespace geosun

#include "frontend/inference.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <math_constants.h>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

namespace geosun {
namespace {
struct Peak {
  uint32_t index;
  float value;
};
struct ExpShift {
  const float* highest;
  __device__ double operator()(float value) const { return exp(double(value) - *highest); }
};
__global__ void Candidates(const float* logits, Peak* peaks, uint32_t* count, int w, int h) {
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
  if (maximum && isfinite(value))
    peaks[atomicAdd(count, 1u)] = {uint32_t(i), value};
}
__global__ void Subpixel(const float* logits, const Peak* peaks, float* output, int count, int w,
                         int h, const float* highest, const double* total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const uint32_t id = peaks[i].index;
  const int x = int(id % w), y = int(id / w);
  double local_max = -CUDART_INF;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const double value = xx >= 0 && yy >= 0 && xx < w && yy < h ? logits[yy * w + xx] : 0;
      local_max = fmax(local_max, value / .5);
    }
  double mass = 0, ox = 0, oy = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx, yy = y + dy;
      const double value = xx >= 0 && yy >= 0 && xx < w && yy < h ? logits[yy * w + xx] : 0;
      const double p = exp(value / .5 - local_max);
      mass += p;
      ox += dx * p;
      oy += dy * p;
    }
  output[3 * i] = float(2 * (x + .5 + ox / mass) / w - 1);
  output[3 * i + 1] = float(2 * (y + .5 + oy / mass) / h - 1);
  output[3 * i + 2] = float(exp(double(peaks[i].value) - *highest) / *total);
}
}  // namespace
std::vector<std::array<float, 3>> DetectKeypoints(const Tensor& logits, int w, int h,
                                                  size_t maximum, cudaStream_t stream) {
  if (logits.Count() != size_t(w) * h)
    throw std::runtime_error("Unexpected detector output dimensions");
  const size_t size = size_t(w) * h;
  DeviceBuffer source(logits.device ? 0 : size * sizeof(float));
  if (!logits.device)
    CudaCheck(cudaMemcpyAsync(source.data, logits.Floats(), size * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  const auto* data = static_cast<const float*>(logits.device ? logits.device->data : source.data);
  DeviceBuffer peaks(size * sizeof(Peak)), counter(sizeof(uint32_t)), highest(sizeof(float)),
      total(sizeof(double));
  size_t max_bytes = 0, sum_bytes = 0;
  CudaCheck(cub::DeviceReduce::Max(nullptr, max_bytes, data, static_cast<float*>(highest.data),
                                   int(size), stream));
  auto exponent = cub::TransformInputIterator<double, ExpShift, const float*>(
      data, ExpShift{static_cast<const float*>(highest.data)});
  CudaCheck(cub::DeviceReduce::Sum(nullptr, sum_bytes, exponent, static_cast<double*>(total.data),
                                   int(size), stream));
  DeviceBuffer workspace(std::max(max_bytes, sum_bytes));
  CudaCheck(cub::DeviceReduce::Max(workspace.data, max_bytes, data,
                                   static_cast<float*>(highest.data), int(size), stream));
  CudaCheck(cub::DeviceReduce::Sum(workspace.data, sum_bytes, exponent,
                                   static_cast<double*>(total.data), int(size), stream));
  CudaCheck(cudaMemsetAsync(counter.data, 0, sizeof(uint32_t), stream));
  Candidates<<<(w * h + 255) / 256, 256, 0, stream>>>(data, static_cast<Peak*>(peaks.data),
                                                      static_cast<uint32_t*>(counter.data), w, h);
  CudaCheck(cudaGetLastError());
  uint32_t count;
  double mass;
  CudaCheck(cudaMemcpyAsync(&count, counter.data, sizeof(count), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(&mass, total.data, sizeof(mass), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  if (!std::isfinite(mass) || mass <= 0)
    throw std::runtime_error("Nonfinite detector density");
  std::vector<Peak> candidates(count);
  CudaCheck(cudaMemcpyAsync(candidates.data(), peaks.data, count * sizeof(Peak),
                            cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  maximum = std::min(maximum, candidates.size());
  const auto cmp = [](const Peak& a, const Peak& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  };
  if (maximum < candidates.size())
    std::nth_element(candidates.begin(), candidates.begin() + maximum, candidates.end(), cmp);
  candidates.resize(maximum);
  std::sort(candidates.begin(), candidates.end(), cmp);
  std::vector<std::array<float, 3>> out(maximum);
  if (!maximum)
    return out;
  DeviceBuffer output(maximum * sizeof(out[0]));
  CudaCheck(cudaMemcpyAsync(peaks.data, candidates.data(), maximum * sizeof(Peak),
                            cudaMemcpyHostToDevice, stream));
  Subpixel<<<unsigned((maximum + 255) / 256), 256, 0, stream>>>(
      data, static_cast<const Peak*>(peaks.data), static_cast<float*>(output.data), int(maximum), w,
      h, static_cast<const float*>(highest.data), static_cast<const double*>(total.data));
  CudaCheck(cudaGetLastError());
  CudaCheck(cudaMemcpyAsync(out.data(), output.data, output.bytes, cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  return out;
}
}  // namespace geosun

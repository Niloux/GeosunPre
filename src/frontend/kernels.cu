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
struct KeypointDetector::Implementation {
  std::unique_ptr<DeviceBuffer> source, peaks, counter, highest, total, workspace, output;
  void Ensure(std::unique_ptr<DeviceBuffer>& buf, size_t bytes) {
    if (!buf || buf->bytes < bytes)
      buf = std::make_unique<DeviceBuffer>(bytes);
  }
};
KeypointDetector::KeypointDetector() : impl_(std::make_unique<Implementation>()) {}
KeypointDetector::~KeypointDetector() = default;
std::vector<std::array<float, 3>> KeypointDetector::Detect(const Tensor& logits, int w, int h,
                                                           size_t maximum, cudaStream_t stream) {
  if (logits.Count() != size_t(w) * h)
    throw std::runtime_error("Unexpected detector output dimensions");
  const size_t size = size_t(w) * h;
  auto& d = *impl_;
  if (!logits.device) {
    d.Ensure(d.source, size * sizeof(float));
    CudaCheck(cudaMemcpyAsync(d.source->data, logits.Floats(), size * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  }
  const auto* data = static_cast<const float*>(logits.device ? logits.device->data : d.source->data);
  d.Ensure(d.peaks, size * sizeof(Peak));
  d.Ensure(d.counter, sizeof(uint32_t));
  d.Ensure(d.highest, sizeof(float));
  d.Ensure(d.total, sizeof(double));
  size_t max_bytes = 0, sum_bytes = 0;
  CudaCheck(cub::DeviceReduce::Max(nullptr, max_bytes, data, static_cast<float*>(d.highest->data),
                                   int(size), stream));
  auto exponent = cub::TransformInputIterator<double, ExpShift, const float*>(
      data, ExpShift{static_cast<const float*>(d.highest->data)});
  CudaCheck(cub::DeviceReduce::Sum(nullptr, sum_bytes, exponent, static_cast<double*>(d.total->data),
                                   int(size), stream));
  d.Ensure(d.workspace, std::max(max_bytes, sum_bytes));
  CudaCheck(cub::DeviceReduce::Max(d.workspace->data, max_bytes, data,
                                   static_cast<float*>(d.highest->data), int(size), stream));
  CudaCheck(cub::DeviceReduce::Sum(d.workspace->data, sum_bytes, exponent,
                                   static_cast<double*>(d.total->data), int(size), stream));
  CudaCheck(cudaMemsetAsync(d.counter->data, 0, sizeof(uint32_t), stream));
  Candidates<<<(w * h + 255) / 256, 256, 0, stream>>>(data, static_cast<Peak*>(d.peaks->data),
                                                      static_cast<uint32_t*>(d.counter->data), w, h);
  CudaCheck(cudaGetLastError());
  uint32_t count;
  double mass;
  CudaCheck(cudaMemcpyAsync(&count, d.counter->data, sizeof(count), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaMemcpyAsync(&mass, d.total->data, sizeof(mass), cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  if (!std::isfinite(mass) || mass <= 0)
    throw std::runtime_error("Nonfinite detector density");
  std::vector<Peak> candidates(count);
  CudaCheck(cudaMemcpyAsync(candidates.data(), d.peaks->data, count * sizeof(Peak),
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
  const size_t out_bytes = maximum * sizeof(out[0]);
  d.Ensure(d.output, out_bytes);
  CudaCheck(cudaMemcpyAsync(d.peaks->data, candidates.data(), maximum * sizeof(Peak),
                            cudaMemcpyHostToDevice, stream));
  Subpixel<<<unsigned((maximum + 255) / 256), 256, 0, stream>>>(
      data, static_cast<const Peak*>(d.peaks->data), static_cast<float*>(d.output->data),
      int(maximum), w, h, static_cast<const float*>(d.highest->data),
      static_cast<const double*>(d.total->data));
  CudaCheck(cudaGetLastError());
  CudaCheck(cudaMemcpyAsync(out.data(), d.output->data, out_bytes, cudaMemcpyDeviceToHost, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  return out;
}
std::vector<std::array<float, 3>> DetectKeypoints(const Tensor& logits, int w, int h,
                                                  size_t maximum, cudaStream_t stream) {
  KeypointDetector detector;
  return detector.Detect(logits, w, h, maximum, stream);
}
}  // namespace geosun

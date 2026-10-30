#include "frontend/preprocessing.h"
#include "io/scene.h"
#include <cstring>
#include <algorithm>

namespace geosun {
namespace {
struct PinnedBuffer {
  void* data = nullptr;
  size_t bytes = 0;
  ~PinnedBuffer() {
    if (data)
      cudaFreeHost(data);
  }
  void Reserve(size_t size) {
    if (size <= bytes)
      return;
    if (data)
      CudaCheck(cudaFreeHost(data));
    data = nullptr;
    CudaCheck(cudaHostAlloc(&data, size, cudaHostAllocDefault));
    bytes = size;
  }
};
void Reserve(std::shared_ptr<DeviceBuffer>& buffer, size_t bytes) {
  if (!buffer || buffer->bytes < bytes || buffer.use_count() > 1)
    buffer = std::make_shared<DeviceBuffer>(bytes);
}
struct Weights {
  DeviceBuffer offsets, starts, values;
  Weights(const std::vector<ResizeCoefficients>& rows)
      : offsets((rows.size() + 1) * sizeof(int)), starts(rows.size() * sizeof(int)), values([&] {
          size_t n = 0;
          for (const auto& row : rows)
            n += row.values.size();
          return n * sizeof(int);
        }()) {
    std::vector<int> host_offsets{0}, host_starts, host_values;
    for (const auto& row : rows) {
      host_starts.push_back(row.start);
      host_values.insert(host_values.end(), row.values.begin(), row.values.end());
      host_offsets.push_back(int(host_values.size()));
    }
    CudaCheck(cudaMemcpy(offsets.data, host_offsets.data(), offsets.bytes, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(starts.data, host_starts.data(), starts.bytes, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(values.data, host_values.data(), values.bytes, cudaMemcpyHostToDevice));
  }
};
struct ResizePlan {
  Weights horizontal, vertical;
  ResizePlan(int input_w, int input_h, int output_w, int output_h)
      : horizontal(ResizeWeights(input_w, output_w)), vertical(ResizeWeights(input_h, output_h)) {}
};
__device__ uint8_t RoundPixel(int64_t sum) {
  return uint8_t(max(int64_t(0), min(int64_t(255), sum >> 22)));
}
__global__ void Horizontal(const uint8_t* input, uint8_t* output, int input_w, int output_w,
                           int height, const int* offsets, const int* starts,
                           const int* coefficients) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output_w * height * 3)
    return;
  const int c = index % 3, x = (index / 3) % output_w, y = index / (3 * output_w);
  int64_t sum = int64_t(1) << 21;
  const int begin = offsets[x], end = offsets[x + 1];
  for (int k = begin; k < end; ++k)
    sum += int64_t(input[(y * input_w + starts[x] + k - begin) * 3 + c]) * coefficients[k];
  output[index] = RoundPixel(sum);
}
__global__ void VerticalNormalize(const uint8_t* input, float* output, int width, int height,
                                  const int* offsets, const int* starts, const int* coefficients,
                                  float3 mean, float3 deviation) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = width * height;
  if (index >= pixels * 3)
    return;
  const int c = index % 3, x = (index / 3) % width, y = index / (3 * width);
  int64_t sum = int64_t(1) << 21;
  const int begin = offsets[y], end = offsets[y + 1];
  for (int k = begin; k < end; ++k)
    sum += int64_t(input[((starts[y] + k - begin) * width + x) * 3 + c]) * coefficients[k];
  const float m = c == 0 ? mean.x : c == 1 ? mean.y : mean.z;
  const float d = c == 0 ? deviation.x : c == 1 ? deviation.y : deviation.z;
  output[c * pixels + index / 3] = (RoundPixel(sum) / 255.f - m) / d;
}
__global__ void NormalizePixels(const uint8_t* input, float* output, int pixels, float3 mean,
                                float3 deviation) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels)
    return;
  output[i] = (input[3 * i] / 255.f - mean.x) / deviation.x;
  output[pixels + i] = (input[3 * i + 1] / 255.f - mean.y) / deviation.y;
  output[2 * pixels + i] = (input[3 * i + 2] / 255.f - mean.z) / deviation.z;
}
}  // namespace
struct ImagePreprocessor::Implementation {
  cudaStream_t stream;
  cudaEvent_t copied;
  PinnedBuffer host;
  std::shared_ptr<DeviceBuffer> source, intermediate, normalized;
  std::map<std::array<int, 4>, std::unique_ptr<ResizePlan>> plans;
  int width = 0, height = 0;
  explicit Implementation(cudaStream_t value) : stream(value) {
    CudaCheck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  }
  ~Implementation() {
    cudaStreamSynchronize(stream);
    cudaEventDestroy(copied);
  }
};
ImagePreprocessor::ImagePreprocessor(cudaStream_t stream)
    : impl_(std::make_unique<Implementation>(stream)) {}
ImagePreprocessor::~ImagePreprocessor() = default;
void ImagePreprocessor::Upload(const cv::Mat& rgb) {
  if (rgb.type() != CV_8UC3 || !rgb.isContinuous())
    throw std::runtime_error("Expected contiguous RGB8 image");
  auto& p = *impl_;
  CudaCheck(cudaEventSynchronize(p.copied));
  const size_t bytes = rgb.total() * 3;
  p.host.Reserve(bytes);
  Reserve(p.source, bytes);
  std::memcpy(p.host.data, rgb.data, bytes);
  CudaCheck(cudaMemcpyAsync(p.source->data, p.host.data, bytes, cudaMemcpyHostToDevice, p.stream));
  CudaCheck(cudaEventRecord(p.copied, p.stream));
  p.width = rgb.cols;
  p.height = rgb.rows;
}
Tensor ImagePreprocessor::Normalize(int width, int height, const std::array<float, 3>& mean,
                                    const std::array<float, 3>& deviation) {
  auto& p = *impl_;
  const size_t count = size_t(width) * height;
  Reserve(p.normalized, count * 3 * sizeof(float));
  const auto m = make_float3(mean[0], mean[1], mean[2]),
             d = make_float3(deviation[0], deviation[1], deviation[2]);
  if (width == p.width && height == p.height)
    NormalizePixels<<<unsigned((count + 255) / 256), 256, 0, p.stream>>>(
        static_cast<const uint8_t*>(p.source->data), static_cast<float*>(p.normalized->data),
        int(count), m, d);
  else {
    const std::array<int, 4> key{p.width, p.height, width, height};
    auto found = p.plans.find(key);
    if (found == p.plans.end()) {
      if (p.plans.size() == 4)
        p.plans.clear();
      found = p.plans.emplace(key, std::make_unique<ResizePlan>(p.width, p.height, width, height))
                  .first;
    }
    const auto& plan = *found->second;
    const size_t horizontal = size_t(p.height) * width * 3;
    Reserve(p.intermediate, horizontal);
    Horizontal<<<unsigned((horizontal + 255) / 256), 256, 0, p.stream>>>(
        static_cast<const uint8_t*>(p.source->data), static_cast<uint8_t*>(p.intermediate->data),
        p.width, width, p.height, static_cast<const int*>(plan.horizontal.offsets.data),
        static_cast<const int*>(plan.horizontal.starts.data),
        static_cast<const int*>(plan.horizontal.values.data));
    CudaCheck(cudaGetLastError());
    VerticalNormalize<<<unsigned((count * 3 + 255) / 256), 256, 0, p.stream>>>(
        static_cast<const uint8_t*>(p.intermediate->data), static_cast<float*>(p.normalized->data),
        width, height, static_cast<const int*>(plan.vertical.offsets.data),
        static_cast<const int*>(plan.vertical.starts.data),
        static_cast<const int*>(plan.vertical.values.data), m, d);
  }
  CudaCheck(cudaGetLastError());
  Tensor result;
  result.shape = {1, 3, height, width};
  result.device = p.normalized;
  return result;
}
}  // namespace geosun

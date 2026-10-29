#include "frontend/preprocessing.h"
#include "io/scene.h"
#include <opencv2/imgproc.hpp>
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
struct Projection {
  DeviceBuffer coordinates, fractions, weights;
  Projection(const MaskView& view)
      : coordinates(view.x.total() * sizeof(short2)),
        fractions(view.x.total() * sizeof(uint16_t)),
        weights(view.x.total() * sizeof(float)) {
    cv::Mat coordinates_host, fractions_host;
    cv::convertMaps(view.x, view.y, coordinates_host, fractions_host, CV_16SC2);
    CudaCheck(cudaMemcpy(coordinates.data, coordinates_host.data, coordinates.bytes,
                         cudaMemcpyHostToDevice));
    CudaCheck(
        cudaMemcpy(fractions.data, fractions_host.data, fractions.bytes, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(weights.data, view.weight.data, weights.bytes, cudaMemcpyHostToDevice));
  }
};
__device__ float Sample(const float* input, int x, int y, int channel) {
  return x >= 0 && x < 512 && y >= 0 && y < 512 ? input[channel * 512 * 512 + y * 512 + x] : 0;
}
__global__ void Fuse(const float* probabilities, const short2* coordinates,
                     const uint16_t* fractions, const float* weights, float2* sum, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto p = coordinates[i];
  const auto f = fractions[i];
  const float fx = (f % 32) / 32.f, fy = (f / 32) / 32.f;
  const float w[4] = {(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx};
  float value[2];
  for (int c = 0; c < 2; ++c) {
    float v = __fadd_rn(__fmul_rn(Sample(probabilities, p.x, p.y, c), w[0]),
                        __fmul_rn(Sample(probabilities, p.x + 1, p.y, c), w[1]));
    v = __fadd_rn(v, __fmul_rn(Sample(probabilities, p.x, p.y + 1, c), w[2]));
    v = __fadd_rn(v, __fmul_rn(Sample(probabilities, p.x + 1, p.y + 1, c), w[3]));
    value[c] = __fmul_rn(v, weights[i]);
  }
  sum[i].x = __fadd_rn(sum[i].x, value[0]);
  sum[i].y = __fadd_rn(sum[i].y, value[1]);
}
__global__ void Classify(const float2* sum, const float* weights, uchar2* classes, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float reciprocal = weights[i] > 0 ? 1.f / weights[i] : 0;
  classes[i] = make_uchar2(__fmul_rn(sum[i].x, reciprocal) > .90 ? 255 : 0,
                           __fmul_rn(sum[i].y, reciprocal) > .55 ? 255 : 0);
}
__global__ void Morphology(const uchar2* input, uchar2* output, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height)
    return;
  const int x = i % width, y = i / width;
  uint8_t sky = 255, person = 0;
  for (int dy = -3; dy <= 3; ++dy) {
    const int half = abs(dy) == 3 ? 0 : abs(dy) == 2 ? 2 : 3;
    for (int dx = -half; dx <= half; ++dx) {
      const int xx = x + dx, yy = y + dy;
      if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
        const auto value = input[yy * width + xx];
        sky = min(sky, value.x);
        person = max(person, value.y);
      }
    }
  }
  output[i] = make_uchar2(sky, person);
}
__global__ void Composite(const uchar2* classes, const uint8_t* valid, const float* weights,
                          uint8_t* output, int width, int height, int small_width, int small_height,
                          double sx, double sy, bool persons) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height)
    return;
  const int x = i % width, y = i / width;
  const int xx = min(__double2int_rd(x * sx), small_width - 1),
            yy = min(__double2int_rd(y * sy), small_height - 1);
  const int cell = yy * small_width + xx;
  const auto value = classes[cell];
  output[i] = valid[i] && weights[cell] > 0 ? (value.y && persons ? 0 : value.x ? 128 : 255) : 0;
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
struct MaskFusion::Implementation {
  cudaStream_t stream;
  int width, height, small_width, small_height;
  bool persons;
  std::vector<std::unique_ptr<Projection>> projections;
  DeviceBuffer sum, weight, classes, morphed, valid, output;
  PinnedBuffer host;
  Implementation(const std::vector<MaskView>& views, const cv::Mat& total, const cv::Mat& domain,
                 cudaStream_t value, bool person_class)
      : stream(value),
        persons(person_class),
        width(domain.cols),
        height(domain.rows),
        small_width(total.cols),
        small_height(total.rows),
        sum(total.total() * sizeof(float2)),
        weight(total.total() * sizeof(float)),
        classes(total.total() * sizeof(uchar2)),
        morphed(total.total() * sizeof(uchar2)),
        valid(domain.total()),
        output(domain.total()) {
    for (const auto& view : views)
      projections.push_back(std::make_unique<Projection>(view));
    CudaCheck(cudaMemcpy(weight.data, total.data, weight.bytes, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(valid.data, domain.data, valid.bytes, cudaMemcpyHostToDevice));
    host.Reserve(output.bytes);
  }
  ~Implementation() { cudaStreamSynchronize(stream); }
};
MaskFusion::MaskFusion(const std::vector<MaskView>& views, const cv::Mat& weight,
                       const cv::Mat& valid, cudaStream_t stream, bool persons)
    : impl_(std::make_unique<Implementation>(views, weight, valid, stream, persons)) {}
MaskFusion::~MaskFusion() = default;
void MaskFusion::Begin() {
  CudaCheck(cudaMemsetAsync(impl_->sum.data, 0, impl_->sum.bytes, impl_->stream));
}
void MaskFusion::Add(const Tensor& probabilities, size_t view) {
  if (!probabilities.device || probabilities.shape != std::vector<int64_t>{1, 2, 512, 512})
    throw std::runtime_error("Expected resident segmentation probabilities");
  auto& p = *impl_;
  const auto& map = *p.projections.at(view);
  const int count = p.small_width * p.small_height;
  Fuse<<<(count + 255) / 256, 256, 0, p.stream>>>(
      static_cast<const float*>(probabilities.device->data),
      static_cast<const short2*>(map.coordinates.data),
      static_cast<const uint16_t*>(map.fractions.data), static_cast<const float*>(map.weights.data),
      static_cast<float2*>(p.sum.data), count);
  CudaCheck(cudaGetLastError());
}
cv::Mat MaskFusion::Finish() {
  auto& p = *impl_;
  const int count = p.small_width * p.small_height;
  Classify<<<(count + 255) / 256, 256, 0, p.stream>>>(static_cast<const float2*>(p.sum.data),
                                                      static_cast<const float*>(p.weight.data),
                                                      static_cast<uchar2*>(p.classes.data), count);
  CudaCheck(cudaGetLastError());
  Morphology<<<(count + 255) / 256, 256, 0, p.stream>>>(static_cast<const uchar2*>(p.classes.data),
                                                        static_cast<uchar2*>(p.morphed.data),
                                                        p.small_width, p.small_height);
  CudaCheck(cudaGetLastError());
  Composite<<<(p.width * p.height + 255) / 256, 256, 0, p.stream>>>(
      static_cast<const uchar2*>(p.morphed.data), static_cast<const uint8_t*>(p.valid.data),
      static_cast<const float*>(p.weight.data), static_cast<uint8_t*>(p.output.data), p.width,
      p.height, p.small_width, p.small_height, double(p.small_width) / p.width,
      double(p.small_height) / p.height, p.persons);
  CudaCheck(cudaGetLastError());
  CudaCheck(cudaMemcpyAsync(p.host.data, p.output.data, p.output.bytes, cudaMemcpyDeviceToHost,
                            p.stream));
  CudaCheck(cudaStreamSynchronize(p.stream));
  cv::Mat result(p.height, p.width, CV_8UC1);
  std::memcpy(result.data, p.host.data, p.output.bytes);
  return result;
}
}  // namespace geosun

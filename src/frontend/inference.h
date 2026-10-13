#pragma once
#include "app/runtime.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <array>
#include <opencv2/core.hpp>

namespace geosun {
void CudaCheck(cudaError_t error);
struct DeviceBuffer {
  void* data = nullptr;
  size_t bytes = 0;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(const DeviceBuffer&) = delete;
};
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  std::shared_ptr<DeviceBuffer> device;
  int type = 0;  // 0=float32, 1=int64, 2=int32
  size_t Count() const;
  const float* Floats() const;
  int64_t Index(size_t i) const;
  static Tensor Float(std::vector<int64_t> shape, const float* data, size_t count);
  static Tensor Upload(std::vector<int64_t> shape, const float* data, size_t count);
};
Tensor ImageTensor(const cv::Mat& rgb, const std::array<float, 3>& mean = {0, 0, 0},
                   const std::array<float, 3>& std = {1, 1, 1});
std::vector<std::array<float, 3>> DetectKeypoints(const Tensor& logits, int width, int height,
                                                  size_t maximum, cudaStream_t stream = nullptr);
class Models {
  struct Implementation;
  std::unique_ptr<Implementation> impl_;

 public:
  explicit Models(const fs::path& root);
  ~Models();
  const Json& Manifest() const;
  std::vector<Tensor> Predict(const std::string& model, const std::vector<Tensor>& inputs,
                              bool device_output = false);
  cudaStream_t Stream() const;
  Json Profile();
};
Json GpuInfo();
}  // namespace geosun

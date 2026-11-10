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
// Page-locked host memory; model outputs are copied into reusable ones.
struct HostBuffer {
  void* data = nullptr;
  size_t bytes = 0;
  explicit HostBuffer(size_t bytes);
  ~HostBuffer();
  HostBuffer(const HostBuffer&) = delete;
};
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;  // host inputs made by Float
  std::shared_ptr<HostBuffer> host;  // host outputs of Predict, valid for host_bytes
  size_t host_bytes = 0;
  std::shared_ptr<DeviceBuffer> device;
  int type = 0;  // 0=float32, 1=int64, 2=int32
  size_t Count() const;
  const uint8_t* HostData() const { return host ? static_cast<const uint8_t*>(host->data) : bytes.data(); }
  size_t HostSize() const { return host ? host_bytes : bytes.size(); }
  const float* Floats() const;
  int64_t Index(size_t i) const;
  static Tensor Float(std::vector<int64_t> shape, const float* data, size_t count);
  static Tensor Upload(std::vector<int64_t> shape, const float* data, size_t count);
};
std::vector<std::array<float, 3>> DetectKeypoints(const Tensor& logits, int width, int height,
                                                  size_t maximum, cudaStream_t stream = nullptr);
class Models {
  struct Implementation;
  std::unique_ptr<Implementation> impl_;

 public:
  explicit Models(const fs::path& root);
  ~Models();
  const Json& Manifest() const;
  std::array<int, 2> DetectorSize(int width, int height) const;
  std::vector<Tensor> Predict(const std::string& model, const std::vector<Tensor>& inputs,
                              bool device_output = false);
  cudaStream_t Stream() const;
  Json Profile();
};
Json GpuInfo();
}  // namespace geosun

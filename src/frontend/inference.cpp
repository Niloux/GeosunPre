#include "frontend/inference.h"
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <fstream>
#include <cstring>
#include <iostream>
#include <deque>
#include <algorithm>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace geosun {
void CudaCheck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
}
DeviceBuffer::DeviceBuffer(size_t n) : bytes(n) {
  if (n)
    CudaCheck(cudaMalloc(&data, n));
}
DeviceBuffer::~DeviceBuffer() {
  if (data)
    cudaFree(data);
}
size_t Tensor::Count() const {
  size_t n = 1;
  for (auto d : shape) {
    if (d < 0)
      throw std::runtime_error("Unresolved tensor shape");
    n *= size_t(d);
  }
  return n;
}
const float* Tensor::Floats() const {
  if (type != 0 || bytes.size() != Count() * sizeof(float))
    throw std::runtime_error("Expected host float32 tensor");
  return reinterpret_cast<const float*>(bytes.data());
}
int64_t Tensor::Index(size_t i) const {
  if (i >= Count())
    throw std::runtime_error("Tensor index out of range");
  if (bytes.size() != Count() * (type == 1 ? 8 : 4))
    throw std::runtime_error("Expected host integer tensor");
  if (type == 1)
    return reinterpret_cast<const int64_t*>(bytes.data())[i];
  if (type == 2)
    return reinterpret_cast<const int32_t*>(bytes.data())[i];
  throw std::runtime_error("Expected integer tensor");
}
Tensor Tensor::Float(std::vector<int64_t> dims, const float* data, size_t count) {
  Tensor t;
  t.shape = std::move(dims);
  if (t.Count() != count)
    throw std::runtime_error("Tensor data/shape differ");
  t.bytes.resize(count * sizeof(float));
  if (count)
    std::memcpy(t.bytes.data(), data, t.bytes.size());
  return t;
}
Tensor Tensor::Upload(std::vector<int64_t> dims, const float* data, size_t count) {
  Tensor t;
  t.shape = std::move(dims);
  if (count > t.Count())
    throw std::runtime_error("Tensor data exceeds shape");
  t.device = std::make_shared<DeviceBuffer>(t.Count() * sizeof(float));
  if (count < t.Count())
    CudaCheck(cudaMemset(t.device->data, 0, t.device->bytes));
  if (count)
    CudaCheck(cudaMemcpy(t.device->data, data, count * sizeof(float), cudaMemcpyHostToDevice));
  return t;
}
namespace {
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity level, const char* message) noexcept override {
    if (level <= Severity::kWARNING)
      std::cerr << "TensorRT: " << message << std::endl;
  }
};
struct Engine {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::map<std::string, std::shared_ptr<DeviceBuffer>> buffers;
  void* Buffer(const std::string& name, size_t size) {
    auto& p = buffers[name];
    if (!p || p->bytes < size || p.use_count() > 1)
      p = std::make_shared<DeviceBuffer>(size);
    return p->data;
  }
};
struct Measurement {
  std::string model;
  cudaEvent_t begin = nullptr, end = nullptr;
  explicit Measurement(std::string name) : model(std::move(name)) {
    CudaCheck(cudaEventCreate(&begin));
    CudaCheck(cudaEventCreate(&end));
  }
  ~Measurement() {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }
};
struct InferenceTiming {
  size_t calls = 0;
  double load_seconds = 0, host_seconds = 0, gpu_seconds = 0;
};
}  // namespace
struct Models::Implementation {
  fs::path root;
  Json manifest;
  Logger logger;
#ifdef _WIN32
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::map<std::string, Engine> engines;
  cudaStream_t stream = nullptr;
  std::deque<std::unique_ptr<Measurement>> pending;
  std::map<std::string, InferenceTiming> timings;
  void Collect(bool wait) {
    while (!pending.empty()) {
      const auto& p = *pending.front();
      if (wait)
        CudaCheck(cudaEventSynchronize(p.end));
      else {
        const auto status = cudaEventQuery(p.end);
        if (status == cudaErrorNotReady)
          break;
        CudaCheck(status);
      }
      float milliseconds;
      CudaCheck(cudaEventElapsedTime(&milliseconds, p.begin, p.end));
      timings[p.model].gpu_seconds += milliseconds / 1000.;
      pending.pop_front();
    }
  }
  ~Implementation() {
    if (stream)
      cudaStreamSynchronize(stream);
    pending.clear();
    engines.clear();
    runtime.reset();
    if (stream)
      cudaStreamDestroy(stream);
#ifdef _WIN32
    if (library)
      FreeLibrary(library);
#else
    if (library)
      dlclose(library);
#endif
  }
};
Models::Models(const fs::path& root) : impl_(std::make_unique<Implementation>()) {
  impl_->root = root;
  impl_->manifest = ReadJson(root / "manifest.json");
  const int version = impl_->manifest.at("version");
  if (version < 1 || version > 3 || impl_->manifest.at("tensorrt") != "10.13.3.9")
    throw std::runtime_error("Models require the pinned TensorRT 10.13.3.9 runtime");
  CudaCheck(cudaSetDevice(0));
  const auto gpu = GpuInfo();
  const auto hardware = impl_->manifest.value("hardware_compatibility", std::string("none"));
  if (hardware == "ampere_plus") {
    if (gpu.at("compute_capability").get<std::array<int, 2>>() < std::array<int, 2>{8, 0})
      throw std::runtime_error("ampere_plus models require a GPU with compute capability >= 8.0");
  } else if (hardware != "none") {
    throw std::runtime_error("Unknown model hardware compatibility: " + hardware);
  } else if (impl_->manifest.at("gpu") != gpu.at("name")) {
    throw std::runtime_error("Model GPU differs; prepare a new model package for this GPU");
  }
  CudaCheck(cudaStreamCreate(&impl_->stream));
  using Create = void* (*)(void*, int32_t);
  Create create = nullptr;
#ifdef _WIN32
  const auto library = root / "runtime/nvinfer_10.dll";
  impl_->library =
      LoadLibraryExW(library.c_str(), nullptr,
                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (impl_->library)
    create =
        reinterpret_cast<Create>(GetProcAddress(impl_->library, "createInferRuntime_INTERNAL"));
#else
  impl_->library = dlopen((root / "runtime/libnvinfer.so.10").c_str(), RTLD_NOW | RTLD_LOCAL);
  if (impl_->library)
    create = reinterpret_cast<Create>(dlsym(impl_->library, "createInferRuntime_INTERNAL"));
#endif
  if (!create)
    throw std::runtime_error("Cannot load TensorRT runtime from " + (root / "runtime").u8string());
  impl_->runtime.reset(
      static_cast<nvinfer1::IRuntime*>(create(&impl_->logger, NV_TENSORRT_VERSION)));
  if (!impl_->runtime)
    throw std::runtime_error("Cannot create TensorRT runtime");
}
Models::~Models() = default;
const Json& Models::Manifest() const { return impl_->manifest; }
std::array<int, 2> Models::DetectorSize(int width, int height) const {
  const int longest = std::max(width, height);
  const std::array<int, 2> size{int(int64_t(width) * 1536 / longest) / 8 * 8,
                                int(int64_t(height) * 1536 / longest) / 8 * 8};
  if (impl_->manifest.at("version") == 1) {
    if (size != impl_->manifest.at("detector_size").get<std::array<int, 2>>())
      throw std::runtime_error(
          "Static detector resolution differs; upgrade the model package with prepare-models.ps1 "
          "-Models models/dynamic -OnnxFrom models");
  } else {
    const auto& profile = impl_->manifest.at("detector_profile");
    const auto minimum = profile.at("min").get<std::array<int, 2>>(),
               maximum = profile.at("max").get<std::array<int, 2>>();
    for (size_t axis = 0; axis < size.size(); ++axis)
      if (size[axis] < minimum[axis] || size[axis] > maximum[axis])
        throw std::runtime_error("Resized detector input is outside the model profile: " +
                                 std::to_string(size[0]) + "x" + std::to_string(size[1]));
  }
  return size;
}
cudaStream_t Models::Stream() const { return impl_->stream; }
Json Models::Profile() {
  impl_->Collect(true);
  Json result = Json::object();
  for (const auto& [name, t] : impl_->timings)
    result[name] = {{"calls", t.calls},
                    {"load_seconds", t.load_seconds},
                    {"host_seconds", t.host_seconds},
                    {"gpu_seconds", t.gpu_seconds}};
  return result;
}
std::vector<Tensor> Models::Predict(const std::string& name, const std::vector<Tensor>& inputs,
                                    bool device_output) {
  const auto started = Clock::now();
  impl_->Collect(impl_->pending.size() >= 32);
  auto found = impl_->engines.find(name);
  if (found == impl_->engines.end()) {
    const auto path = impl_->root / "engines" / (name + ".engine");
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Missing model engine: " + path.u8string());
    std::vector<uint8_t> data(fs::file_size(path));
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size()))
      throw std::runtime_error("Truncated model engine");
    Engine e;
    e.engine.reset(impl_->runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!e.engine)
      throw std::runtime_error("TensorRT engine incompatible with runtime/GPU: " + name);
    e.context.reset(e.engine->createExecutionContext());
    if (!e.context)
      throw std::runtime_error("Cannot allocate inference context");
    found = impl_->engines.emplace(name, std::move(e)).first;
    Log("Loaded " + name);
    impl_->timings[name].load_seconds += Seconds(started);
  }
  auto& e = found->second;
  const auto& spec = impl_->manifest.at("models").at(name);
  if (inputs.size() != spec.at("inputs").size())
    throw std::runtime_error("Wrong inference input count");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto key = spec.at("inputs")[i].get<std::string>();
    const auto& t = inputs[i];
    nvinfer1::Dims dims;
    dims.nbDims = int32_t(t.shape.size());
    for (int d = 0; d < dims.nbDims; ++d)
      dims.d[d] = t.shape[d];
    if (!e.context->setInputShape(key.c_str(), dims) ||
        e.engine->getTensorDataType(key.c_str()) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error("Unsupported model input: " + key);
    void* pointer = t.device ? t.device->data : e.Buffer(key, t.bytes.size());
    if (!t.device)
      CudaCheck(cudaMemcpyAsync(pointer, t.bytes.data(), t.bytes.size(), cudaMemcpyHostToDevice,
                                impl_->stream));
    if (!e.context->setTensorAddress(key.c_str(), pointer))
      throw std::runtime_error("Cannot bind model input");
  }
  std::vector<Tensor> outputs;
  for (const auto& key : spec.at("outputs")) {
    const auto label = key.get<std::string>();
    const auto dims = e.context->getTensorShape(label.c_str());
    Tensor t;
    for (int i = 0; i < dims.nbDims; ++i)
      t.shape.push_back(dims.d[i]);
    const auto type = e.engine->getTensorDataType(label.c_str());
    if (type == nvinfer1::DataType::kFLOAT)
      t.type = 0;
    else if (type == nvinfer1::DataType::kINT64)
      t.type = 1;
    else if (type == nvinfer1::DataType::kINT32)
      t.type = 2;
    else
      throw std::runtime_error("Expected float32/integer model output: " + label);
    const size_t bytes = t.Count() * (t.type == 1 ? 8 : 4);
    if (!e.context->setTensorAddress(label.c_str(), e.Buffer(label, bytes)))
      throw std::runtime_error("Cannot bind model output");
    if (device_output)
      t.device = e.buffers.at(label);
    else
      t.bytes.resize(bytes);
    outputs.push_back(std::move(t));
  }
  auto measurement = std::make_unique<Measurement>(name);
  CudaCheck(cudaEventRecord(measurement->begin, impl_->stream));
  if (!e.context->enqueueV3(impl_->stream))
    throw std::runtime_error("TensorRT inference failed: " + name);
  CudaCheck(cudaEventRecord(measurement->end, impl_->stream));
  impl_->pending.push_back(std::move(measurement));
  for (size_t i = 0; !device_output && i < outputs.size(); ++i) {
    const auto label = spec.at("outputs")[i].get<std::string>();
    auto& t = outputs[i];
    CudaCheck(cudaMemcpyAsync(t.bytes.data(), e.buffers.at(label)->data, t.bytes.size(),
                              cudaMemcpyDeviceToHost, impl_->stream));
  }
  if (!device_output)
    CudaCheck(cudaStreamSynchronize(impl_->stream));
  auto& timing = impl_->timings[name];
  ++timing.calls;
  timing.host_seconds += Seconds(started);
  return outputs;
}
Json GpuInfo() {
  int count = 0;
  CudaCheck(cudaGetDeviceCount(&count));
  if (count < 1)
    throw std::runtime_error("No CUDA device");
  cudaDeviceProp p;
  CudaCheck(cudaGetDeviceProperties(&p, 0));
  size_t free, total;
  CudaCheck(cudaMemGetInfo(&free, &total));
  return {{"name", p.name},
          {"compute_capability", {p.major, p.minor}},
          {"memory_total", total},
          {"memory_free", free}};
}
}  // namespace geosun

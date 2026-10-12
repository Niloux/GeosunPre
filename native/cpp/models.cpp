#include "models.h"

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <fstream>
#include <iostream>

struct CudaModels::Implementation {
  fs::path root;
  Json manifest;
  std::map<std::string, torch::jit::Module> modules;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  std::vector<at::Tensor> graph_inputs, graph_outputs;
  std::vector<at::Tensor> Forward(torch::jit::Module& module,
                                  const std::vector<at::Tensor>& inputs) {
    std::vector<torch::jit::IValue> arguments(inputs.begin(), inputs.end());
    const auto value = module.forward(arguments);
    std::vector<at::Tensor> output;
    if (value.isTensor())
      output.push_back(value.toTensor());
    else
      for (const auto& item : value.toTuple()->elements())
        output.push_back(item.toTensor());
    return output;
  }
};
CudaModels::CudaModels(const fs::path& root)
    : impl_(std::make_unique<Implementation>()) {
  impl_->root = root;
  impl_->manifest = ReadJson(root / "manifest.json");
  if (impl_->manifest.at("version") != 1 ||
      impl_->manifest.at("attention") != "efficient")
    throw std::runtime_error("Unsupported native model manifest");
  at::globalContext().setSDPUseFlash(false);
  at::globalContext().setSDPUseMemEfficient(true);
  at::globalContext().setSDPUseMath(false);
  at::globalContext().setSDPUseCuDNN(false);
  at::set_num_threads(4);
}
CudaModels::~CudaModels() = default;
const Json& CudaModels::Manifest() const { return impl_->manifest; }
std::vector<at::Tensor> CudaModels::Predict(
    const std::string& kind, const std::vector<at::Tensor>& inputs) {
  c10::InferenceMode guard;
  // The exported graph is already frozen. Runtime JIT fusion changes the
  // detector's floating-point order after warm-up and therefore feature IDs.
  torch::jit::GraphOptimizerEnabledGuard stable_graph(false);
  if (!torch::cuda::is_available())
    throw std::runtime_error("CUDA inference is unavailable");
  if (!impl_->manifest.at("models").contains(kind))
    throw std::runtime_error("Unknown model: " + kind);
  if (!impl_->modules.count(kind)) {
    std::ifstream stream(impl_->root / fs::u8path(kind + ".pt"),
                         std::ios::binary);
    if (!stream) throw std::runtime_error("Missing native model: " + kind);
    impl_->modules.emplace(kind, torch::jit::load(stream, torch::kCUDA));
    impl_->modules.at(kind).eval();
    std::cerr << "Native CUDA: loaded " << kind << '\n';
  }
  auto& module = impl_->modules.at(kind);
  if (kind != "matcher") return impl_->Forward(module, inputs);
  if (!impl_->graph) {
    for (const auto& input : inputs)
      impl_->graph_inputs.push_back(input.clone());
    const auto stream = c10::cuda::getStreamFromPool(false);
    torch::cuda::synchronize();
    {
      c10::cuda::CUDAStreamGuard stream_guard(stream);
      for (int i = 0; i < 3; ++i) impl_->Forward(module, impl_->graph_inputs);
      impl_->graph = std::make_unique<at::cuda::CUDAGraph>();
      impl_->graph->capture_begin();
      impl_->graph_outputs = impl_->Forward(module, impl_->graph_inputs);
      impl_->graph->capture_end();
    }
    stream.synchronize();
  }
  for (size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].sizes() != impl_->graph_inputs[i].sizes())
      return impl_->Forward(module, inputs);
  for (size_t i = 0; i < inputs.size(); ++i)
    impl_->graph_inputs[i].copy_(inputs[i]);
  impl_->graph->replay();
  return impl_->graph_outputs;
}

#pragma once
#include <ATen/core/Tensor.h>

#include <memory>

#include "runtime.h"
class CudaModels {
 public:
  explicit CudaModels(const fs::path& root);
  ~CudaModels();
  const Json& Manifest() const;
  std::vector<at::Tensor> Predict(const std::string& kind,
                                  const std::vector<at::Tensor>& inputs);

 private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

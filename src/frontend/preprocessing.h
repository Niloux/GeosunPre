#pragma once
#include "frontend/inference.h"

namespace geosun {
// The caller owns the CUDA stream and keeps it alive until these helpers finish.
class ImagePreprocessor {
  struct Implementation;
  std::unique_ptr<Implementation> impl_;

 public:
  explicit ImagePreprocessor(cudaStream_t stream);
  ~ImagePreprocessor();
  void Upload(const cv::Mat& rgb);
  Tensor Normalize(int width, int height, const std::array<float, 3>& mean = {0, 0, 0},
                   const std::array<float, 3>& deviation = {1, 1, 1});
};
struct MaskView {
  cv::Mat source, x, y, weight;
};
class MaskFusion {
  struct Implementation;
  std::unique_ptr<Implementation> impl_;

 public:
  MaskFusion(const std::vector<MaskView>& views, const cv::Mat& weight, const cv::Mat& valid,
             cudaStream_t stream);
  ~MaskFusion();
  void Begin();
  void Add(const Tensor& probabilities, size_t view);
  cv::Mat Finish();
};
}  // namespace geosun

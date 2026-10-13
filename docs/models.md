# GPU 模型契约

固定 TensorRT 10.13.3.9、GPU 0；按阶段加载释放模型，主程序直接调用 C++ API。运行库为 `models/runtime/nvinfer_10.dll`，运行包不包含 ONNX parser、builder resource、Python 或原始权重。

manifest.json：version=1、tensorrt=`10.13.3.9`、gpu、detector_size、descriptor_size、max_features=3072、descriptor_dimension=256、segmentation_mean、segmentation_std、models。各模型记录 inputs/outputs 名称数组、参考 shapes、engine SHA256。

| 引擎 | 输入 | 输出 |
| --- | --- | --- |
| detector | image float32 `[1,3,H,W]` RGB/255 | logits float32 `[1,1,H,W]` |
| descriptor | image float32 `[1,3,784,784]`；keypoints `[1,N,2]` | descriptors float32 `[1,N,256]` |
| matcher | ka/kb float32 `[1,3072,2]`；da/db `[1,3072,256]`；mask_a/mask_b float32 `[1,3072]` | matches int32 `[1,3072]`；confidence float32 `[1,3072]` |
| segmentation | image float32 `[1,3,512,512]`，RGB/255 后按 mean/std 归一化 | probabilities float32 `[1,2,512,512]`，天空/人物 |

检测最长边 1536，各边向下取 8 的倍数；3000×4000 输入对应 W=1152、H=1536。检测和描述使用双三次预处理，避免重复模型内归一化。CPU 参考与 GPU 实现共享 22 位整数系数，横向和纵向各自舍入并截断为 uint8，再按原规则归一化。

检测输出稠密 logits，原生 CUDA 执行 3×3 NMS，选最多 12288 个候选，用温度 0.5 的局部 softmax 修正子像素。有效域和语义筛选后，用 8×8 网格配额和置信度选最多 3072 个。归一化关键点 `(2x/W-1,2y/H-1)`。

描述采样为双线性 grid_sample、align_corners=false，描述数量动态。匹配保留互为最大值且置信度>0.1 的对应，未匹配为 -1。实际数量 32–3072、Na/Nb 可不同；张量补零到 3072，有效前缀掩码为 1，其余为 0。自注意力、交叉注意力及双 softmax 均屏蔽补齐项，补齐项不改变有效特征的数学归一化。仅保存实际前缀结果。名称及类型须与表格一致。

固定权重来源：

- [LoMa](https://github.com/davnords/LoMa/tree/8fb59c458f2a2ef44314f462467b229c27786bf0)：LoMaB、DeDoDe-G。
- [DaD](https://github.com/Parskatt/dad/releases/tag/v0.1.0) 检测器。
- [DINOv2](https://github.com/facebookresearch/dinov2)：ViT-L/14。
- [SegFormer B2 ADE20K](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)：revision `de01bae28967510f9ddd496c60a969357195400c`。

导出和 ONNX→TensorRT 编译是独立资源准备。引擎针对构建 GPU，更换 GPU 时重新构建，不默认承诺跨型号兼容。转换后需用真实图像检查有限输出、动态形状、描述子与匹配行为。

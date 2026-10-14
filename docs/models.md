# GPU 模型契约

固定 TensorRT 10.13.3.9、GPU 0；按阶段加载释放模型，主程序直接调用 C++ API。运行库为 `models/runtime/nvinfer_10.dll`，运行包不包含 ONNX parser、builder resource、Python 或原始权重。

## 默认准备流程

在项目根目录执行：

```powershell
.\prepare-models.ps1
```

默认向 `models/` 准备四个引擎，检测器支持动态宽高，不需要指定原图分辨率。匹配器主干为 FP16，Softmax、Reduce 和 Normalization 层，以及最终投影、相似度和匹配评分保留 FP32，所有浮点输入输出保持 float32。`run.ps1` 和 `package.ps1` 默认使用这个目录。

首次准备下载固定权重、导出 ONNX 并构建 TensorRT 引擎；完整且兼容的模型包在真实推理校验后复用。同 GPU 上更换照片分辨率、横竖方向或宽高比不需要重建模型。更换 GPU、模型协议或精度时需指定新的输出目录。

## 模型接口

manifest.json：version=2、tensorrt=`10.13.3.9`、gpu、matcher_precision、matcher_score_precision、detector_profile、descriptor_size、max_features=3072、descriptor_dimension=256、segmentation_mean、segmentation_std、models。各模型记录 inputs/outputs 名称数组、参考 shapes、engine SHA256；检测器 shapes 为 `[1,3,-1,-1]`。matcher_score_precision 记录最终评分精度，默认 FP16 主干搭配 FP32 评分；旧清单缺少该字段时按 matcher_precision 解析。

detector_profile 按 `[宽,高]` 记录 `min=[16,16]`、`opt=[1152,1536]`、`max=[1536,1536]`。opt 只用于 TensorRT 编译优化，不限制原图尺寸。运行端仍兼容 version=1 的固定尺寸包，但该包继续受 detector_size 限制。

| 引擎 | 输入 | 输出 |
| --- | --- | --- |
| detector | image float32 `[1,3,H,W]` RGB/255 | logits float32 `[1,1,H,W]` |
| descriptor | image float32 `[1,3,784,784]`；keypoints `[1,N,2]` | descriptors float32 `[1,N,256]` |
| matcher | ka/kb float32 `[1,3072,2]`；da/db `[1,3072,256]`；mask_a/mask_b float32 `[1,3072]` | matches int32 `[1,3072]`；confidence float32 `[1,3072]` |
| segmentation | image float32 `[1,3,512,512]`，RGB/255 后按 mean/std 归一化 | probabilities float32 `[1,2,512,512]`，天空/人物 |

检测最长边缩放到 1536，各边向下取 8 的倍数；3000×4000 输入对应 W=1152、H=1536，4000×3000 对应 W=1536、H=1152，1920×1080 对应 W=1536、H=864，均使用同一引擎。范围约束作用于缩放后的检测输入，原图无需限制为这些尺寸；短边缩放后小于 16 时明确报错。原图与 `.pro` 标定尺寸仍须一致。

检测和描述使用双三次预处理，避免重复模型内归一化。CPU 参考与 GPU 实现共享 22 位整数系数，横向和纵向各自舍入并截断为 uint8，再按原规则归一化。

检测输出稠密 logits，原生 CUDA 执行 3×3 NMS，选最多 12288 个候选，用温度 0.5 的局部 softmax 修正子像素。有效域和语义筛选后，用 8×8 网格配额和置信度选最多 3072 个。归一化关键点 `(2x/W-1,2y/H-1)`。

描述采样为双线性 grid_sample、align_corners=false，描述数量动态。匹配保留互为最大值且置信度>0.1 的对应，未匹配为 -1。实际数量 32–3072、Na/Nb 可不同；张量补零到 3072，有效前缀掩码为 1，其余为 0。自注意力、交叉注意力及双 softmax 均屏蔽补齐项，补齐项不改变有效特征的数学归一化。仅保存实际前缀结果。名称及类型须与表格一致。

固定权重来源：

- [LoMa](https://github.com/davnords/LoMa/tree/8fb59c458f2a2ef44314f462467b229c27786bf0)：LoMaB、DeDoDe-G。
- [DaD](https://github.com/Parskatt/dad/releases/tag/v0.1.0) 检测器。
- [DINOv2](https://github.com/facebookresearch/dinov2)：ViT-L/14。
- [SegFormer B2 ADE20K](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)：revision `de01bae28967510f9ddd496c60a969357195400c`。

导出和 ONNX→TensorRT 编译是独立资源准备，由根目录 `prepare-models.ps1` 完成。引擎针对构建 GPU，更换 GPU 时重新构建，不默认承诺跨型号兼容。

## 自定义参数

按目标设备或精度调整准备参数；升级旧固定尺寸模型包时使用新目录：

```powershell
.\prepare-models.ps1 -Models models/other-gpu -OnnxFrom models
.\prepare-models.ps1 -Models models/dynamic -OnnxFrom models
.\prepare-models.ps1 -Images "D:/照片/A.JPG","D:/照片/B.JPG"
```

- `-Models`：输出目录，默认 `models`。完整且兼容的包检查清单、引擎 SHA256、推理 DLL 与真实推理后复用，不重建；版本、GPU、动态范围、匹配精度或校验值不符时报错，应指定新目录。
- `-OnnxFrom`：可传已有模型目录或 `onnx/` 目录。四个单文件 ONNX 的名称、float32 输入与形状必须满足当前契约。来源为有清单的 version=1 固定尺寸包时，重新导出动态检测器，复用其他三个 ONNX；只提供旧静态 ONNX 而缺少清单时拒绝导入。
- `-MatcherPrecision`：`fp16`（默认）、`fp32` 或 `tf32`，用于显式调整匹配引擎的计算精度。TF32 允许 Tensor Core 矩阵乘法。模型清单保存该选择；未声明 `matcher_precision` 的清单按 FP32 解析。修改精度时使用新目录，并用真实场景检查最终几何质量。
- 当 `-OnnxFrom` 指向同 GPU、TensorRT 版本的完整模型包时，复用未改变接口和精度且清单及校验通过的引擎；升级固定尺寸包只需重建 detector，切换匹配精度只需重建 matcher。
- 旧 FP16 模型未保留最终评分的 FP32 精度时，使用新目录和 `-OnnxFrom` 升级；复用 ONNX 和其他引擎，只重建 matcher。置信度仍按原有相对误差 `<0.005` 验收，不放宽门限。
- `-Images`：两张真实验证照片；默认使用固定 LoMa 源码包中的 `0015_A.jpg` / `0015_B.jpg`。验证包括检测器动态宽高及边界、四模型有限输出及形状、描述子数量 32 / 127 / 3072、匹配有效数量不等、补零值变化不影响有效结果、同图匹配和语义概率范围。新导出的检测图在横图、竖图、方图及短边边界上比较 PyTorch 与 TensorRT 输出。

使用自定义目录时，运行指定 `--models`，打包指定 `-Models`。复用 ONNX 可以省去权重导出；更换 GPU 时仍需在目标 GPU 上重新构建并校验引擎。

`-ImageWidth` / `-ImageHeight` 已移除。旧固定尺寸引擎升级时需要一次重新导出和构建检测器，以后更换场景尺寸无需重编译。原始模型目录与已有工作目录保持不变，使用新模型时另建 work。

## 缓存与校验

脚本自动准备项目内 uv、Python 3.11.13、PyTorch 2.8.0（CUDA 12.8）、ONNX 与 TensorRT，均缓存于 `.cache/model-prepare/`，不修改系统 Python。首次需要联网；缓存完整后可离线准备与校验。已有 `models/source/` 的固定源码与权重按 SHA256 校验后直接复用。

新模型在同级 `目录.pending/` 中准备，通过验证且 Python 进程退出后，由 PowerShell 改名为目标目录，避免进程中的 Windows 文件占用阻止发布。中断后相同请求复用已完成的 ONNX 和引擎；已有完整清单的 `.pending` 包重新校验后直接发布。参数或导出脚本变化时拒绝混用未完成目录。保留原始权重、已有模型和运行成果。新目录包含清单、`engines/`、`runtime/`、`onnx/` 与 `preparation.json`；打包仅携带运行资源。

DaD 的 ImageNet 归一化留在检测图内；描述图不增加归一化，并固定 784×784 对应的 DINO 位置编码。匹配图在自注意力、交叉注意力和双 softmax 中屏蔽补齐项。检测、描述与分割引擎允许 FP16 内部计算，匹配引擎精度由准备参数确定；所有浮点输入输出保持 float32。原生 `check-models --models MODELS --image FILE` 依次运行四模型，检查形状、有限值、同图匹配与语义概率。运行时明确拒绝与当前 GPU 型号不匹配的模型包。

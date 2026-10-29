# GPU 模型契约

固定 TensorRT 10.13.3.9、GPU 0；按阶段加载释放模型，主程序直接调用 C++ API。运行库为 `models/runtime/nvinfer_10.dll`，运行包不包含 ONNX parser、builder resource、Python 或原始权重。

## 默认准备流程

在项目根目录执行：

```powershell
.\prepare-models.ps1
```

默认向 `models/` 准备六个引擎（version=4），检测器支持动态宽高，不需要指定原图分辨率。匹配器主干为 FP16，Softmax、Reduce 和 Normalization 层，以及最终投影、相似度和匹配评分保留 FP32，所有浮点输入输出保持 float32；geometry（MoGe-2 ViT-L normal）为 FP16，输入为 644×644 方形透视图、固定 2116 个 ViT token，只含原始前向输出，偏移恢复与尺度对齐由原生监督导出完成。grounding（SAM3）为掩码阶段检测人物、影子和人形展品，见下文。SAM3 检查点 [facebook/sam3](https://huggingface.co/facebook/sam3) 在 Hugging Face 需登录申请，无法自动下载：首次准备传入 `-Sam3 sam3.pt`，按 SHA256 校验后复制到 `.cache/model-prepare/source/sam3.pt`，以后不再需要该参数。`-NoGrounding` 准备不含 grounding 的 version=3 包。`run.ps1` 默认使用这个目录；交付打包默认使用 `models-ampere-plus/`，见 README。

首次准备下载固定权重、导出 ONNX 并构建 TensorRT 引擎；完整且兼容的模型包在真实推理校验后复用。同 GPU 上更换照片分辨率、横竖方向或宽高比不需要重建模型。更换 GPU、模型协议或精度时需指定新的输出目录。

## 模型接口

manifest.json：version=3、tensorrt=`10.13.3.9`、gpu、hardware_compatibility、matcher_precision、matcher_score_precision、detector_profile、descriptor_size、max_features=3072、descriptor_dimension=256、segmentation_mean、segmentation_std、geometry_size=644、geometry_tokens=2116、geometry_source（MoGe 提交、模型名与 revision）、models；version=4 另含 grounding_size=1008、grounding_mask_size=288、grounding_prompts=`["person","shadow","robot"]`、grounding_top_k=20、grounding_source（模型名、检查点 SHA256、导出所用 ultralytics 版本）。各模型记录 inputs/outputs 名称数组、参考 shapes、engine SHA256；检测器 shapes 为 `[1,3,-1,-1]`。matcher_score_precision 记录最终评分精度，默认 FP16 主干搭配 FP32 评分；旧清单缺少该字段时按 matcher_precision 解析。hardware_compatibility 为 `none` 或 `ampere_plus`，缺少时按 `none` 解析；gpu 始终记录构建 GPU。

detector_profile 按 `[宽,高]` 记录 `min=[16,16]`、`opt=[1152,1536]`、`max=[1536,1536]`。opt 只用于 TensorRT 编译优化，不限制原图尺寸。运行端仍兼容 version=1 的固定尺寸包，但该包继续受 detector_size 限制。version=1/2 的包没有 geometry 引擎，可继续用于重建和 LAS 监督；version=3 及以下没有 grounding 引擎，掩码中的人物仍由 segmentation 决定。

| 引擎 | 输入 | 输出 |
| --- | --- | --- |
| detector | image float32 `[1,3,H,W]` RGB/255 | logits float32 `[1,1,H,W]` |
| descriptor | image float32 `[1,3,784,784]`；keypoints `[1,N,2]` | descriptors float32 `[1,N,256]` |
| matcher | ka/kb float32 `[1,3072,2]`；da/db `[1,3072,256]`；mask_a/mask_b float32 `[1,3072]` | matches int32 `[1,3072]`；confidence float32 `[1,3072]` |
| segmentation | image float32 `[1,3,512,512]`，RGB/255 后按 mean/std 归一化 | probabilities float32 `[1,2,512,512]`，天空/人物 |
| geometry | image float32 `[1,3,644,644]` RGB/255 | points float32 `[1,644,644,3]`；normal `[1,644,644,3]`；mask `[1,644,644]`；metric_scale `[1]` |
| grounding | image float32 `[1,3,1008,1008]`，整幅照片拉伸缩放，RGB/127.5−1 | scores float32 `[3,20]`；boxes `[3,20,4]`；masks `[3,20,288,288]`；semantic `[3,1,288,288]` |

grounding 为 SAM3 概念检测，三个提示 person、shadow、robot 的文本特征在导出时算好并固化进图，不含文本编码器；改提示须重新导出并使用新模型目录。每个提示输出按联合分数（查询分数乘存在分数）降序的前 20 个查询：分数、归一化 cxcywh 框、实例掩码 logits 及该提示的语义 logits。ViT 的复数旋转位置编码导出为等价的实数 cos/sin 形式，导出前在随机输入上与原实现比对。TensorRT 构建工作区为 12 GiB；FP16 下注意力打分会超出半精度范围（PyTorch FP16 的注意力在 FP32 中累加，因此不受影响），故每个 Softmax 及其输入链直到打分矩阵乘保留 FP32，其余可用 FP16。

检测最长边缩放到 1536，各边向下取 8 的倍数；3000×4000 输入对应 W=1152、H=1536，4000×3000 对应 W=1536、H=1152，1920×1080 对应 W=1536、H=864，均使用同一引擎。范围约束作用于缩放后的检测输入，原图无需限制为这些尺寸；短边缩放后小于 16 时明确报错。原图与 `.pro` 标定尺寸仍须一致。

检测和描述使用双三次预处理，避免重复模型内归一化。GPU 实现使用 22 位整数系数，横向和纵向各自舍入并截断为 uint8，再按原规则归一化。

检测输出稠密 logits，原生 CUDA 执行 3×3 NMS，选最多 12288 个候选，用温度 0.5 的局部 softmax 修正子像素。有效域和语义筛选后，用 8×8 网格配额和置信度选最多 3072 个。归一化关键点 `(2x/W-1,2y/H-1)`。

描述采样为双线性 grid_sample、align_corners=false，描述数量动态。匹配保留互为最大值且置信度>0.1 的对应，未匹配为 -1。实际数量 32–3072、Na/Nb 可不同；张量补零到 3072，有效前缀掩码为 1，其余为 0。自注意力、交叉注意力及双 softmax 均屏蔽补齐项，补齐项不改变有效特征的数学归一化。仅保存实际前缀结果。名称及类型须与表格一致。

固定权重来源：

- [LoMa](https://github.com/davnords/LoMa/tree/8fb59c458f2a2ef44314f462467b229c27786bf0)：LoMaB、DeDoDe-G。
- [DaD](https://github.com/Parskatt/dad/releases/tag/v0.1.0) 检测器。
- [DINOv2](https://github.com/facebookresearch/dinov2)：ViT-L/14。
- [SegFormer B2 ADE20K](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)：revision `de01bae28967510f9ddd496c60a969357195400c`。
- [MoGe](https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a) 源码与 [MoGe-2 ViT-L normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal)：revision `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`，均为 MIT 许可。
- [SAM3](https://huggingface.co/facebook/sam3)：`sam3.pt`，SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`，SAM License；模型代码取自 ultralytics 8.4.192 及其 CLIP 分词器（固定提交），两者不带依赖安装，只用于导出。

导出和 ONNX→TensorRT 编译是独立资源准备，由根目录 `prepare-models.ps1` 完成。引擎针对构建 GPU，更换 GPU 时重新构建，不默认承诺跨型号兼容。

## ONNX 发布包

权重→ONNX 与 GPU 无关，只需在开发机做一次；ONNX→引擎随 GPU 变化。将前者冻结为发布包，其他机器构建引擎时不再下载权重、LoMa 源码或 SegFormer：

```powershell
.\prepare-models.ps1 -OnnxBundle packages/onnx-bundle-v1
.\prepare-models.ps1 -Models models/other-gpu -OnnxFrom packages/onnx-bundle-v1
```

`-OnnxBundle` 总是从固定权重重新导出，不接受 `-Models`、`-OnnxFrom`、`-Sam3`、`-MatcherPrecision`；发布包只含五个基础模型，以它为 `-OnnxFrom` 构建时 grounding 由目标机器缓存的 SAM3 检查点导出（或传 `-Sam3`）；输出目录已存在时报错。包内含 `onnx/` 五个单文件 ONNX、`references/` 导出时的 PyTorch 输入输出、`images/` 两张验证照片及 `onnx-bundle.json`（version=2、来源版本、检测器范围、max_features、geometry 输入尺寸与 token 数及全部文件 SHA256）。version=1 的四模型发布包仍可作为 `-OnnxFrom`，geometry 由固定 MoGe 权重导出，原包保持只读。导出同样经 `.pending` 目录校验后发布。

`-OnnxFrom` 指向发布包时先校验清单与全部 SHA256，默认用包内照片验证，并用包内参考数据完成 PyTorch/TensorRT 逐模型比较，门限与首次导出相同。构建仍需项目内 Python、PyTorch 与 TensorRT 环境，但不需要原始权重。

## 自定义参数

按目标设备或精度调整准备参数；升级旧固定尺寸模型包时使用新目录：

```powershell
.\prepare-models.ps1 -Models models/other-gpu -OnnxFrom models
.\prepare-models.ps1 -Models models/dynamic -OnnxFrom models
.\prepare-models.ps1 -Images "D:/照片/A.JPG","D:/照片/B.JPG"
```

- `-Models`：输出目录，默认 `models`。完整且兼容的包检查清单、引擎 SHA256、推理 DLL 与真实推理后复用，不重建；版本、GPU、动态范围、匹配精度或校验值不符时报错，应指定新目录。
- `-OnnxFrom`：可传已有模型目录或 `onnx/` 目录。四个单文件 ONNX 的名称、float32 输入与形状必须满足当前契约。来源为有清单的 version=1 固定尺寸包时，重新导出动态检测器，复用其他三个 ONNX；只提供旧静态 ONNX 而缺少清单时拒绝导入。
- `-HardwareCompatibility`：`none`（默认）或 `ampere_plus`。`none` 引擎只用于构建 GPU 同型号；`ampere_plus` 使用 TensorRT AMPERE_PLUS 兼容级别，一套引擎可在任意算力 ≥ 8.0 的 GPU（RTX 30/40/50、A 系列等）上运行，代价是部分优化策略不可用。构建 GPU 本身须算力 ≥ 8.0。修改级别时使用新目录，并在真实场景比较耗时与几何质量。
- `-MatcherPrecision`：`fp16`（默认）、`fp32` 或 `tf32`，用于显式调整匹配引擎的计算精度。TF32 允许 Tensor Core 矩阵乘法。模型清单保存该选择；未声明 `matcher_precision` 的清单按 FP32 解析。修改精度时使用新目录，并用真实场景检查最终几何质量。
- version=2 的四模型包升级到 version=3 时使用新目录，例如 `-Models models-v3 -OnnxFrom models`：复用四个 ONNX 与引擎，只导出并构建 geometry。
- `-Sam3`：SAM3 检查点路径，只在缓存中还没有时需要。`-NoGrounding`：不建 grounding，生成 version=3 包，与 `-Sam3` 冲突。由 version=3 包升级时使用新目录，例如 `-Models models-new -OnnxFrom models-v3`：复用五个 ONNX 与引擎，只导出并构建 grounding（ONNX 约 1.9 GB）。项目中原 version=3 包保留为 `models-v3/` 与 `models-ampere-plus-v3/`，记录它们的工作目录续跑时须 `--models` 指定原目录。grounding 在两张验证照片上比较 PyTorch FP32 与 TensorRT：全部查询分数差 < 0.05，PyTorch 分数 > 0.3 的每个检测与引擎同提示最接近掩码的 IoU ≥ 0.95；验证照片应含人物或影子，例如 `-Images` 指定两张展厅照片。真实推理另检查形状、分数范围与降序、框归一化。
- 当 `-OnnxFrom` 指向同 GPU、TensorRT 版本的完整模型包时，复用未改变接口和精度且清单及校验通过的引擎；升级固定尺寸包只需重建 detector，切换匹配精度只需重建 matcher。
- 旧 FP16 模型未保留最终评分的 FP32 精度时，使用新目录和 `-OnnxFrom` 升级；复用 ONNX 和其他引擎，只重建 matcher。置信度仍按原有相对误差 `<0.005` 验收，不放宽门限。
- `-Images`：两张真实验证照片；默认使用固定 LoMa 源码包中的 `0015_A.jpg` / `0015_B.jpg`。验证包括检测器动态宽高及边界、四模型有限输出及形状、描述子数量 32 / 127 / 3072、匹配有效数量不等、补零值变化不影响有效结果、同图匹配和语义概率范围。新导出的检测图在横图、竖图、方图及短边边界上比较 PyTorch 与 TensorRT 输出。geometry 在两张照片上比较 PyTorch FP32 与 TensorRT：PyTorch 掩码有效处对数深度差 p99 < 0.01、法向夹角 p99 < 1°、掩码差 p99 < 0.05、metric_scale 相对差 < 0.5%；真实推理还检查有效掩码过半、有效深度为正、法向为单位长度。

使用自定义目录时，运行指定 `--models`，打包指定 `-Models`。复用 ONNX 可以省去权重导出；更换 GPU 时仍需在目标 GPU 上重新构建并校验引擎。

`-ImageWidth` / `-ImageHeight` 已移除。旧固定尺寸引擎升级时需要一次重新导出和构建检测器，以后更换场景尺寸无需重编译。原始模型目录与已有工作目录保持不变，使用新模型时另建 work。

## 缓存与校验

脚本自动准备项目内 uv、Python 3.11.13、PyTorch 2.8.0（CUDA 12.8）、ONNX 与 TensorRT，均缓存于 `.cache/model-prepare/`，不修改系统 Python。首次需要联网；缓存完整后可离线准备与校验。已有 `models/source/` 的固定源码与权重按 SHA256 校验后直接复用。

新模型在同级 `目录.pending/` 中准备，通过验证且 Python 进程退出后，由 PowerShell 改名为目标目录，避免进程中的 Windows 文件占用阻止发布。中断后相同请求复用已完成的 ONNX 和引擎；已有完整清单的 `.pending` 包重新校验后直接发布。参数或导出脚本变化时拒绝混用未完成目录。保留原始权重、已有模型和运行成果。新目录包含清单、`engines/`、`runtime/`、`onnx/` 与 `preparation.json`；打包仅携带运行资源。

DaD 的 ImageNet 归一化留在检测图内；描述图不增加归一化，并固定 784×784 对应的 DINO 位置编码。匹配图在自注意力、交叉注意力和双 softmax 中屏蔽补齐项。检测、描述与分割引擎允许 FP16 内部计算，匹配引擎精度由准备参数确定；所有浮点输入输出保持 float32。原生 `check-models --models MODELS --image FILE` 依次运行四模型，检查形状、有限值、同图匹配与语义概率；version=3 包再运行 geometry，检查形状、有限值、掩码范围、单位法向、有效深度为正且有效掩码过半；version=4 包再运行 grounding，检查形状、有限值、分数范围与降序，并报告各提示分数 ≥ 0.5 的检测数。运行时明确拒绝与当前 GPU 型号不匹配的 `none` 模型包，以及算力低于 8.0 的 GPU 加载 `ampere_plus` 模型包。

# 资源与离线运行

## 依赖

```bash
uv sync --locked
```

LoMa、SegFormer 和原生 Ceres 扩展都是必需依赖，无 extras 和数值后端选择。安装需要 Python ≥ 3.13、C++17 编译器和 Python 开发头文件。版本由 `uv.lock` 固定；完整 GPU 阶段使用 CUDA 12.8 的 PyTorch 2.8、FlashAttention 和 CUDA Graph。

修改原生扩展后重新构建：

```bash
uv sync --locked --reinstall-package geosunpre-native
```

原生扩展契约及许可证见 [native](../native/README.md)。

## 本地权重

运行启用离线加载，不自行下载。提前准备以下文件：

```text
.cache/torch/hub/checkpoints/
  dad.pth
  dinov2_vitl14_pretrain.pth
  loma_B.pt

.cache/huggingface/
  models--nvidia--segformer-b2-finetuned-ade-512-512/
    snapshots/de01bae28967510f9ddd496c60a969357195400c/
      config.json
      preprocessor_config.json
      model.safetensors       # 或 pytorch_model.bin
```

Hugging Face snapshot 文件可以是指向 `blobs/` 的有效链接。`--model-cache` 指向 Hugging Face 缓存根目录，不是 `models--...` 子目录。

LoMa 使用 `TORCH_HOME`，默认当前目录的 `.cache/torch`；SegFormer 使用 `--model-cache`、`HF_HUB_CACHE` 或默认 `.cache/huggingface`。`run.sh` 从仓库根目录运行，因此默认缓存位于仓库中；直接 CLI 的相对默认缓存以启动目录为基准。续跑继承冻结的 SegFormer 缓存路径，并核对已使用模型的权重内容及路径。

`prepare` 不需要权重或 GPU，可先检查输入再准备资源。完整流程在启动昂贵阶段前检查待处理缓存、GPU、FlashAttention、原生扩展、导航/LAS 和交付磁盘空间。数值阶段已缓存时无需初始化 GPU；缺失资源会报出原因。

## 资源控制与读数

默认 GPU 特征缓存上限 2 GiB，且不超过 GPU 总显存的 1/8。用 `GEOSUNPRE_GPU_FEATURE_CACHE_MIB=512` 下调预算或设为 0 禁用缓存；这只改变缓存资源，不切换算法。CPU 特征有界 LRU 和各阶段后的内存回收继续保留。

冻结观测通过完整文件 SHA 验证缓存内容，特征数组按需求读取：局部标定读取选中片段；定稿、已完成模型检查和导出不物化全场 keypoints/scores；首次冻结和首次建全局数据库仍读取全部必要数组。元数据和内容身份检查不会因此跳过。这个改动减少无用数组读取，实际整条流程耗时及资源峰值仍以真实场景测量为准。

`workers` 控制 CPU 前端预处理、LAS 查询和联合求解。局部与全局视觉 BA 使用固定 4 线程原生深度约束。原生扩展缺失直接报错，没有 Python 数值回退。

`pipeline_status.json` 的计时覆盖记录阶段和编排，不含 CLI、配置及任务身份初始化；完整进程耗时应另行记录。它保存阶段墙钟、CPU 时间、RSS 采样及进程高水位、PyTorch CUDA 分配/保留峰值和阶段结束占用，以及嵌套操作耗时。父子操作耗时包含关系不能相加；RSS 采样可能漏掉瞬时峰值，PyTorch 指标不代表整卡显存。资源监控不会为了读指标主动初始化 CUDA。

2026-10-08 的[篮球场真实验收](benchmarks/basketball_architecture_20261008.md)完成了新工作目录的完整流程：整进程 21 分 59.25 秒，RSS 高水位 4.64 GiB，PyTorch CUDA 分配/保留峰值 2.45/4.77 GiB；20 分钟目标尚未达到。这是当前实现的一次测量，没有修改前对照，不能据此宣称提速或降低峰值。后续验收仍使用新工作目录和独立负载；保留完整进程计时、日志与原始阶段状态，再执行会更新最近状态的缓存导出。不要将缓存恢复计时当成新任务计时。

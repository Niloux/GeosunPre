# 资源与离线运行

## 环境与权重

支持 Linux 和 Windows 本机，使用 Python 3.13、C++17 编译器和 Python 开发头文件。依赖由 `uv.lock` 固定，GPU 阶段使用 CUDA 12.8 的 PyTorch 2.8 和 CUDA Graph。Linux 匹配使用 FlashAttention；Windows 版 PyTorch 2.8 未编译该内核，使用 `EFFICIENT_ATTENTION` CUDA 内核，实际后端写入运行记录和匹配缓存。

Windows 需要 Visual Studio 的“使用 C++ 的桌面开发”工作负载和 Windows SDK；不需要另行编译 PyTorch 或安装第三方 `flash-attn`。运行：

```powershell
.\setup-windows.ps1
.\setup-windows.ps1 -CheckOnly
```

脚本复用已安装的 `uv`，即使当前进程的 PATH 尚未刷新也会检查用户 PATH。Python 下载至项目 `.cache/python/`，依赖环境位于 `.venv/`；不修改系统 Python。环境检查仅导入依赖、确认 CUDA 可用并加载原生扩展，结果保存在 `.cache/environment_check.json`。权重和匹配内核由实际阶段的资源预检检查。

```bash
uv sync --locked

# 修改原生扩展后重建
uv sync --locked --reinstall-package geosunpre-native
```

LoMa、SegFormer 和原生 Ceres 扩展均必需，扩展缺失直接报错。契约与许可证见 [native](../native/README.md)。运行离线加载，提前准备：

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

snapshot 可用指向 `blobs/` 的有效链接。从 WSL 迁移到 Windows 时，将上述权重复制为实际文件；Linux 缓存的符号链接不能直接作为 Windows 文件读取。无需迁移 `.venv` 或 `uv` 缓存。

LoMa 使用 `TORCH_HOME`，默认 `.cache/torch`；SegFormer 使用 `--model-cache`、`HF_HUB_CACHE` 或 `.cache/huggingface`，参数指向缓存根目录。`run.sh` 和 `run.ps1` 在仓库根运行；直接 CLI 的相对路径以启动目录为基准。续跑继承缓存位置并核对已使用权重。

在 Windows 上使用新工作目录；WSL 的旧工作目录冻结了 Linux 绝对路径，不能直接跨系统续跑。默认 `copy` 导出可直接使用；`--image-mode link` 创建符号链接，需要 Windows 开发者模式或相应权限。

`prepare` 无需 GPU/权重；昂贵阶段前预检待处理缓存、CUDA、平台注意力内核、扩展、导航/LAS 和交付空间。缓存数值阶段不初始化 GPU。

## 资源与计时

GPU 特征缓存默认至多 2 GiB，且不超过总显存的 1/8。`GEOSUNPRE_GPU_FEATURE_CACHE_MIB=512` 可下调，0 禁用；CPU 保留有界 LRU 和阶段后回收。特征数组按需加载；自动选标定支撑及首次冻结/建库会访问必要的全场数组。

`workers` 控制 LAS 查询与联合求解；视觉建图和 BA 使用固定 4 线程。掩码预处理与后处理各使用两个 CPU worker。

`pipeline_status.json` 保存阶段墙钟、CPU 时间、RSS 采样/高水位、PyTorch CUDA 峰值及阶段结束占用。阶段计时不含 CLI/任务初始化；完整进程耗时另记。父子操作不能相加，RSS 采样可能漏峰值，PyTorch 读数不代表整卡显存；监控不主动初始化 CUDA。

性能比较使用独立负载、相同输入与交付方式，区分从头运行和缓存恢复、峰值和结束占用。[本轮验证](benchmarks/generalization_20261009.md)复用了前端缓存，不能与[2026-10-08 历史运行](benchmarks/basketball_architecture_20261008.md)作提速对照。

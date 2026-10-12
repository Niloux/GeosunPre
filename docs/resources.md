# 资源与离线运行

Windows 默认生产入口已切换为独立 C++/CUDA 程序，Python 用于开发环境、一次性权重转换和显式参考流程。部署目录由 `package-native.ps1` 生成，包含 EXE、必要 DLL 和转换后的模型，详见[原生运行](../native/README.md)。本页的 `uv`、Python 导入和原生扩展说明适用于开发及参考环境。

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

Windows 数值阶段还需要 `.cache/native/build/bin/geosunpre-native.exe`，通过 `setup-native.ps1` 构建，`setup-windows.ps1` 会调用它。独立程序链接 COLMAP 4.2.0、Ceres 2.2.0、SuiteSparse 与官方 OpenBLAS 0.3.33 Windows LP64 优化内核；Ceres 同时编译 CUDA 稠密求解支持。BLAS 默认一线程，视觉求解沿用四线程，避免嵌套线程争抢。

`GEOSUNPRE_NUMERIC_BACKEND=python` 可显式运行原有参考实现，Linux 默认使用该实现；`GEOSUNPRE_NATIVE_EXECUTABLE` 可指定原生程序位置。选择 C++ 时缺少程序或执行失败会直接报错。旧工作目录和已验收成果保留；实际比较在新工作目录中使用相同输入、缓存和交付选项。

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

原生阶段另外记录 `native_processes`、`native_cpu_seconds`、`native_rss_peak_bytes` 和 `combined_average_cpu_cores`，包含 C++ 子进程的 CPU 时间及独立进程 RSS 高水位；父进程 RSS 与子进程峰值不能相加当作同时占用。Ceres 的 CUDA 分配不包含在 PyTorch 显存读数中。每次原生求解的建题、检查、完整求解器报告和构建环境也保存在对应 `ba.json`。

性能比较使用独立负载、相同输入与交付方式，区分从头运行和缓存恢复、峰值和结束占用。[本轮验证](benchmarks/generalization_20261009.md)复用了前端缓存，不能与[2026-10-08 历史运行](benchmarks/basketball_architecture_20261008.md)作提速对照。

[Windows 原生数值验证](benchmarks/windows_native_20261012.md)记录了同一模型的收敛 BA 对照、整场重跑和仍待归因的系统差距。

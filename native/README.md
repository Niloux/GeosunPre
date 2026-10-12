# 原生 C++/CUDA 流程

## 独立 Windows C++ 求解程序

`cpp/` 提供输入准备、SegFormer/LoMa CUDA 前端、观测冻结、标定、SfM/BA、LAS 配准、条件联合优化及 COLMAP 导出验包。生产运行只启动 C++ 程序；LibTorch 调用已编译的 CUDA 内核。正深度和联合代价分别共用 `depth_cost.h` 与 `joint_costs.h`，沿用鱼眼、独立曝光、gauge 和验收约定。

```powershell
.\setup-native.ps1 -VcpkgRoot "D:/Dev/vcpkg"
.\.cache\native\build\bin\geosunpre-native.exe info
.\.cache\native\build\bin\geosunpre-native.exe ba --input "work/源模型" --initial-model "work/初始标定" --output "work/新结果" --stage fixed --protect-depth
.\.cache\native\build\bin\geosunpre-native.exe map --database "work/database.db" --seed "work/seed" --images "input/场景/Image" --output "work/新建图"
```

`ba` 输出 `ba.json` 和 `sparse/0/`；`map` 输出 `mapping.json` 和 `sparse/0/`。只读输入，数据库在新输出内复制，模型回读通过后发布，已有输出和残留 `.pending` 目录拒绝覆盖。Windows UTF-8 图像名和中文路径通过宽字符入口与 filesystem 处理。`--iterations`、`--workers`、`--blas-threads` 用于阶段执行与实际性能比较；`ba --gpu` 显式选择 Ceres CUDA/DENSE_SCHUR，是否优于稀疏 CPU 求解需要真实模型比较。

完整入口和部署：

```powershell
# 离线转换一次，使用已有参考任务的实图/特征
.\.venv\Scripts\python.exe scripts/export_native_models.py --work "work/参考场景" --output .cache/native/models
.\package-native.ps1
.\run.ps1 --input "input/场景" --work "work/原生场景"
.\.cache\native\runtime\geosunpre-native.exe check --work "work/原生场景"
```

`package-native.ps1` 生成 `.cache/native/runtime/`：主程序、`inference/` 下的 LibTorch/CUDA 程序与 DLL、`models/` 下的四个权重。该目录不包含 Python、WSL 或图像转换示例；运行需可用的 NVIDIA 驱动。构建时使用 `.venv` 中的 LibTorch SDK，也可通过 CMake 的 `GEOSUNPRE_TORCH_ROOT` 指定独立 SDK。模型转换属于开发准备，运行程序不会启动转换工具。

默认 `run.ps1` 启动原生流程。原有 Python 任务由 `run-reference.ps1` 显式运行，其数值阶段仍可通过 `native_runtime.py` 调用 C++。原生失败直接报错，不改变后端。实际整场验证状态见 [MIGRATION.md](MIGRATION.md)。

BA 会输出建题、求解结束及约 30 秒一次的迭代进度，进度在迭代返回时更新；`--progress` 可显示详细迭代。显式 CUDA 求解失败直接报错。真实场景的数值对照与验包结果见[验证记录](../docs/benchmarks/windows_native_20261012.md)。

## 保留的 Python 扩展

`geosunpre-native` 是主工程必需的本地扩展，由 `uv sync --locked` 构建安装。需要 C++17 编译器及 Python 开发头文件，pybind11 固定为 3.0.1。

扩展实现 `PositiveDepthCost` 的残差和解析 Jacobian，支持四线程受限视觉 BA。`Sim3PointPlaneCost` 与 `Sim3PositionPriorCost` 为联合优化提供 LAS 点到面及相机中心位置先验，Jacobian 在 C++ 中计算。固定兼容 pyceres 2.6 / Ceres 2.2.0，导入检查 Ceres 版本，当前实现版本为 2。修改相关版本须重新核对 ABI 并执行数值及实际求解检查。

全局变换采用 Eigen 顺序四元数 `[x,y,z,w]` 与 `[tx,ty,tz,log_scale]` 两个参数块。相机中心由 `cam_from_world` 的 `-R.T @ t` 计算，输入姿态四元数不参与求解。`joint_costs.h` 是增量编译依赖并随源码包发布。

```bash
# 主工程重建本地扩展
uv sync --locked --reinstall-package geosunpre-native

# 独立发行包
uv build native

```

没有 Python 深度约束回退。缺失联合约束时会要求重建扩展。`vendor/` 仅包含 Ceres CostFunction 接口及构造/析构实现，不包含另一份求解器；来源见 [vendor/README.md](vendor/README.md)，许可证见 [vendor/LICENSE.ceres](vendor/LICENSE.ceres)，随源码包和 wheel 分发。

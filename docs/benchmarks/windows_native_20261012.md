# Windows 原生数值阶段（2026-10-12）

本记录对应第一阶段：独立 C++ 增量建图、四步标定中的受限 BA、全局 BA、正深度约束和数值检查。当时 Python 负责数据准备、编排和其余模块。原始输入、Windows 和 WSL 的原结果均保留，数值阶段成果在 `work/native_sfm_20261012/colmap/`。后续完整 C++/CUDA 迁移和实际验收见[第二阶段记录](windows_cpp_pipeline_20261012.md)。

## 已定位的开销

历史 Windows / WSL 记录使用 i7-13700F、RTX 4090、相同依赖版本和配置协议 8，均为 536 幅图像，前端均从头运行。候选数为 10,429 / 10,425，两次运行日期、几何输入和系统负载并未完全控制。

| 历史记录 | Windows | WSL |
| --- | ---: | ---: |
| 完整流程 | 3481.31 s | 1238.89 s |
| SfM 阶段 | 2692.69 s | 843.17 s |
| 增量建图操作 | 1278.93 s | 549.93 s |
| 三次全局 BA 操作合计 | 1260.59 s | 213.06 s |

Windows 两份 Python 数值轮子的 OpenBLAS 均为 `GENERIC / SINGLE_THREADED / NO_LAPACK`。在隔离的轮子副本中替换优化 BLAS，对同一冻结模型运行三步 BA，空闲机器上的线性求解时间由 5.175 s 降至 3.207 s。原安装目录的 DLL 没有修改。

另一项开销来自 Python 逐观测构建正深度约束及检查残差。在同一实际模型的完整求解中，Python 建题约 13.64 s，原生建题 1.65 s；原生健康检查 0.047 s。原生约束采用原有解析 Jacobian，模型和求解约定保持一致。

两边原有 `pycolmap.has_cuda` 都为 false，原有 BA 都采用四线程 CPU / SPARSE_SCHUR。这不能单独解释 Windows 的性能退化。历史记录中额外的耗时放大尚未完全复现和归因。

## 同一模型的完整 BA 对照

只读复用旧 Windows 建图结果：536 幅图像、185,852 个点、942,403 条观测。固定内参，开启正深度约束、相同 gauge、Huber 2、四线程和收敛条件。

| 指标 | 原 Python/轮子 | 原生 C++ |
| --- | ---: | ---: |
| 墙钟时间 | 92.404 s | 47.687 s |
| Ceres 总求解时间 | 71.561 s | 44.388 s |
| Ceres 线性求解时间 | 52.905 s | 27.512 s |
| Ceres 迭代数 | 36 | 36 |
| 最终目标值 | 1290917.474805743 | 1290917.474805740 |
| 无效深度 | 0 | 0 |

独立原生程序约快 **1.94 倍**，目标值在浮点误差范围内一致。Python 计时覆盖 BA 函数；C++ 计时还包含模型读取、写入和回读。生产适配器额外的模型传输开销不包含在这项比例内。

证据位于 `work/native_diagnosis_20261012/` 的 `generic_global_converged_idle.json` 和 `cpp_global_converged/ba.json`。三步诊断仅用于分解开销，完整收敛对照用于验证数值结果。

## 完整场景与交付

新工作目录复用未变化的掩码、536 幅特征和 10,429 对匹配缓存，不复用标定、建图和 BA 模型。其余配置与交付参数沿用原任务；默认复制原图。

| 指标 | 新结果 |
| --- | ---: |
| 自动标定 | 64/64 幅注册，93.11 s |
| SfM 阶段 | 1292.08 s（21 分 32 秒） |
| 增量建图操作 | 858.89 s |
| 三次 BA 操作合计 | 298.62 s |
| 最终曝光 | 536/536 |
| 稀疏点 / 观测 | 71,609 / 460,251 |
| 无效深度 | 0 |
| 拟合重投影 median / P90 | 1.0659 / 2.3249 px |
| LAS 支撑比例 | 52.16% |
| LAS 点到面 median / P90 | 9.15 / 54.37 mm |
| 定稿 / LAS 配准 / 导出 | 8.14 / 48.99 / 66.45 s |
| 初始化 PLY | 7,587,128 点，5 cm |

共享内参 BA 收敛，两镜头的单调性、参数界限、观测覆盖和整体/分镜头误差验收通过。联合求解按原有噪声代理规则跳过，采用合格的 `global_registration`。图像、掩码、五份 COLMAP 二进制和初始化 PLY 的独立回读验包通过。

新 SfM 耗时约为历史 Windows 的 48%，仍高于历史 WSL 的 843.17 s。此项为整场重跑记录，标定结果和系统状态不同；缓存复用的本轮总耗时也不能当作从头运行的速度比例。

## 构建与使用

原生程序采用 MSVC Release、COLMAP 4.2.0、Ceres 2.2.0、SuiteSparse 7.14.0、官方 OpenBLAS 0.3.33 LP64 / DYNAMIC_ARCH。BLAS 默认一线程，Ceres 四线程，依赖与产物均位于 `.cache/native/`。

```powershell
.\setup-native.ps1 -VcpkgRoot "D:/Dev/vcpkg"
.\.cache\native\build\bin\geosunpre-native.exe info

# 用相同冻结输入做完整 BA 对照，输出必须是新路径
.\.venv\Scripts\python.exe -X utf8 scripts/profile_ba.py --model "work/源场景/sfm/mapping" --initial-model "work/源场景/calibration/stages/principal/sparse/0" --stage fixed --iterations 400 --output "work/对照/python.json"
.\.cache\native\build\bin\geosunpre-native.exe ba --input "work/源场景/sfm/mapping" --initial-model "work/源场景/calibration/stages/principal/sparse/0" --stage fixed --protect-depth --iterations 400 --output "work/对照/cpp"
```

原 Python 实现可显式选择 `GEOSUNPRE_NUMERIC_BACKEND=python`；原生失败直接报错。C++ 对已有输出或 `.pending` 拒绝覆盖，数值产物回读后发布。BA 输出建题、求解结束及约 30 秒一次的迭代进度；进度在迭代返回时更新。

Ceres CUDA/DENSE_SCHUR 在真实 64 幅模型上运行通过，完整报告记录了实际 CUDA 后端。该短运行用于验证 CUDA 分支，未作为收敛交付或大规模 BA 提速结论。536 幅生产运行沿用 CPU 稀疏 Schur。

## 第一阶段结束时的后续迁移范围

| 模块 | 当前状态 / 下一步 |
| --- | --- |
| 数值核心 | 独立 C++ 建图、视觉 BA、约束和检查已接入，Ceres CUDA 可显式使用 |
| SfM 准备与轨迹处理 | 几何验证、选图、数据库准备和轨迹筛选仍在 Python，继续沿冻结特征编号迁移 |
| LAS 与联合更新 | 采样、表面对应、配准、联合建题和 PLY 仍在 Python，先保持原坐标、采样与验收约定 |
| 模型前端 | SegFormer / LoMa 仍为 PyTorch；导出固定权重和预处理后接入 C++/CUDA 推理，核对特征与匹配语义 |
| 编排与交付 | Python 适配器暂时传递模型，最后迁移 CLI、缓存恢复、交付和验包 |

当前只在篮球场数据上完成实际阶段验证。几何验收为拟合数据检查，独立测量精度和 3DGS 渲染尚未验证。

# Geosun 预处理与 3DGS 训练整体包

Windows x64，需算力 ≥ 8.0 的 NVIDIA GPU（RTX 30/40/50、RTX A 系列等）、显存 ≥ 12 GB（推荐 16 GB 及以上），驱动支持 CUDA 12.8（建议 R570 及以上）。无需安装 Python、CUDA Toolkit、Visual Studio 或联网。

磁盘空间要求：`geosun.exe` 启动时会自动对 `--output` 所在卷执行可用空间预检（全流程至少需 18 GiB 或 `1.5 × 输入场景体积 + 10 GiB`；单步预处理至少 `max(8 GiB, 1.5 × 输入场景体积)`；单步训练至少 6 GiB；单步导出至少 4 GiB）。

跨显卡使用时选文件名带 `ampere_plus` 的包。带 `none` 的包要求显卡型号与模型构建时一致，兼容级别和构建显卡可在 `bundle.json` 中查看。

| 目录 | 内容 |
| --- | --- |
| `preprocess/` | GeosunPre 运行包：`geosunpre.exe`、模型和运行库，说明见其 `README.md` 和 `docs/` |
| `trainer/` | Geosun 3DGS Trainer：`bin/geosun_3dgs_train.exe` 和 `config/trainer.json` |
| `geosun.exe` | 整体入口：预处理 → 交付回读 → 训练 → 导出 |
| `bundle.json` | 两个子包的版本、提交、配对关系和 `package.json` 的 SHA256 |

两个子包各自带运行库，保持目录结构，不要把文件合并到同一目录。将整体包解压到纯 ASCII 路径（例如 `D:\Geosun`），否则训练器不能用 GPU 解码 JPEG。

## 使用

在包目录打开 PowerShell 或 CMD：

```powershell
.\preprocess\geosunpre.exe info
.\preprocess\geosunpre.exe check-models --image "D:/照片/样例.JPG"
.\geosun.exe --scene "D:/数据/场景" --output "D:/成果/场景"
```

`--output` 下生成 `delivery/`（训练数据）、`train/project.licht`、`export/`（含 `splats.ply`、`mesh.ply` 和 `mesh_osgb/`）以及 `logs/`（含总控日志 `geosun.log`、预处理日志 `preprocess.log`、训练日志 `train.log`、导出日志 `export.log` 与训练器运行期崩溃诊断目录 `runtime/logs/`，全程不写入系统用户目录 `%USERPROFILE%`；排障时可直接打包 `logs/` 目录）。导出由包内训练器完成。默认纹理像素尺寸为 0.02 场景单位，可用 `--texel-size` 调整。

- 预处理中途失败：以相同命令重跑即续跑，已完成的交付只做回读检查。
- 训练中途失败：移走 `train.pending` 后重跑，或用新的 `--output`。
- `--step preprocess` 只预处理，`--step train` 只对已有交付训练。
- `--step export` 从已有 `train/project.licht` 补导。失败时保留 `export.pending`，将它移走后可重新补导。
- `--preprocess-arg` 传给 `geosunpre.exe run`，例如 `--preprocess-arg --ply-voxel --preprocess-arg 0.1`；`--train-arg` 传给训练器，覆盖 `trainer.json` 中的设置，例如 `--train-arg --iter --train-arg 7000`。

输入、配置或程序改变时使用新的 `--output`，保留已有成果。更新任一程序时组装新的整体包，不覆盖旧包。

## 许可与商业分发说明

本整包采用 GPLv3 第 5 条定义的“聚合体（Aggregate）”架构：

- **独立入口与预处理程序（`geosun.exe`、`preprocess/`）**：独立专有程序，仅通过操作系统进程调用和磁盘标准数据目录（COLMAP/PLY/PNG/JSON）与训练器交互；所含第三方库及五模型权重（LoMa、DaD、DINOv2、MoGe-2、SAM3）许可证见 `preprocess/licenses/` 与 `preprocess/models/runtime/`。
- **独立训练与导出程序（`trainer/`）**：基于 LichtFeld Studio 构建的独立无头训练与导出可执行程序，遵循 **GNU General Public License v3.0 or later（GPL-3.0-or-later）** 及 **LichtFeld Studio NVIDIA GPU Library Exception v1.0**。完整许可文本及对应源码（Corresponding Source）获取说明见 `trainer/LICENSE`、`trainer/LICENSE-EXCEPTION.txt`、`trainer/THIRD_PARTY_LICENSES.md`、`trainer/share/LichtFeld-Studio/licenses/` 及 `trainer/README.md`。

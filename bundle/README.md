# Geosun 3DGS

从双鱼眼照片和 LAS 点云完成预处理、3DGS 训练及成果导出。

Windows x64，需算力 ≥ 8.0 的 NVIDIA GPU（RTX 30/40/50、RTX A 系列等）、显存 ≥ 12 GB（推荐 16 GB 及以上），驱动支持 CUDA 12.8（建议 R570 及以上）。无需安装 Python、CUDA Toolkit、Visual Studio 或联网。

磁盘空间要求：`Geosun.exe` 启动时会自动检查 `--output` 所在卷的可用空间。全流程需要 `max(18 GiB, 1.5 × 输入场景体积 + 10 GiB)`，单步预处理需要 `max(8 GiB, 1.5 × 输入场景体积)`，单步训练需要 6 GiB，单步导出需要 4 GiB。

## 使用

将 ZIP 解压到纯 ASCII 路径（例如 `D:\Geosun`），以便训练器使用 GPU 解码 JPEG。保持解压后的目录结构，在 `Geosun.exe` 所在目录打开 PowerShell 或 CMD，运行全流程。

```powershell
.\Geosun.exe --scene "D:/数据/场景" --output "D:/成果/场景"
```

`--output` 也可以设为场景目录本身，或场景下的新目录。磁盘空间估算会排除本次的成果和工作缓存。

新建预处理任务时，默认使用场景根目录中修改时间最新的 PCC `.pro`。可用 `--project 1110.pro` 手动改选。多份 `.pro` 并列最新时，程序会列出候选，需用 `--project` 指定。

```powershell
.\Geosun.exe --scene "D:/数据/场景" --output "D:/数据/场景"
```

此例把 `delivery/`、`train/`、`export/` 和 `logs/` 放在场景根目录。预处理根据选中的项目名读取同名的 CameraPos、导航表及默认 LAS。LAS 存于其他位置或使用了自定义文件名时，追加 `--preprocess-arg --las --preprocess-arg "D:/点云/办公室0918.las"`。续跑沿用已选项目。

`--output` 下生成 `delivery/`（训练数据）、`train/project.licht`、`export/`（含 `splats.ply`、`mesh.ply` 和 `mesh_osgb/`）以及 `logs/`（含总控日志 `geosun.log`、预处理日志 `preprocess.log`、训练日志 `train.log`、导出日志 `export.log` 与训练器运行期崩溃诊断目录 `runtime/logs/`，全程不写入系统用户目录 `%USERPROFILE%`；排障时可直接打包 `logs/` 目录）。导出由包内训练器完成。默认纹理像素尺寸为 0.02 场景单位，可用 `--texel-size` 调整。

已有 3DGS 成果时，可以用相同的 `--output` 重跑。全流程会复用通过回读校验的 `delivery/`，等新训练和导出都完成后一起替换 `train/` 和 `export/`。

- `--step preprocess` 只预处理，中断后可用同一命令续跑。已完成的交付会回读校验。
- `--step train` 用现有交付训练，成功后替换 `train/`。`--step export` 从已有 `train/project.licht` 导出，成功后替换 `export/`。
- 训练或导出中断会留下 `.pending` 目录。对应阶段重跑时，会先将它改名保留，再生成新成果。
- `--preprocess-arg` 传给 `geosunpre.exe run`，例如 `--preprocess-arg --ply-voxel --preprocess-arg 0.1`；`--train-arg` 传给训练器，覆盖 `trainer.json` 中的设置，例如 `--train-arg --iter --train-arg 7000`。

需要重新预处理或保留多轮 3DGS 成果时，使用新的 `--output`，例如 `D:/数据/场景/3DGS-1119`。更新任一程序时组装新的整体包。

## 查版本和自检

`bundle.json` 的 `version` 是整包版本。清单也记录两个子包各自的版本、Git 提交、硬件兼容信息和 `package.json` 的 SHA256。

`preprocess/` 包含 GeosunPre、模型和运行库，`trainer/` 包含训练器和训练配置。排障时可运行以下命令查看预处理程序信息，或用真实照片检查模型。

```powershell
.\preprocess\geosunpre.exe info
.\preprocess\geosunpre.exe check-models --image "D:/照片/样例.JPG"
```

## 许可与商业分发说明

本整包采用 GPLv3 第 5 条定义的“聚合体（Aggregate）”架构：

- **独立入口与预处理程序（`Geosun.exe`、`preprocess/`）**：独立专有程序，仅通过操作系统进程调用和磁盘标准数据目录（COLMAP/PLY/PNG/JSON）与训练器交互；所含第三方库及五模型权重（LoMa、DaD、DINOv2、MoGe-2、SAM3）许可证见 `preprocess/licenses/` 与 `preprocess/models/runtime/`。
- **独立训练与导出程序（`trainer/`）**：基于 LichtFeld Studio 构建的独立无头训练与导出可执行程序，遵循 **GNU General Public License v3.0 or later（GPL-3.0-or-later）** 及 **LichtFeld Studio NVIDIA GPU Library Exception v1.0**。完整许可文本及对应源码（Corresponding Source）获取说明见 `trainer/LICENSE`、`trainer/LICENSE-EXCEPTION.txt`、`trainer/THIRD_PARTY_LICENSES.md`、`trainer/share/LichtFeld-Studio/licenses/` 及 `trainer/README.md`。

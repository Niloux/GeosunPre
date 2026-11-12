# Geosun 预处理与 3DGS 训练整体包

Windows x64，需算力 ≥ 8.0 的 NVIDIA GPU（RTX 30/40/50、RTX A 系列等）、显存 ≥ 8 GB，驱动支持 CUDA 12.8（建议 R570 及以上）。无需安装 Python、CUDA Toolkit、Visual Studio 或联网。

| 目录 | 内容 |
| --- | --- |
| `preprocess/` | GeosunPre 运行包：`geosunpre.exe`、模型和运行库，说明见其 `README.md` 和 `docs/` |
| `trainer/` | Geosun 3DGS Trainer：`bin/geosun_3dgs_train.exe` 和 `config/trainer.json` |
| `geosun.ps1` | 整体入口：预处理 → 交付回读 → 训练 → 导出 |
| `bundle.json` | 两个子包的版本、提交、配对关系和 `package.json` 的 SHA256 |

两个子包各自带运行库，保持目录结构，不要把文件合并到同一目录。将整体包解压到纯 ASCII 路径（例如 `D:\Geosun`），否则训练器不能用 GPU 解码 JPEG。

## 使用

在包目录打开 PowerShell：

```powershell
.\preprocess\geosunpre.exe info
.\preprocess\geosunpre.exe check-models --image "D:/照片/样例.JPG"
.\geosun.ps1 -Scene "D:/数据/场景" -Output "D:/成果/场景"
```

`-Output` 下生成 `delivery/`（训练数据）、`train/project.licht` 和 `export/`（含 `splats.ply`、`mesh.ply` 和 `mesh_osgb/`）。导出由包内训练器完成。默认纹理像素尺寸为 0.02 场景单位，可用 `-TexelSize` 调整。

- 预处理中途失败：以相同命令重跑即续跑，已完成的交付只做回读检查。
- 训练中途失败：移走 `train.pending` 后重跑，或用新的 `-Output`。
- `-Step preprocess` 只预处理，`-Step train` 只对已有交付训练。
- `-Step export` 从已有 `train/project.licht` 补导。失败时保留 `export.pending`，将它移走后可重新补导。
- `-PreprocessArgs` 传给 `geosunpre.exe run`，例如 `-PreprocessArgs '--ply-voxel','0.1'`；`-TrainArgs` 传给训练器，覆盖 `trainer.json` 中的设置，例如 `-TrainArgs '--iter','7000'`。

输入、配置或程序改变时使用新的 `-Output`，保留已有成果。更新任一程序时组装新的整体包，不覆盖旧包。

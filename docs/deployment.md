# GeosunPre 独立运行包

Windows x64，需与模型包匹配的 NVIDIA GPU 和驱动。无需安装 Python、CUDA Toolkit、Visual Studio 或联网。

在本目录打开 PowerShell：

```powershell
.\geosunpre.exe info
.\geosunpre.exe check-models --image "D:/照片/样例.JPG"
.\geosunpre.exe run --input "D:/数据/场景" --work "D:/工作/场景"
.\geosunpre.exe run --work "D:/工作/场景"
.\geosunpre.exe check --output "D:/工作/场景/colmap"
```

模型默认读取 EXE 同级的 `models/`；`--models` 可显式指定其他模型包，续跑优先使用工作目录记录的模型位置。开发脚本通过进程环境变量 `GEOSUNPRE_MODELS` 指定项目模型目录。

`info` 显示程序路径、构建版本、GPU 和默认模型目录。`check-models` 使用指定的真实照片执行四个模型，照片仅用于本次检查，不会写入模型或运行包。

输入、配置或模型改变时使用新工作目录，保留已有成果。默认交付原图、掩码、鱼眼 COLMAP 模型和 5 cm LAS 初始化 PLY；下游需支持 OPENCV_FISHEYE。`--no-ply` 省略 PLY，`--ply-voxel 0.1` 生成 10 cm 初始化。

换 GPU 时需重新准备相应模型，程序的 CUDA 编译目标也须支持该 GPU。完整输入、模型、输出和缓存约定见 `docs/`。

# GeosunPre 独立运行包

Windows x64，需算力 ≥ 8.0 的 NVIDIA GPU（RTX 30/40/50、RTX A 系列等），建议驱动 R570 及以上（开发机验证驱动 610.88）。`package.json` 记录版本、提交、引擎级别和全部文件 SHA256。无需安装 Python、CUDA Toolkit、Visual Studio 或联网。

运行包由项目的 `build.ps1` 编译程序，再由 `package.ps1` 收集程序、模型、运行库和文档。默认准备流程生成 FP16 匹配引擎，并保留关键层的 FP32 计算。将整个运行包复制到目标机器，在包目录打开 PowerShell：

```powershell
.\geosunpre.exe info
.\geosunpre.exe check-models --image "D:/照片/样例.JPG"
.\geosunpre.exe run --input "D:/数据/场景" --output "D:/成果/场景"
.\geosunpre.exe check --output "D:/成果/场景"
```

模型默认读取 EXE 同级的 `models/`；`--models` 可显式指定其他模型包，续跑使用工作目录记录的模型位置。更新程序时从当前源码重新编译并打包，将完整运行包放到新目录。

`info` 显示程序路径、构建版本、GPU 和默认模型目录。`check-models` 使用指定的真实照片执行四个模型，照片仅用于本次检查，不会写入模型或运行包。version=2 模型包的检测器支持动态宽高，检查报告包含实际检测输入尺寸；换场景分辨率或宽高比无需重建模型。

交付直接写入 `--output` 根目录。默认生产模式在交付发布后删除工作目录 `OUT.work` 及其缓存，中途失败时保留，以相同命令续跑；`--debug` 保留工作目录（默认 `work/<输入目录名>`，或 `--work` 指定），用于续跑单阶段及独立 `export`、`refine`、`cleanup`。输入、配置或模型改变时使用新工作目录，保留已有成果。默认交付原图、掩码、鱼眼 COLMAP 模型、5 cm LAS 初始化 PLY，以及 LAS 深度和法向监督图；下游需支持 OPENCV_FISHEYE。`--no-ply` 省略 PLY，`--ply-voxel 0.1` 生成 10 cm 初始化，`--no-las-supervision` 省略监督图。交付目录直接作为定制 LFS 的训练数据路径；监督深度为沿视线距离，须用 `--gut` 训练。旧重建可直接 `export --work OLD --output NEW` 补监督，保留原交付包。

`run` 默认在 LAS 对齐后执行卷帘快门校正并独立清理稀疏点，再导出通过验收的所选模型；交付照片始终为原图拷贝；卷帘校正被采用时另附 `rolling_shutter.json`，未采用的原因见交付目录 `result.json` 的 shutter 及调试工作目录的 `cache/shutter.json`。原图精修默认关闭，首次运行加 `--refine` 或配置 `"refine": true` 在卷帘校正前启用，续跑沿用已冻结的选择。精修未启用或候选未通过时，仍对上一步所选模型校正和清理；未采用原因分别见 `cache/refinement.json`、`cache/shutter.json` 与 `cache/cleanup.json`。

初始匹配包含导航回访候选；SfM 默认不在停滞后额外补匹配，而是处理已注册部分，`sfm_repair_rounds` 可设为 `0`、`1` 或 `2`。实现标记为 18，新流程须使用新工作目录；旧成果仍可 `export`，也可通过 `refine --work OLD --output OUT` 或 `cleanup --work OLD --output OUT` 写入新交付，`--new-work NEW --debug` 保留新工作目录。独立 `refine` 仅执行原图精修，独立 `cleanup` 仅清理稀疏结构；复制旧掩码时均屏蔽无覆盖像素，不重跑语义模型、特征和 SfM。

`ampere_plus` 模型包可直接在上述 GPU 间迁移；算力低于 8.0 时程序明确拒绝加载。使用 `none` 级别的专用模型包时，GPU 型号须与构建 GPU 一致。完整输入、输出和缓存约定见 `docs/`。

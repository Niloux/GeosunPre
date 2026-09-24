# GeosunPre

面向 LichtFeld-Studio 的异步 L/R 双鱼眼视觉 SfM 工程。当前采用 **equal_control 开发基线**：536 张具有同名 CameraPos 记录的 L/R 原图，每张曝光独立求位姿，L/R 各共享一组 OPENCV_FISHEYE 内参。输入 pose 数值、全局 LAS 和 M 图像不参与当前几何拟合；.pro 内参只是初始候选，p3 对应鱼眼第四个畸变参数。该基线已用于 LFS 全训练对照，但尚未通过高图像一致性的最终验收。

## 当前基线

[基线清单与使用说明](baselines/basketball/README.md)固定了等权视觉模型、已有 LFS COLMAP 数据集、配置和关键哈希。默认模型位于 work/basketball/global_sfm/ground_joint_ba/equal_control/sparse/0；新打包的单分支输入位于 work/basketball/baseline_colmap/。LFS 使用原始鱼眼图像与掩码，536 张图全部训练，不划测试集。

从仓库根目录验证或重新打包当前基线：

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre baseline-check --work work/basketball --full
PYTHONPATH=src .venv/bin/python -m geosunpre export-baseline --work work/basketball
```

`baseline-check` 是只读核对；`export-baseline` 只打包已冻结的等权分支，不运行 SfM 或 LFS 训练。导出身份写入 `baseline_export.json`。更改图像、掩码、内参、位姿或配置时，应另建基线版本，不覆盖旧身份。

## 代码和资料

- `src/geosunpre/`：共享相机、特征、几何、BA 和验证部件。
- `src/geosunpre/pipeline/`：输入后的全场数据准备、视觉建图和基线校验/导出。
- `src/geosunpre/experiments/`：局部/迁移对照、加权 A/B 与各轮诊断；独立入口为 `geosunpre-exp` 或 `python -m geosunpre.experiments`。旧 `geosunpre <实验命令>` 暂保留兼容。
- `baselines/basketball/`：纳入 Git 的小型基线身份清单；`work/` 和 `input/` 存放大型产物与原始数据，不纳入 Git。
- [实验报告索引](reports/README.md)和[历史命令与结果](experiments/basketball/HISTORY.md)保留各轮试验的复算入口与未晋升结论。
- [工程边界](docs/engineering_structure.md)说明正式流程与实验的依赖方向。

仅在需要重建全场模型时使用 `geosunpre prepare`、`global-data`、`global-sfm`；它们会处理完整图像集，不是基线校验所需命令。依赖由 `pyproject.toml` 和 `uv.lock` 固定；LoMa 权重和 CUDA 依赖首次安装需要额外下载。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

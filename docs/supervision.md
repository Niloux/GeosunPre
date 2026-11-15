# 深度与法向监督

`run` 的导出阶段及独立 `export` 默认用 MoGe-2 估计注册照片的深度和法向，并用可信 LAS 定尺度。`--supervision-source las` 直接投影 LAS，`--no-las-supervision` 关闭监督。已有调试工作目录可以直接导出到新目录，无需重跑重建：

```powershell
.\run.ps1 export --work "work/场景" --output "D:/成果/场景-supervised"
```

改变导出选项时使用新交付目录。续跑已有无监督交付时，可指定新 `--output`，或加 `--no-las-supervision` 复用原包。

默认最长边 1024 px，深度为沿视线距离的 16 位毫米 PNG，法向使用 OpenCV 相机坐标并朝向相机。`--supervision-max-size` 调整最长边（128–4096），`--supervision-boundary` 设置语义掩码边界的屏蔽宽度（默认 4 个深度图像素，1–32）；遮挡断层的余量按深度自适应，背景侧 2 像素、前景侧约 1 cm。深度须通过静态区域、遮挡、边界和局部深度支撑检查；法向在 LAS 点云上估计并检查平面残差和入射角。两者分别判断有效性，法向无法可靠估计时仍可保留有效深度；无效深度为 0，无效法向为黑色，原照片及 RGB 掩码保持不变。

监督导出只解析一次 LAS，并在同一导出流程中为静态掩码分析与各批次监督投影复用同一份面元缓存。`--supervision-cache-mb` 设置 GPU 缓存上限，默认 2048 MiB（1–4096），且不超过开始时空闲显存的四分之一。超出预算时复用临时紧凑文件，以双缓冲重叠上传和投影；内存保持有界，完成后清除临时文件。该选项不改变监督请求身份、像素含义或重建缓存；监督图像素不保证逐字节可复现，见[输出约定](output.md)。

几何过滤不识别玻璃材质。可用 `--supervision-mask-dir DIR` 追加仅影响几何的排除掩码：按 `DIR/{L,R}/**.JPG.png` 保存原图尺寸灰度 PNG，仅 255 保留；未提供文件的照片沿用原静态区域。不同排除掩码使用新交付目录。

定制 LFS 以交付目录为数据路径，直接读取 `depths/`、`normals/` 和 `supervision.json`：识别到毫米深度和 OpenCV 相机法向后，自动把深度损失切到米制 `metric`、法向损失切到 `camera-opencv`，并关闭单目深度和法向的自动生成；显式指定的非默认选择保持不变。深度是沿视线距离，只有 GUT 光栅化按鱼眼视线渲染，训练须加 `--gut`，否则 LFS 告警且离轴处深度监督有偏。监督默认关闭，用 `--use-depth-loss`、`--use-normal-loss` 开启，权重由 `--depth-loss-weight`、`--normal-loss-weight` 设置。`configs/trainer.json` 统一使用 `mask_mode = "segment_and_ignore"`：室外保留的 `128` 施加天空不透明度惩罚并忽略 `0`；封闭室内经静态掩码将 `128` 全部改回 `255` 后仅含 `{255, 0}`，在该模式下与 `ignore` 等价，避免室内灯箱或天花被挖空。`supervision.json` 记录每图有效像素、生成参数及文件身份，`check` 会回读监督图。

在 LFS 目录中开启监督试训，并与关闭监督的相同设置对照评估：

```powershell
.\build-windows-release\LichtFeld-Studio.exe --headless -d "交付目录" -o "训练输出" --gut --mask-mode segment_and_ignore --use-depth-loss --use-normal-loss
```

MoGe 导出完成后进行跨视角深度一致性检查，每次最多并行处理 8 幅主视图，每个线程最多同时解码两张深度图。

# 深度与法向监督

`run` 的导出阶段及独立 `export` 默认采用**LAS 尺度对齐的 MoGe-2 SSI 监督**（`moge2_ssi`，`depth_alignment = "las_face_bilateral_v1"`）：基于 MoGe-2 ViT-L normal 几何引擎输出尺度偏移不变（SSI）的 16 位相对射线距离图（`depth_source = "moge2_ssi"`、`depth_format = "uint16_png_ssi"`、`depth_type = "ray_distance"`）与 OpenCV 相机系朝向相机的密集法向图（`normal_source = "moge2_normals"`、`normal_frame = "opencv_camera"`），并借由单遍投影的 LAS 射线距离消弭跨子图尺度漂移与近地面系统偏深。`--no-las-supervision` 关闭监督（并跳过静态掩码核验）。已有调试工作目录可以直接导出到新目录，无需重跑重建：

```powershell
.\run.ps1 export --work "work/场景" --output "D:/成果/场景-supervised"
```

改变导出选项时使用新交付目录。续跑已有无监督交付时，可指定新 `--output`，或加 `--no-las-supervision` 复用原包。

## 导出流程与编码约定

1. **LAS 射线深度预投影与缓存复用**：在加载 TensorRT 几何引擎前，沿最终交付相机与有效读出运动执行单遍 GPU 最近 LAS 面元深度投影（静态掩码重算时直接复用同一份 `LasSurfelCache`，避免重复解析 LAS），以 16 位毫米缓存临时写入 `.las_scratch/las_depths.u16` 后立即释放 LAS GPU 缓存并清理临时文件。
2. **透视子图规划与 GPU 重采样**：按监督图相机规划覆盖完整 160° 有效鱼眼视场的 644×644 方形 90° 透视子图（160° 鱼眼通常为 1 张前向加一圈倾斜 50° 共 6–7 张，每条有效视线距子图边缘至少 5°）。原图先按面积缩小到子图中心像素密度的 2 倍，再在 GPU 核函数中双线性重采样到各子图；卷帘快门交付下按每幅照片的读出转角 `ω` 对每个子图像素做 3 步不动点去卷帘反向映射，抽取对齐参考姿态 `s = 0` 的透视子图。
3. **MoGe-2 推理、逐子图 LAS 对齐与三维双边网格细化**：MoGe-2 逐张推理得到 `points`、`normal`、`mask` 与 `metric_scale`，在 64×64 网格上用已知子图焦距恢复 z 偏移并换算为沿视线距离与参考相机系单位法向（强制朝向相机）。对投影 LAS 深度先经 5×5 局部最小抑遮挡清洗（剔除透过前景点隙漏出的背景穿透点），随后分面与 LAS 做稳健尺度和必要时的 Cauchy IRLS 仿射对齐（至少 256 个静态高权重采样；仅在景深跨度充足且明显降差时采用仿射偏移，其余无足够 LAS 支撑的子图回退重叠区中位比值），加权混合到监督网格；再在 `[log d, x, y]` 三维双边网格上对对数残差做保边平滑校正（玻璃穿透与外点阈值 `0.45`，无 LAS 区域平滑衰减回 0），最后按静态允许像素的 `[0.1%, 99.9%]` 分位夹紧 `[min_d, max_d]` 避免极近边角伪影压缩 16 位编码动态范围，并在卷帘快门下将融合法向旋回各像素自身的读出线相机系。
4. **语义掩码与边界屏蔽**：将静态掩码（仅保留 `mask == 255`，排除天空 `128`、动态人物及有效域外 `0`）与可选 `--supervision-mask-dir` 额外排除掩码交集缩放到监督网格（区域内全足迹均为静态才保留），再以 `--supervision-boundary`（默认 4 个监督图像素，1–32）膨胀无效区生成屏蔽边界，屏蔽区在 `depths/` 写为 `0`、在 `normals/` 写为 `(0, 0, 0)`。
5. **SSI 深度与法向 PNG 编码**：
   - `depths/`：16 位单通道 PNG（`CV_16UC1`），`0` 表示无效；对有效允许像素上的对齐后射线距离按 `[min_d, max_d]` 线性映射到 `[0.02, 1.0] * 65535` 并截断至 `[1311, 65535]`（与 LFS `build_depth_png<uint16_t>` 一致）。
   - `normals/`：8 位三通道 RGB PNG（`CV_8UC3`），OpenCV 相机坐标系朝向相机，编码为 `round(127.5 * (n + 1))`，无效像素为 `(0, 0, 0)`。

几何引擎取自 `--supervision-models` 的模型包；未指定时 `run` 用 `--models`，否则用默认模型目录。`--supervision-max-size` 调整最长边（默认 1024 px，128–4096，不放大原图）。静态掩码核验与监督深度投影共用一次 LAS 解析并在 TensorRT 推理前释放缓存，GPU 与显存峰值相互错开；`--supervision-cache-mb` 设置该 LAS 缓存的 GPU 上限，默认 4096 MiB（1–8192，且不超过空闲显存四分之一）。

几何过滤不识别玻璃材质。可用 `--supervision-mask-dir DIR` 追加仅影响几何监督的排除掩码：按 `DIR/{L,R}/**.JPG.png` 保存原图尺寸灰度 PNG，仅 255 保留；未提供文件的照片沿用原静态区域。不同排除掩码使用新交付目录。

## 下游训练与核验

定制 LFS 以交付目录为数据路径，直接读取 `depths/`、`normals/` 和 `supervision.json`：识别到 `depth_format = "uint16_png_ssi"`、`depth_type = "ray_distance"` 与 `normal_frame = "opencv_camera"` 后，保持 `depth_loss_mode = "ssi"`、将 `normal_loss_space` 设为 `camera-opencv`（若为 `auto`），并关闭 `depth_auto_generate` 与 `normal_auto_generate`；显式指定的非默认选择保持不变。深度是沿视线距离，只有 GUT 光栅化按鱼眼视线渲染，训练须开启 `--gut`（见 `configs/trainer.json` 默认配置），否则 LFS 告警且离轴处深度监督有偏。`configs/trainer.json` 统一使用 `mask_mode = "segment_and_ignore"`：室外保留的 `128` 施加天空不透明度惩罚并忽略 `0`；封闭室内经静态掩码将 `128` 全部改回 `255` 后仅含 `{255, 0}`，避免室内灯箱或天花被挖空。`supervision.json` 记录每图有效像素、子图规划、生成参数及文件身份，`check` 会逐图回读校验。

在 LFS 中以交付目录试训示例：

```powershell
.\LichtFeld-Studio.exe --config "交付目录/trainer.json" -d "交付目录" -o "训练输出"
```

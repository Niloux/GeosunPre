# 篮球场：LichtFeld-Studio 3DGS 受控 A/B

本轮把已保存的等权全场 BA 和近地点增权全场 BA 输出成两个 **COLMAP 格式**数据集，并用当前 LFS `lfs-train` 实际导入。536 张有精确同名 pose 记录的 L/R 原始鱼眼图像全部参加训练；M、无 pose 图像、输入 pose 数值和全局 LAS 均未参加 SfM 或初始化。两组分别保留每次曝光的独立位姿，没有同步 rig 约束。

## 同一输入，只比较相机解

- [等权 COLMAP 数据集](../work/basketball/global_sfm/lfs_ab/equal_control)和[近地点加权 COLMAP 数据集](../work/basketball/global_sfm/lfs_ab/near_floor_2p0)各有 `sparse/0/cameras.bin`、`images.bin`、`points3D.bin`，并共用 `images/`、`masks/` 与 `sparse/0/points3D.ply`。LFS 已确认优先加载共同的 PLY；二进制 `points3D.bin` 也统一为等权模型，以免改变点云加载选项时悄悄更换初始化。
- 两组的 536 个图像名、图像 ID、2D 观测位置和三维点关联一致；535 个图像的保存位姿不同，L/R 的内参也有小幅差异。原图不重采样。共同点云从**等权视觉 SfM**按点重投影误差 ≤2 原图像素、轨迹长度 ≥4、离相机中心中位位置 ≤50 视觉模型单位筛到 35,279 点。这个距离不是米，不把视觉点云当作 LAS 配准真值。
- [共同掩码](../work/basketball/global_sfm/lfs_ab/shared/masks)是修正版语义有效域与同一候选 160° 鱼眼 FOV 的交集，白色有效、黑色忽略。它是训练有效域，不是镜头硬件视场的真值。此前 `masks_v2/valid` 只含天空、人物和设备有效性，直接拿去训练会漏掉硬 FOV。
- [冻结协议](../work/basketball/global_sfm/lfs_ab/protocol.json)记录所有 536 张原图和掩码的内容哈希汇总、两组 COLMAP 文件、共同点云和训练配置哈希。未覆盖原全场模型；加权候选仍只是待比较方案。

[运行身份](../work/basketball/global_sfm/lfs_ab/runtime_identity.json)记录当前 LFS Git HEAD 和实际 `lfs-train` 二进制 SHA256；LFS 工作区的配置文件有本地修改，因此复跑以二进制哈希和导出配置为准。

## LFS 导入与训练约定

[完整 LFS 配置](../work/basketball/global_sfm/lfs_ab/lfs_config.json)来自本机完整篮球场模板，但去掉了旧数据集路径、旧输出路径和 LiDAR `init_path`。配置采用原生 `OPENCV_FISHEYE`，`gut=true`、`undistort=false`、`mask_mode=ignore`、半分辨率 1500×2000、MRNF 与每曝光校正。两组相同，30,000 步，在 7,000 / 15,000 / 30,000 步保存并评价。

按用户要求 `use_test_split=false`，**536 张全部训练**。LFS 每镜头每 8 张取 1 张，合计 68 张，仅用于训练视图渲染诊断；它们没有从优化中留出。PSNR、SSIM、LPIPS 和后处理得到的近地点/鱼眼外圈误差都不能叫独立泛化测试。LFS 训练采样器在当前代码中使用固定种子，两个进程顺序运行，不并发争用 GPU。

两组均通过 `--iter=1` 的实际导入与 GPU 前置核对：[等权日志](../work/basketball/global_sfm/lfs_ab/preflight_equal.log)、[加权日志](../work/basketball/global_sfm/lfs_ab/preflight_weighted.log)显示各读入 2 个 COLMAP 相机、536 张图、536 张掩码、共同的 35,279 点 PLY，且训练集为 536 张。LFS 加载时会打印 “Undistort” 预计算日志；实际将相机准备为去畸变针孔只发生在 `optimization.undistort=true` 时，本配置为 `false`，训练路径继续用鱼眼畸变。1 步运行只是导入检查，不构成重建质量证据。

顺序训练入口为[run_ab.sh](../work/basketball/global_sfm/lfs_ab/run_ab.sh)。训练完成后自动输出[配对 JSON](../work/basketball/global_sfm/lfs_ab/lfs_comparison.json)和[可读报告](../work/basketball/global_sfm/lfs_ab/lfs_comparison.md)。分析程序在相同评价图像上逐张对比 LFS 指标，并从 LFS 保存的 GT／render 并排 PNG 中计算有效区、可靠近地点轨迹周围 12 半分辨率像素补丁、鱼眼候选有效半径外 30% 的 8-bit MAE。近地点掩区只由冻结的等权视觉模型训练轨迹决定，两个分支相同，不使用 LAS 或输入 pose 值；这些依然是训练视图指标。

训练使用 `--no-download`，当前环境没有 LPIPS 权重，LFS 输出的 LPIPS 列为空；实际对照使用 PSNR、SSIM 与上述有效区 MAE。

## 30,000 步结果与选择

两组训练及自动分析均正常完成。最终评价的 **68/68 张训练视图逐张对应**，两组保存的 GT 面板也逐像素一致。下表的 MAE 是共同有效掩码中的 8-bit RGB 绝对误差，越低越好；每张图先算误差，再对 68 张取均值。

|范围|PSNR 等权 → 加权 (dB)|全有效区 MAE 等权 → 加权|近地点 MAE 等权 → 加权|鱼眼外圈 MAE 等权 → 加权|
|---|---:|---:|---:|---:|
|全部|19.9286 → 19.9214|17.7135 → 17.7123|14.5478 → 14.4943|18.5984 → 18.5844|
|L|20.7269 → 20.7098|15.8115 → 15.8476|12.8944 → 12.8699|16.6689 → 16.6695|
|R|19.1303 → 19.1330|19.6156 → 19.5770|16.2012 → 16.1187|20.5278 → 20.4993|

近地点 MAE 平均降低 0.0535 个 8-bit 色阶，约为等权组近地点 MAE 的 0.37%；68 张中 42 张下降，L/R 各为 21/34 张。近地点的改善方向与先前偏好一致，但幅度很小；全有效区基本持平，L 的全有效区略变差。抽看的[球场 L 等权](../work/basketball/global_sfm/lfs_ab/runs/equal_control/eval_step_30000/32.png)、[球场 L 加权](../work/basketball/global_sfm/lfs_ab/runs/near_floor_2p0/eval_step_30000/32.png)、[步道 R 等权](../work/basketball/global_sfm/lfs_ab/runs/equal_control/eval_step_30000/33.png)和[步道 R 加权](../work/basketball/global_sfm/lfs_ab/runs/near_floor_2p0/eval_step_30000/33.png)中，地面细线与纹理的模糊仍然明显，没有可确认的整体视觉跃升。

因此目前**保留等权相机解为默认 LFS 输入**，近地点加权解保留为可复现候选，不将其晋升为最终相机解。这个选择针对当前相同图像、掩码、初始化和 LFS 配置的单次完整训练；两组数值都是训练视图诊断，没有独立测试集，也没有 LPIPS 结果。下一轮若要提高最终图像一致性，应继续定位地面重影与 L/R 差异的来源，不能把这 0.0535 的近地点差异视为已解决。

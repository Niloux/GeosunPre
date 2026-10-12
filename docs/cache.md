# 原生缓存

所有数值小端。字符串为 uint64 字节长度及 UTF-8 字节，无终止符；格式标记也是这种字符串。不同版本不做自动兼容。

scene.json 保存曝光、初始相机和输入元数据。state.json 版本 1，保存冻结场景、配置、模型元数据和阶段产物元数据。文件变化时拒绝复用；输入、配置或模型改变时使用新 work。

`cache/features/索引.bin`：字符串标记 `GEOSUN_FEATURES_1`；uint32 image、dimension；uint64 N（32–3072）；N 条 float32 `(x,y,nx,ny,score)`；N×256 个 float32 描述子；256 个 float32 检索向量（描述子均值的 L2 归一化）。

`cache/matches/a_b.bin`：标记 `GEOSUN_MATCHES_1`；uint32 a、b（a<b）；uint64 M；M 条 `(uint32 feature_a,feature_b,float32 score)`。候选在 pairs.json。补匹配保持特征身份、重新数值求解，最多两轮针对未注册图像补边。

`cache/{visual,aligned}.bin`：标记 `GEOSUN_RECONSTRUCTION_1`；相机 JSON 字符串；uint32 gauge_image、scale_image；float64 gauge_baseline；uint64 图像数、点数。图像为 uint32 camera、uint8 registered、7 个 float64 `qx qy qz qw tx ty tz`、uint64 N、N 对 float64 像素坐标。点为 3 个 float64 XYZ、3 个 uint8 RGB、float64 error、uint64 轨迹长度及 `(uint32 image,feature)` 列表。点引用由轨迹重建；空轨迹保留内部编号，导出时省略。

掩码为 `cache/masks/索引.png`。阶段完成后才提交状态，中断时保留完成的单图/单对缓存；`.writing` 不作为完成依据。残留交付 `.pending` 在重试时改名保留，再重新生成。

`cache/incremental.bin` 使用同一重建格式，保存 SfM 的部分注册结果，每次局部优化后原子更新。补匹配及中断恢复会重新验证匹配几何、重建候选关联，保留已有位姿、三维点和观测身份；该文件不作为视觉验收或导出依据。

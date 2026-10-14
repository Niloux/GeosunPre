# 原生缓存

所有数值小端。字符串为 uint64 字节长度及 UTF-8 字节，无终止符；格式标记也是这种字符串。不同版本不做自动兼容。

scene.json 保存曝光、初始相机和输入元数据。state.json 版本 1，保存冻结场景、配置、模型元数据和阶段产物元数据。文件变化时拒绝复用；输入、配置或模型改变时使用新 work。

state.json 的 pipeline_version=8 标识当前实现，恢复时须与程序的实现标记一致；不匹配时使用新目录。状态与结果保存本次执行的构建版本、时间和 EXE 路径。cache/sfm.json 保存几何验证、轨迹补全和各次 BA 的收敛、有效观测及耗时，最终 result.json 同时包含此报告。timing 细分缓存读取、轨迹构建、候选选择、位姿注册、三角化、过滤、检查点和着色耗时。导航误差按最终所选模型计算。

cache/{masks,features,matches}_profile.json 保存本次阶段的计算及复用数量、读图或缓存读取、预处理、输入等待、写入和推理耗时，最终 result.json 的 frontend 字段包含这些报告。匹配报告还保存配对总数、每图连接数及定向补边的图像编号。GPU 推理由 CUDA 事件测量；累计工作时间可以重叠，不能简单相加为总耗时。恢复运行的报告仅计本次计算，补匹配记录在 matches 报告的 repairs 数组，续跑沿用已完成的轮数。

`cache/features/索引.bin`：字符串标记 `GEOSUN_FEATURES_2`；uint32 image、dimension；uint64 N（32–3072）；N 条 float32 `(x,y,nx,ny,score)`；N×256 个 float32 描述子；256 个 float32 检索向量（描述子均值的 L2 归一化）；N 条 uint8 RGB。颜色取原图对应像素，避免 SfM 末尾重复解码照片。

`cache/matches/a_b.bin`：标记 `GEOSUN_MATCHES_1`；uint32 a、b（a<b）；uint64 M；M 条 `(uint32 feature_a,feature_b,float32 score)`。候选在 pairs.json。初始近邻按曝光排序编号保留间隔 1–4、6、8、10、12；检索按编号距离 `[12,48)`、`[48,160)`、`[160,N)` 分段，每段从前 4 个候选选最多 3 个，排除同镜头相隔不足 2 秒的重复候选，L/R 不作同步假设。补匹配由 `sfm_repair_rounds` 控制，取值 `0`（默认）、`1` 或 `2`；启用时保持已有边和特征身份，针对未注册或未达到每图 32 个观测、8 个覆盖格子的图像扩展密集近邻及检索，补边不执行初始稀疏化及检索去重。

`cache/{visual,aligned}.bin`：标记 `GEOSUN_RECONSTRUCTION_1`；相机 JSON 字符串；uint32 gauge_image、scale_image；float64 gauge_baseline；uint64 图像数、点数。图像为 uint32 camera、uint8 registered、7 个 float64 `qx qy qz qw tx ty tz`、uint64 N、N 对 float64 像素坐标。点为 3 个 float64 XYZ、3 个 uint8 RGB、float64 error、uint64 轨迹长度及 `(uint32 image,feature)` 列表。点引用由轨迹重建；空轨迹保留内部编号，导出时省略。

掩码为 `cache/masks/索引.png`。阶段完成后才提交状态，中断时保留完成的单图/单对缓存；`.writing` 不作为完成依据。残留交付 `.pending` 在重试时改名保留，再重新生成。

`cache/incremental.bin` 使用同一重建格式，保存 SfM 的部分注册结果，每次局部优化后原子更新。补匹配及中断恢复会重新验证匹配几何、重建候选关联，保留已有位姿、三维点和观测身份；该文件不作为视觉验收或导出依据。

默认首次停滞就对已注册部分完成最终优化和验收；显式启用补匹配时，在轮数用完后处理已注册部分。通过后写入 visual.bin 并继续配准、精修、导出。所有重建缓存保留全部输入索引及 registered 标记，未注册图像不出现在点轨迹中。cache/unregistered.json 保留 SfM 阶段未注册的原输入索引；全部注册后删除。state.registration 保存最终所选模型的注册统计，sfm 报告的 quality.registration 保存精修前的统计。status=completed 表示交付完成，是否全注册看最终 registration.complete。

默认阶段为 `prepare → masks → features → matches → sfm → align → refine → export`。align 产物为 `cache/aligned.bin` 和 `cache/alignment.json`，精修不改写它们。refine 使用同一重建格式写入 `cache/refinement_candidate.bin` 和 `cache/refined.bin`：前者用于诊断候选，后者始终为最终所选模型，即使精修候选未通过也保存原模型。`cache/refinement.json` 分别记录相机优化及强力清理的验收、采用结果和原因；`result.json` 保留原阶段报告，并更新最终 quality、导航误差和 refinement。`selected_model=refined` 指定导出及独立精修读取 refined.bin；未声明该字段的旧结果仍读取 aligned.bin。

refine 完成后才提交阶段状态，所有产物纳入文件元数据校验；中断时重跑该阶段，不将未完成候选视为成果。续跑会复用已完成精修。默认导出和独立 `export` 都使用 result.json 指定的模型，其元数据也用于交付包的请求身份。

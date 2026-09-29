# 工程结构与运行说明

## 模块职责

```text
src/geosunpre/
  cli.py                  # run / export / check
  config.py               # 场景选择、匹配后端及冻结身份
  artifacts.py            # 内容哈希、原子 JSON 写入、不可变协议
  scene.py                # Geosun 输入校验、L/R 图像选择
  camera.py               # 原生鱼眼相机与像素约定
  masks.py                # SegFormer、固定设备遮挡、预览
  features.py             # LoMa 与内容缓存
  matching.py             # FlashAttention、诊断 math 路径及 CUDA Graph
  parallel.py             # 有界、有序的 CPU 并行任务
  profiling.py            # RSS / CUDA 采样、嵌套计时与空闲内存回收
  colmap.py               # COLMAP 原子发布和独立校验
  pipeline/
    runner.py             # 阶段顺序、最终 BA、正式导出
    runtime.py            # 工作目录锁、阶段状态与耗时
    data.py               # 全场掩码、特征、候选配对和匹配
    observations.py       # 生产全量观测身份与模式分发
    holdout.py            # 验证模式冻结留出观测
    quality.py            # 生产质量诊断与报告
    calibration.py        # 复用全局特征做局部标定
    reconstruction.py     # 全局建图、轨迹筛选、BA 与报告
  sfm/
    geometry.py           # 两视图几何、紧凑成轨与旋转闭环检查
    tracks.py             # 时序配对与留出抽样
    reconstruction.py     # 数据库、种子模型与有界 BA
    depth_guard.py        # 正深度约束
    track_filter.py       # 训练轨迹一致性筛选
    evaluation.py         # 冻结查询的几何预测评估
    statistics.py         # 成对轨迹统计
```

基础模块不依赖阶段编排。`pipeline/` 组合这些部件；`colmap.py` 可独立使用，读取已有模型和场景即可导出。正式入口不依赖 `work/` 下的实验脚本。原生扩展位于 `native/`，是主工程必需依赖；math/Python 路径仅供显式诊断和旧配置续跑。

## 工作目录

```text
WORK/
  pipeline_config.json     # 场景选择配置及身份
  pipeline_status.json     # 最近一次调用的阶段状态
  run_result.json          # 最近成功调用的结果
  .pipeline.lock           # flock 锁；文件留存不表示进程仍运行
  manifest.json            # 初始输入清单
  images.txt
  fov_masks/
  features/                # 内容寻址的特征缓存
  matches/                 # 按匹配后端区分身份的匹配缓存
  calibration/
    manifest.json
    frontend.json
    split.json             # 仅验证模式
    run_contract.json
    mapping/
    stages/
    calibration_identity.json
    summary.json
  sfm/
    manifest.json
    masks_v2/              # 版本 2 的语义、设备掩码和原因图
    mask_review.html
    frontend.json
    observation_protocol.json  # 生产模式
    holdout_protocol.json      # 以下三项仅验证模式
    split.json
    triangle_validation_split.json
    geometry_protocol.json
    mapping/
    stages/
    optimization_summary.json  # 生产模式
    comparison.json            # 验证模式
    final/sparse/0/        # 最终原生鱼眼模型
    final/quality_summary.json     # 生产模式
    final/validation_summary.json  # 验证模式
    index.html
  colmap/                  # 可交付数据集
```

上表展示 local 标定模式；input 模式的标定阶段仅冻结输入内参身份，不生成局部建图与 BA 产物。

工作目录使用绝对路径引用原始数据和缓存，不承诺可搬迁。`colmap/` 的复制模式用于可搬迁交付。

## 运行与恢复

1. `run` 获得 Linux `flock` 排他锁，再写入配置和阶段产物。另一个进程使用相同目录时立即失败。
2. 新目录保存版本 4 配置，默认 production + local + flash；版本 1/2/3 继续按 validation 运行；续跑加载已保存配置。版本 1 未写匹配后端时解释为 math，显式 flash 不变；旧文件保持原字节。命令行传入不同配置、不同输入根目录，或未由新流程管理的非空目录，均拒绝混用。
3. JSON 先写同级临时文件再替换。冻结协议逐项比对内容；特征和匹配的内容哈希继续贯穿标定、全局几何和最终导出。
4. 状态文件只是执行记录。是否能复用结果由实际产物、完成身份和依赖检查决定。
5. 全量重跑会检查并复用完整缓存或阶段；`--stage` 只运行指定阶段。状态标记为 `completed` 仅表示本次请求的阶段完成。
6. Python 异常或 Ctrl-C 会记录失败或中断。进程被强制杀死时可能仍显示 `running`；进程退出后系统会释放锁，重跑将核对实际产物。

恢复支持存在明确边界：全局建图可归档中断的数据库后重建；局部标定遇到不完整数据库、模型或报告时保守报错；最终 BA 若留下不完整模型也拒绝直接覆盖。遇到此类提示应使用新工作目录重新运行，并保留失败目录用于定位。导出采用整目录发布，不暴露未校验的数据集。

阶段结束后，状态文件的 `resources` 保存 RSS、CPU 时间和 PyTorch CUDA 分配器指标；
`operations` 保存子操作的包含子调用耗时，父子耗时不能相加。
`host_memory_reclaim` 记录空闲堆页回收前后的 RSS；指标口径及缓存预算见 [资源报告](resources.md)。

## 固定的数值流程

保留已实跑的视觉流程：1536 检测尺寸、每图最多 3072 个 LoMa 特征，时序加描述子检索候选配对；生产模式使用全部质量筛选后的观测；验证模式在所有几何拟合前冻结全局留出；局部依次优化固定内参、焦距与低阶径向、全部径向、主点；全局依次固定标定 BA、训练轨迹筛选 BA、共享标定 BA，最后执行最多 40 次迭代的有界 BA。

相机共享内参，每次曝光使用独立 frame。正深度与 160° 单调投影检查、训练覆盖检查、全部图像注册要求均保留。标定片段选择属于配置；不根据留出误差搜索参数或选择模型。

生产与验证模式的差异和来源契约见 [模式说明](production_pipeline.md)。

## 当前边界

- 输入适配 Geosun 的 `.pro`、CameraPos 和 L/R 原图命名；设备遮挡规则针对当前设备。
- SfM 无人工标注依赖，掩码预览只供检查，不阻塞流程。
- 导出保持原生 `OPENCV_FISHEYE`；使用方需要支持该相机模型。
- 视觉结果为任意尺度，不提供导航对齐、米制尺度、地面约束或 LAS 融合。
- 数值与阶段测试包含真实 PyCOLMAP 模型读写和小模型 BA；管线集成测试对大规模建图与 BA 使用替身。
- 当前生产完整运行见 [生产报告](production_pipeline.md)：20 分 31.30 秒，未达到 20 分钟目标；未验证下游渲染质量。历史性能、资源与消融记录见 [验证摘要索引](benchmarks/README.md)。

## 历史重构验证（2026-09-28）

- 39 项测试通过，包含小模型实际 BA、留出隔离、配置冻结、工作目录锁、失败记录与 COLMAP 迁移校验。
- 依赖锁检查通过，0.2.0 wheel 和源码包构建成功；源码包包含配置、文档与测试，不包含运行数据。
- wheel 安装到独立目录，从源码目录外执行命令。真实篮球场 `prepare` 选出 536 张图和 50 张标定图。
- 新 `check` 对此前完整运行的交付包执行内容校验通过：536 张图、61,495 个点，包含组合掩码。
- 2026-09-28 该轮未重新执行完整 GPU 掩码推理、特征匹配和全场重建；不据此声称重构后的数值结果已经全量复现。

# 原生 BA 约束

`geosunpre-native` 是主工程必需的本地扩展，由 `uv sync` 构建安装。
需要 C++17 编译器及 Python 开发头文件；pybind11 版本固定为 3.0.1。

扩展实现 `PositiveDepthCost` 的残差和解析 Jacobian，替代 BA 中的 Python 回调，
允许使用四个求解线程。`Sim3PointPlaneCost` 和 `Sim3PositionPriorCost` 支持独立全局配准变量下的
LAS 点到面约束与相机中心位置先验；其 Jacobian 也在 C++ 中计算。
兼容 pyceres 2.6 / Ceres 2.2.0，导入时检查 Ceres 版本，当前实现版本为 2。
修改 pyceres、pybind11 或 Ceres 版本时须重新核对 ABI 并执行数值与实际求解测试。

全局变换使用 Eigen 顺序四元数 `[x,y,z,w]` 和 `[tx,ty,tz,log_scale]` 两个参数块。
相机先验使用 `cam_from_world` 的中心 `-R.T @ t`，没有使用输入姿态四元数。
`joint_costs.h` 已声明为增量编译依赖，并随源码包发布。

`vendor/` 仅包含 Ceres 的 CostFunction 接口与构造/析构实现，不包含另一份求解器。
来源与修改说明见 [vendor/README.md](vendor/README.md)，原许可证见
[vendor/LICENSE.ceres](vendor/LICENSE.ceres)。许可证随源码包和 wheel 分发。

从仓库根目录构建独立发行包：

```bash
uv build native
```

正式默认要求原生模块；`GEOSUNPRE_DEPTH_BACKEND=python` 仅供诊断对照。
性能与完整数据验证见 [性能记录](../docs/performance.md)。

修改扩展后，在项目环境中重新安装：

```bash
uv sync --reinstall-package geosunpre-native
```

默认 joint 主流程和 `joint-alignment` 会明确检查实现版本 2 及新残差是否存在；
`uv run --no-sync` 不会替你重建旧扩展。
解析导数与实际 Ceres 求解测试见 [联合优化测试](../tests/test_joint_alignment.py)。

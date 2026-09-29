# 原生深度约束

`geosunpre-native` 是主工程必需的本地扩展，由 `uv sync` 构建安装。
需要 C++17 编译器及 Python 开发头文件；pybind11 版本固定为 3.0.1。

扩展实现 `PositiveDepthCost` 的残差和解析 Jacobian，替代 BA 中的 Python 回调，
允许使用四个求解线程。兼容 pyceres 2.6 / Ceres 2.2.0，导入时检查 Ceres 版本。
修改 pyceres、pybind11 或 Ceres 版本时须重新核对 ABI 并执行数值与实际求解测试。

`vendor/` 仅包含 Ceres 的 CostFunction 接口与构造/析构实现，不包含另一份求解器。
来源与修改说明见 [vendor/README.md](vendor/README.md)，原许可证见
[vendor/LICENSE.ceres](vendor/LICENSE.ceres)。许可证随源码包和 wheel 分发。

从仓库根目录构建独立发行包：

```bash
uv build native
```

正式默认要求原生模块；`GEOSUNPRE_DEPTH_BACKEND=python` 仅供诊断对照。
性能与完整数据验证见 [性能记录](../docs/performance.md)。

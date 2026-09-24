# 篮球场实验区

实验代码位于 [geosunpre.experiments](../../src/geosunpre/experiments)，与 [正式管线](../../src/geosunpre/pipeline) 分开。这里保存复算入口和实验结论；大型产物仍位于被 Git 忽略的 `work/basketball/`，没有因目录整理而移动。

从仓库根目录运行实验命令：

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre.experiments --help
```

已有报告中的 `geosunpre <实验命令>` 暂时保留兼容；新的复算记录应使用 `geosunpre-exp` 或上述模块入口。实验不得就地覆盖 [等权开发基线](../../baselines/basketball/README.md)。LFS A/B、近地点加权 BA、轨迹清理及球场点位 probe 的候选结果均应根据冻结协议、原图检查和对照证据显式决定是否晋升。

- [实验报告索引](../../reports/README.md)：按输入、全场模型、LFS、前端和球场点位导航。
- [历史命令与结果](HISTORY.md)：此前根 README 的逐轮复算说明，命令从仓库根目录执行。
- [工程边界](../../docs/engineering_structure.md)：共享函数、正式管线及实验之间的依赖规则。

当前报告和工作目录中的 `DIAGNOSTIC_ONLY.json`、`NOT_PROMOTED.json` 是结论的一部分。用户提供的粗标 JSON 保留原件，但不能当精确像素真值；训练视图指标不能被称作新视角验收。

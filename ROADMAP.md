# XUVDB 路线图

定位:Python 原生的**可编辑稀疏体积 + 互通层**。上不卷 fVDB 的训练算子(稀疏卷积/
attention/可微训练是 fvdb-core 的领地),下不做邻居搜索(求解器侧的事);守住三者中
独有的生态位:零 C++ 构建即得"宿主端任意结构编辑 + 内核端可写值 + 纯 Python 读写
OpenVDB 流"。

## v0.2 — 遍历与加速

- [ ] 叶级 active bbox 缓存:`stamp/csg/fill/prune/load` 时失效重算;
- [ ] 内核端 DDA:`ray.py` 的块跳逻辑移植为 quadrants kernel,可用 GPU 后端;
- [ ] `iter_voxels` / 批量索引查询向量化,消除逐体素 Python 循环。

验收:射线基准对宿主实现加速比明确;`pytest` 全绿;GPU 后端冒烟通过。

## v0.3 — 采样与统计

- [ ] Triquadratic(三次)采样器(宿主 + 内核);
- [ ] 宿主侧梯度 / stencil API(7/19 点差分模板);
- [ ] 每叶 min/max 统计缓存(带宽门控、快速极值查询);
- [ ] GPU map-reduce 原语(active 求和/极值/计数)。

验收:三次采样与 OpenVDB 参考实现数值对齐;统计缓存与暴力计算一致。

## v0.4 — 生态互通

- [ ] `.vdb` 读侧支持 Blosc(Houdini 默认压缩,当前明确报错);
- [ ] PointDataGrid 读侧(吃 Houdini 粒子缓存);
- [ ] `.xuvdb` v2:校验和 + 可选压缩 + 版本迁移;
- [ ] f16 值类型;仿射变换(带旋转)。

验收:Houdini 默认设置导出的真实文件可读;损坏文件被校验和拦截。

## v0.5 — torch 桥(fVDB 方向的克制版:做互通,不做框架)

- [ ] 值缓冲 ⇄ `torch.Tensor`(含 CUDA 路径);
- [ ] 可微采样包装(自定义 autograd function 包住 `GpuVolume.sample`);
- [ ] 多网格批量打包/传输(GridBatch 式一次上下载)。

不做:稀疏卷积 / attention / 训练算子——正确姿势是提供导出接口。

## v1.0 — 稳定承诺

- [ ] `.xuvdb` v2 格式冻结 + 向后兼容保证;
- [ ] semver 承诺、公开 API 审查(定死 `stamp_sphere` 的覆盖 vs min-union 语义);
- [ ] 与 NanoVDB 的公开性能基线对比。

---

依赖面变化从 v0.4 起须在 changelog 显式声明(blosc 依赖、torch extra)。

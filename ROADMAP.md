# XUVDB 路线图

定位:Python 原生的**可编辑稀疏体积 + 互通层**。上不卷 fVDB 的训练算子(稀疏卷积/
attention/可微训练是 fvdb-core 的领地),下不做邻居搜索(求解器侧的事);守住三者中
独有的生态位:零 C++ 构建即得"宿主端任意结构编辑 + 内核端可写值 + 纯 Python 读写
OpenVDB 流"。

## v0.2 — 遍历与加速 ✅ 0.2.0

- [x] 叶级 active bbox 缓存:`stamp/csg/fill/prune/load` 时失效重算;
- [x] 内核端 DDA:`ray.py` 的块跳逻辑移植为 quadrants kernel(`GpuVolume.ray_surface_hit`,
  批量射线一次 launch),CPU / CUDA / Vulkan 后端验证通过;
- [x] `active_indices` / `active_values` / `probe_batch` 向量化查询,`_sample_linear`
  小批量走 `get_value`、大批量走 `probe_batch`。

实测(GTX 1650,300 粒子 union_spheres 网格,1000 射线):内核批量 DDA 在 CUDA 上
**7.0 us/射线**,宿主标量 DDA 2.4-3.3 ms/射线(约 350-470 倍);宿主 bbox 跳跃在密集
窄带网格上收益中性,内核批量路径是推荐入口。

## v0.3 — 采样与统计 ✅ 0.3.0

- [x] Triquadratic(三次)采样器(宿主 + 内核):3x3x3 二次 B 样条,权重
  `[0.5(1-u)^2, 0.5+u-u^2, 0.5u^2]`,与 OpenVDB `QuadraticSampler` 同式;
- [x] 宿主侧 `sample_gradient`(中心差分,order=1/2)+ `stencil7_batch` / `stencil19_batch`;
- [x] 每叶 min/max 统计缓存(`Leaf.value_range` / `VdbGrid.value_range`,
  与 bbox 共用 `invalidate()` 失效点);
- [x] GPU map-reduce:`GpuVolume.reduce()`(active 求和/极值/计数,一次 launch,
  掩码按 u32 打包进 packed volume)。

验收:三次采样与稠密独立参考实现逐点对齐(atol 1e-5)、内核-宿主奇偶(atol 2e-5)、
常量场复现、球面梯度径向;统计与暴力计算一致;CPU/CUDA/Vulkan 三后端冒烟通过。

## v0.4 — 生态互通 ✅ 0.4.0

- [x] `.vdb` 读侧支持 Blosc(OpenVDB `bloscToStream` 帧级语义:i64 长度前缀、clevel 9 +
  byte shuffle;可选 `blosc` 包,缺包时明确报错),并补了**写侧**(可产生 Blosc 文件);
- [x] `.xuvdb` v2:CRC32 校验和(恒在)+ 可选 zlib 载荷压缩 + 旋转矩阵字段,
  v1 文件保持可读,损坏文件被校验和拦截;
- [x] f16 值类型(原生格式 + 宿主;`.vdb` 写侧升格 f32,读侧 half 网格本就支持);
- [x] 刚体仿射变换(`rotation` 正交阵:world = R@(index·s)+t,宿主采样/射线/stamp/
  自有格式/`.vdb` AffineMap 读写全通;内核仅轴对齐,旋转网格显式报错)。
- [ ] PointDataGrid 读侧 → **顺延 v0.5**:没有真实 Houdini 粒子缓存做测试,不可证伪的
  代码不发布(读侧遇点云网格仍报明确错误)。

验收:Blosc 往返逐值一致;被篡改文件被 CRC 拦截;旋转网格采样/射线/两种格式往返与
轴对齐数值一致;34+ 测试全绿。

## v0.5 — torch 桥 ✅ 0.5.0(fVDB 方向的克制版:做互通,不做框架)

- [x] 值缓冲 ⇄ `torch.Tensor`(`grid_to_tensors` / `tensors_to_grid`,任意 device;
  f16/f64 上转 f32,轴对齐标量网格);
- [x] 可微采样 `sample_t` / `sample`:自定义 autograd Function,对叶值(scatter-add)
  和采样点(角权重解析导数)都有梯度,双向对有限差分校验;
- [x] 多网格批量打包 `gpu.VolumeBatch`:多网格共享一份 key/value 缓冲,各自变换/叶尺寸,
  `sample(points, volume_ids)` 一次 launch 全部采样(分段二分);
- [ ] PointDataGrid 读侧 → 继续顺延:仍无真实 Houdini 粒子缓存可验证(不可测不发布)。

不做:稀疏卷积 / attention / 训练算子——正确姿势是提供导出接口。

## v1.0 — 稳定承诺 ✅ 1.0.0

- [x] `.xuvdb` v2 格式冻结,v1 永久可读(测试锁定 `io.VERSION == 2`);
- [x] semver 承诺 + 公开 API 冻结清单(测试锁定;`xuvdb.kernels` 声明为内部);
- [x] `stamp_sphere` 语义定案:SDF stamp 按 min-union 复合(与 `csg('union')` 逐叶等价,
  测试锁定),fog stamp 覆盖写——README 快速上手里"两次 stamp 即 CSG"从愿景变为事实;
- [x] 公开性能基线 [BENCHMARKS.md](BENCHMARKS.md)(可复现脚本;含与 NanoVDB 的设计级
  对比说明——真正的 head-to-head 需要 NanoVDB C++ 构建,欢迎贡献)。

---

依赖面变化从 v0.4 起须在 changelog 显式声明(blosc 依赖、torch extra)。

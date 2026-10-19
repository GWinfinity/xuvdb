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

## v1.3.4 — 解析器加固（安全审计 High/Medium 修复）✅ 1.3.4

针对外部二进制审计（3 High / 3 Medium / 5 Low，全部经 PoC 实测）修复高/中危项，
Low 项（L-1~L-5）与"建议级"改动维持原状。行为变化：**越界叶坐标的文件现在在解析
阶段被拒绝**（此前载入成功但 GPU 采样静默撞键）；超大声明量的文件被拒绝或警告
跳过。验收以审计复测判据为准（PoC 构造式回归，见 `tests/test_security.py`，17 项）：

- [x] **H-1 物化放大**：`n_tiles`/`n_int5`/`n_leaves` 按"剩余字节数可装下的最小记录数"
  校验（伪造即 ValueError）；`.vdb` tile 物化受 `max_tile_voxels` 与全局 2^24 体素预算
  双重约束（耗尽警告并跳过，8.4 KB mask 文件不再能换 343 GB）；`.xuvdb` 单网格物化
  超 2^24 体素拒绝并指向 `iter_leaves`/`iter_slabs`；
- [x] **H-2 解压炸弹**：容器级 zlib 统一走 `decompressobj(max_length)` 有界解压
  （小文件 ≥256 MiB 余量、≤64:1 压缩比、硬顶 4 GiB），CRC 移到解压后——CRC 防篡改
  不防资源耗尽；`.vdb` ZIP 值块补齐解压后长度比对（对齐既有 blosc 分支）；
- [x] **H-3 键空间溢出**：`xuvdb_pack_key` 内核侧越界返回 -1（查询回退背景，与宿主
  一致；不再进位串轴撞键）；`gpu._pack_key` 越界显式 ValueError；`io.load` /
  `iter_leaves` / `read_vdb` 解析期拒绝越界叶/文件；宿主树保持无界（约束只落在
  打包与解析边界）；
- [x] **M-1 非有限参数**：`voxel_size`/`origin_world`/`background`/`rotation` 构造期
  `np.isfinite` 校验（NaN/Inf 此前绕过正数守卫，Inf 使 world_to_index 塌缩成常数）；
- [x] **M-2 线格式枚举白名单**：未知 type/class 码一律 ValueError（此前 class 码越界
  抛 IndexError、未知 type 码被静默解释成 float64）；`iter_leaves` 复用同一校验并补
  `leaf_log2` 范围检查（顺带收窄 L-2 的双路径漂移）；
- [x] **M-3 网格名混淆**：`.vdb` 描述符名与元数据名不一致即拒绝（"取 A 得 B"）；
  `write_vdb` 拒绝名字含 U+001E（OpenVDB 重名分隔符，roundtrip 按名检索会 KeyError）；
- [x] 回归：全套 106 项 + 新增安全测试 17 项全绿；quadrants 内核守卫经
  CPU 后端编译验证（注意：qd.func 内不支持非 static 分支早退，守卫写成三元表达式）。

## v1.3 — 表面算子(marching cubes / offset / mesh→SDF / metadata) ✅ 1.3.0

- [x] `level_set_to_mesh`:marching cubes(256 情形查找表,向量化逐叶推进,跨立方体棱边
  去重 + 等值点精确过角时的顶点焊接);单位球体积误差 0.095%,法向 100% 朝外,
  23k 三角 0.24s;
- [x] `mesh_to_level_set` / `dilated_mesh_shell`:三角形网格 → 窄带 level set
  (分块点到三角形精确距离 + 候选对收集向量化;+X 射线奇偶符号;bin 空间哈希加速);
- [x] `offset`:窄带 upwind(Godunov)等距偏移,OpenVDB `LevelSetFilter::offset` 的语义
  等价物;符号约定正值外扩。±0.1 偏移实测误差 0.25~0.31 体素。**精度依赖 reinit 的
  缺席仍是边界**:反复大偏移会偏离真距离场(与上游 OpenVDB 版同契约,文档化);
- [x] `.vdb` grid metadata 往返:`grid.metadata` dict(string/bool/int/float/vec3d),
  保留键(name/class/file_*)拒绝写入;
- [x] `requires-python` 放宽至 <3.15(3.14 就绪;cp314t 还受 quadrants wheel 制约)。

## v1.2 — 三短板(常数区压缩 / 流式 / reinit) ✅ 1.2.0(reinit 顺延)

- [x] 常数区压缩(OpenVDB tile 的叶级等价):`.xuvdb` v3 常数叶编码,整叶同值坍缩为
  一个值 + 掩码位——均匀填充网格实测 **33.9KB → 224B(151x)**;内存侧 `VdbGrid.compress()`
  (inactive 归零背景 + 丢弃纯背景叶,有损、显式调用);
- [x] out-of-core / slab 流式:`iter_leaves`(惰性逐叶,不建树)+ `iter_slabs`
  (按 slab 产出稠密片,峰值内存 = 单 slab + 单叶;axis=0 随流冲刷);
- [→顺延] 窄带 SDF 重整(reinit):**实验后顺延**。Sussman PDE 方案(带锚定、掩码限制)
  会把本已精确的场带坏(单球表面 |sdf| 0.0024 → 0.056,200 迭代不收敛);26 邻域
  chamfer-Dijkstra 方案在凹缝处比 min-of-SDF 原场更差((9,4,2) 处 −0.10 → −0.15,真值
  ≈ −0.04;根因:界面亚体素信息在体素化时已丢失,网格重整法的固有边界)。正确方案
  需要图元感知的局部精确重算,属 2.0+ 的设计变更。测量数据在 1.2.0 开发提交中。

## v1.3.2 — slab 算子分块化（十亿级内存封顶）✅ 1.3.2

- [x] `offset` / `smooth` 的 dense slab 按 `max_slab_voxels`(默认 64M,~2.5GB f64
  临时区)沿最长轴分块:锁定步进(每块抽取上一步全局态 +1 体素 halo、只写回内部),
  结果与整块一致(smooth 逐位相同,offset 在 ±0.1 世界单位内 atol 2e-3 实测);
- [x] 共享工具 `slab.py`:`tile_ranges`(预算+最小内部宽度约束)与 `write_region`
  (按叶写回内部、不动 halo);offset 块内宽 ≥ |distance|+band+3,保证外推数据自足;
- [x] 实测:100M active(11.8 万叶)整块 offset 在本机 OOM(单数组 6.8GB),分块后
  通过（200M active、23.7 万叶、本机 32GB 内存，offset +2 体素 3034s，active 200.6M→210.9M）——十亿级验证从"需 256-512GB"降到"64-128GB 工作站可跑全算子";
- [x] 新测试 `test_slab_chunking.py`:tile 预算/最小宽度、smooth 逐位等价、
  offset 膨胀/收缩等价、参数校验(6 项)。

## v1.3.1 — 平滑算子 smooth ✅ 1.3.1

- [x] `smooth(grid, iterations, alpha)`:OpenVDB `LevelSetFilter::smooth` 的等价物——
  窄带 SDF 值的离散 Laplacian Jacobi 松弛(``phi <- phi + alpha*(邻域均值-phi)``,
  0 < alpha <= 1 凸组合无条件稳定)。只在活动 bbox 内松弛,盒外体素/halo/远场
  Dirichlet 冻结 ⇒ 多次调用与单次多迭代严格等价(链式调用只受 float32 写回
  舍入影响);active 掩码、叶结构、band 宽度全部不动,无重打带、无拓扑变化。
  边界语义与 OpenVDB"只更新 active 值"一致;
- [x] 实现沿用 offset 的"密集 slab + 逐叶写回"范式:slab = 活动 bbox + 1 体素
  halo,双缓冲 Jacobi,写回只覆盖活动 bbox(⚠️ 不调 `prune()`——stamp 产生的
  带外值叶要保留,prune 会按"无 active 体素"丢弃它们,破坏链式等价性);
- [x] 校验:isotropic voxel_size、轴对齐、标量网格;vec 网格 / 旋转网格 / alpha
  越界显式报错;ShapeKernel 移植层的 `voxSmoothen` 由此从
  `NotImplementedError` 桩变为完整功能(门面 `smoothen()` 优先委派本算子)。

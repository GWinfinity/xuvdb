# XUVDB 太虚 — 基于 quadrants 的稀疏体素格式（可编辑 · 与 OpenVDB 互转）

仓库:[AtomGit](https://atomgit.com/allan_/xuvdb)(主) · [GitHub 镜像](https://github.com/GWinfinity/xuvdb) ·
[PyPI](https://pypi.org/project/xuvdb/) · [性能基线](https://github.com/GWinfinity/xuvdb/blob/main/BENCHMARKS.md)

> 「不游乎太虚。」——《庄子·知北游》
> 「太虚无形，气之本体。」——张载《正蒙·太和》

`xuvdb` 取「太虚」之音：这是中文里对 VDB「无界而稀疏的索引域」最准确的翻译——无界
（root 哈希域不设上限），无形（未分配的空间没有形态，采样即得背景值）。

`xuvdb` 基于 quadrants 内核：**自己的 `.xuvdb` 格式**（可修改、可在内核里写值），
以及**与 OpenVDB 的双向互转**（原生 `.vdb` 文件读写，无需安装 OpenVDB；有 `pyopenvdb`
绑定时还能内存级互转）。

## 命名规范（三层名字，各归其位）

| 层 | 名字 | 规则 |
|---|---|---|
| 项目名 | `xuvdb` | 拼音；不用 Open 前缀（ASWF 语境下暗示基金会血统） |
| 命名空间 / 扩展名 | `xuvdb` / `.xuvdb` | API 标识符一律英文：`xuvdb.prune()`、`xuvdb.VdbGrid`，绝不是 `xuvdb.sunyi()`。道家词只活在概念层（文档题词、日志、可视化标签） |
| 互转文件 | `.vdb` | **自有格式绝不写 `.vdb` 后缀**（Houdini／Blender／Cycles／Arnold 按扩展名当 OpenVDB 解析，格式不符时静默出错或崩溃，极难定位；`save()` 对 `.vdb` 路径直接拒绝）。要互通就单独导出：`write_vdb()` 产出真正的 OpenVDB 流 |

## 为什么是它

|  | OpenVDB | NanoVDB | XUVDB |
|---|---|---|---|
| 结构 | 5 层 B+树，CPU C++ | 同构只读缓冲，GPU | 两级：dict 叶根 + 稠密叶块 |
| 可修改 | ✅（CPU） | ❌ GPU 只读 | ✅ Python 端任意结构编辑；**内核端可写值** |
| 值缓冲 GPU 可写 | ❌ | ❌ | ✅ `write_voxels` 拓扑固定就地写 |
| 端到端可微 | ❌ | ❌ | 采样对叶值/坐标有解析梯度（`torch_bridge.sample_t`）；稀疏卷积等训练算子仍无 |
| Python 依赖 | pyopenvdb（需自行构建） | C++ 工具链 | numpy + quadrants（quadrants 是带 JIT 的完整工具链：首次调用现场编译内核；GPU 需驱动，无 GPU 时 CPU 后端可用——射线慢 ~450 倍、采样只慢 ~4 倍，见 BENCHMARKS.md） |

头条数字（**300 粒子网格**，即「千级粒子」验证规模；CPU 后端）：内核三线性
**~39M 点/s**，内核批量射线 **~156k 射线/s**（宿主逐点 0.3k/s）。
**内存的诚实账**：本基准网格活动体素占包围盒的 55%，这个占用率下稀疏叶表（60 叶 × 16³
稠密缓冲 ≈ 1.0MB）反而比包围盒稠密（237KB）**大**——稀疏叶格式的内存优势只在**低占用率**
（活动占比远低于 1/叶体素数 ≈ 0.4%）或域远大于包围盒时成立；落盘后经 active-mask + Blosc
压缩可回到 ~120KB 量级。完整表、复现脚本与规模声明见 BENCHMARKS.md 与「版本与稳定性」一节。

定位不是替换任何求解器，而是补一个**可编辑的稀疏空间表示层**。

## 安装

独立 Python 包（src 布局，`import xuvdb` 即用）：

```bash
uv pip install .            # 或 pip install .
uv pip install -e ".[test]" # 开发模式 + pytest
```

可选 extras：`[openvdb]`（pyopenvdb 内存级互转）、`[torch]`（torch 桥：张量往返 + 可微采样）、
`[genesis]`（运行引擎侧示例需要 genesis-world）、`[test]`（pytest）。

## 快速上手

```python
import numpy as np
import xuvdb

# 1) 编辑：窄带 level set 球（体素 0.05 世界单位；band=3 是 3 个体素；leaf_log2=4 即 16^3 叶）
#    动词语义：SDF stamp = 与现有场取 min（并集）；fog stamp（给 value=）= 覆盖；scatter_* = 累加
grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                     name="shield", grid_class="level set")
grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
grid.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)          # min-union：并入第二个球（= csg union）
grid.fill_box((-2, -2, -2), (2, 2, 2), 0.0)              # 任意稠密填充（示例）
grid.prune()

# 2) 自有格式落盘 / 读回（多网格、f32/f64/vec3）
xuvdb.save("scene.xuvdb", [grid], compress=True)   # v2: zlib 载荷 + CRC32 校验和（默认恒校验）
grids = xuvdb.load("scene.xuvdb")

# 3) 与 OpenVDB 互通（显式导出：真正的 OpenVDB 流，Houdini/Blender 直接打开）
xuvdb.write_vdb("scene.vdb", [grid], blosc=True)   # blosc=True 输出 Blosc 压缩值块（需 blosc 包）
back = xuvdb.read_vdb("scene.vdb", grid_name="shield")

# 4) 内核采样 / 写值（等同 Warp example_nvdb 的用法，但值可写）
vol = xuvdb.GpuVolume(grid)
pts = np.array([[0.3, 0.2, 0.36]], dtype=np.float32)
d = vol.sample(pts, linear=True)          # SDF 距离（order=0/1/2 也提供三次 B 样条）
q = vol.sample(pts, order=2)              # 三次采样：C1 平滑（宿主侧 grid.sample_quadratic 同款）
g = grid.sample_gradient(pts)             # 中心差分梯度（宿主）；stencil7/19_batch 供模板算子
stats = vol.reduce()                      # active 求和/极值/计数（一次 kernel launch）
n = vol.sdf_normal(pts)                   # 有限差分表面法向
vol.write_voxels(pts, np.array([-0.01], np.float32))
vol.sync_to_host(refresh_mask=True)   # 内核写进 inactive 体素的值默认不进 write_vdb/active_*；
                                      # refresh_mask 重算掩码使其可见（sample/.xuvdb/to_dense 不受影响）

# 5) 粒子 ⇄ 体积（液体/油）
drops = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
fog = xuvdb.VdbGrid(voxel_size=0.05, name="liquid", grid_class="fog volume")
fog.scatter_particles(drops, h=4 * 0.05, weights=1.0)     # SPH cubic 核密度 splat
surf = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, grid_class="level set")
surf.union_spheres(drops, radius=0.03)                    # particle level set 表面代理

# 6) torch 桥（互通不做框架：张量往返 + 可微采样；pip install 'xuvdb[torch]'）
from xuvdb.torch_bridge import grid_to_tensors, sample_t, tensors_to_grid
t = grid_to_tensors(grid, device="cuda")            # keys/values/active -> tensors
t["values"].requires_grad_(True)                    # 叶值张量可训练
pts_t = torch.rand(64, 3, device="cuda", requires_grad=True)
field = sample_t(t, pts_t, grid.voxel_size, grid.origin_world,
                 grid.leaf_log2, grid.background)   # 前向 == sample_linear
field.sum().backward()                              # 解析梯度: dL/dvalues + dL/dpoints
back = tensors_to_grid(t, grid.voxel_size, grid.origin_world,
                       grid.leaf_log2, grid.background, name=grid.name)

# 7) 多网格批量打包（GridBatch 式：一次 launch 采样所有网格）
batch = xuvdb.gpu.VolumeBatch([grid_a, grid_b])     # 各自的变换/叶尺寸都可不同
vals = batch.sample(points, volume_ids)             # (m,3) 点 + 每点所属网格 id

# 8) DDA 射线（空叶块按块跳过，交叉点二分细化到亚体素）
t, point, value = xuvdb.ray_surface_hit(grid, (0.3, 0.2, 2.0), (0, 0, -1))
# 内核端批量射线（一次 launch 跑全部射线，CPU/CUDA/Vulkan 可用）：
hits = vol.ray_surface_hit(origins, dirs)   # vol = xuvdb.GpuVolume(grid)，逐射线 (t, point, value) 或 None
```

稠密场 ⇄ 稀疏网格：

```python
# 稠密 numpy SDF → 稀疏
sparse = xuvdb.VdbGrid.from_dense(dense_sdf, origin=ijk_min, voxel_size=h,
                                  background=band_h, grid_class="level set")
dense, ijk_min = sparse.to_dense()        # 反向：渲染器 / 求解器输入
```

## 格式

### `.xuvdb`（自有格式 v3，小端）

```
"XUVDB" | u8 version=3 | u8 flags(bit0=zlib 载荷, bit1=CRC32 尾注) | u32 n_grids
payload（bit0 时为 zlib 压缩）:
per grid:
  str name | u8 type(0=f32,1=f64,2=vec3f,3=f16) | u8 leaf_log2 | u8 class
  | u8 grid_flags(bit0=后随旋转阵) | u8 rsv
  f64[3] voxel_size | f64[3] origin_world | [f64[9] rotation] | background
  u32 n_leaves
  per leaf（按叶原点排序）: i32[3] origin | u8 kind
    kind=0（稠密）: u64[dim³/64] active mask | 值稠密数组
    kind=1（常数, v3 tile 等价物）: u8 active | 一个值   ← 整块同值坍缩, ~400x 磁盘压缩
trailer（bit1）: u32 CRC32（对未压缩载荷计算）
```

- v1/v2 文件永久可读；值缓冲是**全量稠密叶**（非掩码过滤），
  内核写进 inactive 体素的值在 save/load 后保留。
- 叶内线性序 `n = x·dim² + y·dim + z`（z 最快），**与 OpenVDB leaf 序一致**，互转零转置。
- 变换：`world = R @ (index · voxel_size) + origin_world`，体素中心在整数索引处；
  R 缺省为单位阵。
- 流式消费：`iter_leaves`（惰性逐叶）/ `iter_slabs`（按 slab 产稠密片）——out-of-core 原语。

### `.vdb`（OpenVDB 官方流格式，仅显式导出用）

字节布局逐一对照 OpenVDB 源码实现（`io/Archive.cc`、`GridDescriptor.cc`、`Compression.h`、
`tree/*.h`、`math/Maps.h`、`Metadata.h`）：

- 头 57B：`int64 magic 0x56444220`、u32 文件版本、u32 库主/次版本、u8 offsets 标志、36 字符 UUID；
- 文件级元数据表 → i32 网格数 → **描述符与网格流交错**（描述符、i64×3 偏移、网格流、下一描述符…）；
- 网格流：u32 压缩标志 → 元数据表（name/class/file_* 统计）→ 变换 → 树（`i32 buffer_count`、
  root 背景 + tiles + 子节点）；
- 变换是**按类型字符串分发的表**（`math/Maps.h` 各 `write()` 的布局）：ScaleTranslate 家族
  6×Vec3d、Scale/UniformScaleMap 5×Vec3d（OpenVDB 等向体素的默认产物即 UniformScaleMap）、
  TranslationMap 1×Vec3d、Affine/UnitaryMap 一个 Mat4d，NonlinearFrustumMap 显式拒绝；
- 树：root → InternalNode(5)（32³ 桌、512×u64 双掩码、值表）→ InternalNode(4)（64 项）→
  LeafNode(8³)（拓扑段只有值掩码，origin 由树路径隐含；缓冲段掩码重写一遍 + 值块）；
- 值块：`io::writeCompressedValues` 语义 —— 1 字节 metadata（0=惰性值全为 +bg、1=-bg、
  2/4/5=带 1~2 个惰性值/选择掩码、6=全量数组）+ 值（按 ACTIVE_MASK 只存 active）。

写侧：文件版本固定 **224**——本实现不产出 half 网格与 root tile，224（`_MULTIPASS_IO`）
的读者面最广（上游 225 起才有 half-grid 文件项）；值块压缩可选 `COMPRESS_ACTIVE_MASK`
（默认，无 zip/blosc）或叠加 `COMPRESS_BLOSC`（`blosc=True`，需 `pip install blosc`，帧参数
对齐 OpenVDB `bloscCompress`：clevel 9 + byte shuffle）。带旋转的网格写 AffineMap，轴对齐写
ScaleTranslateMap；任何 OpenVDB ≥ 9 可读。
读侧：`COMPRESS_NONE` / `COMPRESS_ZIP`（stdlib zlib）/ `COMPRESS_BLOSC`（需 blosc 包）/
`COMPRESS_ACTIVE_MASK` / `_HalfFloat` 网格；root/internode 活动 tile 物化为稠密叶
（受 `max_tile_voxels` 上限保护）。

## 已知边界

- **三个短板的现状**（1.2.0）：常数区压缩已补（`.xuvdb` v3 常数叶编码 + `compress()` 内存
  压缩）；out-of-core 以 `iter_leaves` / `iter_slabs` 流式原语提供（不是完整分页引擎）；
  窄带 SDF 重整（reinit）**未提供**——Sussman PDE 与 chamfer-Dijkstra 两种实验方案都在
  凹缝处过不了质量门槛（根因：界面亚体素信息在体素化时已丢失），需要图元感知的局部
  精确重算，顺延至 2.0+（测量数据见 ROADMAP）。
- `GpuVolume` 只支持 f32 标量网格（f16/f64/vec3 为宿主与格式层类型）；写入只改值不改拓扑、
  不动 active 掩码（掩码是宿主侧状态）。**掩码语义**：内核写进 inactive 体素的值能通过
  `sample*`、`.xuvdb` 存取、`to_dense`、射线看到（叶是稠密缓冲），但 **`write_vdb` 与
  `active_*`/`reduce` 按掩码过滤**——`sync_to_host(refresh_mask=True)` 可把非背景值重标为
  active；结构性编辑后仍需重新打包。
- 内核仅支持轴对齐网格；带 `rotation` 的网格在宿主侧全功能（采样/射线/stamp/两种格式），
  交给 `GpuVolume` 会显式报错。
- `scatter_particles`／`union_spheres` 是 Python 循环 + 叶切片向量化：千级粒子适用，
  大规模生产需按叶批处理（未做）——这个量级下 `np.add.at` 的稠密散射也可能更快，本路径的
  价值在于结果直接在 GPU 上、免回传。`union_spheres` 的 min-of-spheres 距离在重叠粒子间
  的凹桥区是真实距离的上界（min 保 1-Lipschitz，并集外部精确）；两处已知偏差：**透镜重叠
  区**（点同在两球内部）取的是更深那颗，比联合体真实边界偏深；**中轴脊上梯度不连续**，
  法向会在脊线两侧跳变——做碰撞/渲染代理够用，做表面重建会出菱形接缝，精确表面请离线
  用正规表面重建精修。
- `.vdb` 读侧不支持：实例化网格（instance parent）、点云网格（PointDataGrid）、
  `5_4_3` 以外的树形、**NonlinearFrustumMap 变换**（显式报错）；其余变换类型
  （ScaleTranslate/Scale/UniformScale/Translation/Affine/Unitary）逐类型分发支持。
  Blosc 压缩块需要可选依赖 `pip install blosc`（OpenVDB 帧级语义）。
  写侧不产生 root tile（全部以叶表达）；half 网格写侧升格 f32（读侧 half 网格支持）。
- 与求解器自动微分的边界：XUVDB 提供的是**采样/写入原语**；把 VDB 值直接接入反传图需要
  包一层自定义求导规则（这正是 FastSweeping 等算子不可微的同一边界）。

## 许可证

Apache-2.0（与上游 quadrants、genesis-world 一致），见 [LICENSE](https://atomgit.com/allan_/xuvdb/blob/main/LICENSE)。

## 版本与稳定性(1.0.0 起)

- **semver**:主版本 = 破坏性变更,次版本 = 向后兼容的新功能,修订 = 修复;
- **`.xuvdb` 格式 v3 冻结**:v1/v2 永久可读;未来字段只增不改,版本号随破坏性变更递增;
- **公开 API** = 本 README 与 `xuvdb.__all__` 所列(`VdbGrid`/`Leaf`/`GpuVolume`/`VolumeBatch`/
  `save`/`load`/`write_vdb`/`read_vdb`/`to_openvdb`/`from_openvdb`/`ray_surface_hit`/
  `torch_bridge`/`init_runtime`);`xuvdb.kernels` 是内部实现,不承诺稳定;
  类型标注随包分发(`py.typed`)。
- **语义定案**:SDF `stamp_sphere` 按 min-union 复合(= `csg('union')`),fog stamp 覆盖写,
  `scatter_*` 累加——三组动词三种语义,详见快速上手第 1 节的对照注释;
- **参数单位**:`band`/`h` 以体素计,`background`/`radius` 以世界单位计,`leaf_log2` 是
  log2(叶维 = 2^leaf_log2);改名留给 2.0(如 `band_voxels`),1.x 只文档化不改名;
- **1.0 承诺范围**:承诺的是**格式稳定性与公开 API 稳定性,不承诺规模化性能**——
  千级粒子的 splat、单网格量级的采样是当前验证过的规模;
- 性能基线见 [BENCHMARKS.md](https://atomgit.com/allan_/xuvdb/blob/main/BENCHMARKS.md)
  (可复现脚本 `examples/bench_suite.py`)。

## 测试

```
pytest tests/ -q
```

覆盖：树编辑/CSG/稠密互转、粒子 splat（质量守恒、可加性、vec3 速度场、双核函数）、
`union_spheres`（表面/窄带/逐粒子半径/射线命中/与 stamp 复合）、`.xuvdb` 多网格多类型
往返、`.vdb` 头部字节与偏移校验、f32/f64/vec3/fog/level-set 往返、多叶尺寸重分块、
负坐标、惰性值压缩路径、`save()` 拒绝 `.vdb` 后缀、GPU 采样对齐宿主三线性、内核写值
往返、DDA 射线（含空块跳跃、内部出发、tmax 截断、变换偏移）。

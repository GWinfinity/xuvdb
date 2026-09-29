# XUVDB 太虚 — 基于 quadrants 的稀疏体素格式（可编辑 · 与 OpenVDB 互转）

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
| 可微分包差 | ❌ | ❌ | 值缓冲可被 quadrants 内核读写（拓扑固定） |
| Python 依赖 | pyopenvdb（需自行构建） | — | 仅 numpy + quadrants |

定位不是替换任何求解器，而是补一个**可编辑的稀疏空间表示层**。

## 安装

独立 Python 包（src 布局，`import xuvdb` 即用）：

```bash
uv pip install .            # 或 pip install .
uv pip install -e ".[test]" # 开发模式 + pytest
```

可选 extras：`[openvdb]`（pyopenvdb 内存级互转）、`[genesis]`（运行引擎侧示例需要 genesis-world）、
`[test]`（pytest）。

## 快速上手

```python
import numpy as np
import xuvdb

# 1) 编辑：窄带 level set 球（体素 0.05，带宽 3 体素）
grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                     name="shield", grid_class="level set")
grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
grid.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)          # CSG：并入第二个球
grid.fill_box((-2, -2, -2), (2, 2, 2), value=0.0)         # 任意稠密填充（示例）
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
vol.write_voxels(pts, np.array([-0.01], np.float32)); vol.sync_to_host()

# 5) 粒子 ⇄ 体积（液体/油）
drops = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
fog = xuvdb.VdbGrid(voxel_size=0.05, name="liquid", grid_class="fog volume")
fog.scatter_particles(drops, h=4 * 0.05, weights=1.0)     # SPH cubic 核密度 splat
surf = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, grid_class="level set")
surf.union_spheres(drops, radius=0.03)                    # particle level set 表面代理

# 6) DDA 射线（空叶块按块跳过，交叉点二分细化到亚体素）
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

### `.xuvdb`（自有格式，小端）

```
"XUVDB" | u8 version=1 | u8 flags | u16 n_grids
per grid:
  str name | u8 type(0=f32,1=f64,2=vec3f) | u8 leaf_log2 | u8 class | u8 rsv
  f64[3] voxel_size | f64[3] origin_world | background
  u32 n_leaves
  per leaf（按叶原点排序）: i32[3] origin | u64[dim³/64] active mask | 值稠密数组
```

- 叶内线性序 `n = x·dim² + y·dim + z`（z 最快），**与 OpenVDB leaf 序一致**，互转零转置。
- 叶块与 OpenVDB LeafNode 同为稠密缓冲：level set 内部体素的 `-background` 值在
  save/load 后保留。
- 变换约定与 OpenVDB 线性映射一致：`world = index · voxel_size + origin_world`，
  体素中心在整数索引处。

### `.vdb`（OpenVDB 官方流格式，仅显式导出用）

字节布局逐一对照 OpenVDB 源码实现（`io/Archive.cc`、`GridDescriptor.cc`、`Compression.h`、
`tree/*.h`、`math/Maps.h`、`Metadata.h`）：

- 头 57B：`int64 magic 0x56444220`、u32 文件版本、u32 库主/次版本、u8 offsets 标志、36 字符 UUID；
- 文件级元数据表 → i32 网格数 → **描述符与网格流交错**（描述符、i64×3 偏移、网格流、下一描述符…）；
- 网格流：u32 压缩标志 → 元数据表（name/class/file_* 统计）→ 变换（ScaleTranslate 家族
  = 类型字符串 + 6×Vec3d）→ 树（`i32 buffer_count`、root 背景 + tiles + 子节点）；
- 树：root → InternalNode(5)（32³ 桌、512×u64 双掩码、值表）→ InternalNode(4)（64 项）→
  LeafNode(8³)（拓扑段只有值掩码，origin 由树路径隐含；缓冲段掩码重写一遍 + 值块）；
- 值块：`io::writeCompressedValues` 语义 —— 1 字节 metadata（0=惰性值全为 +bg、1=-bg、
  2/4/5=带 1~2 个惰性值/选择掩码、6=全量数组）+ 值（按 ACTIVE_MASK 只存 active）。

写侧：文件版本 224、压缩 = `COMPRESS_ACTIVE_MASK`（无 zip/blosc），任何 OpenVDB ≥ 9 可读。
读侧：支持 `COMPRESS_NONE` / `COMPRESS_ZIP`（stdlib zlib）/ `COMPRESS_ACTIVE_MASK` /
`_HalfFloat` 网格；Blosc 抛出明确错误；root/internode 活动 tile 物化为稠密叶
（受 `max_tile_voxels` 上限保护）。

## 已知边界

- `GpuVolume` 只支持 f32 标量网格（f16/f64/vec3 为宿主与格式层类型）；写入只改值不改拓扑、
  不动 active 掩码（掩码是宿主侧状态）。结构性编辑后需重新打包。
- 内核仅支持轴对齐网格；带 `rotation` 的网格在宿主侧全功能（采样/射线/stamp/两种格式），
  交给 `GpuVolume` 会显式报错。
- `scatter_particles`／`union_spheres` 是 Python 循环 + 叶切片向量化：千级粒子适用，
  大规模生产需按叶批处理（未做）。`union_spheres` 的 min-of-spheres 距离在重叠粒子间
  的凹桥区是真实距离的上界（Lipschitz 精确），做碰撞/渲染代理足够，精确表面请离线
  用正规表面重建精修。
- `.vdb` 读侧不支持：实例化网格（instance parent）、点云网格（PointDataGrid）、
  `5_4_3` 以外的树形；Blosc 压缩块需要可选依赖 `pip install blosc`（OpenVDB 帧级语义）。
  写侧不产生 root tile（全部以叶表达）；half 网格写侧升格 f32（读侧 half 网格支持）。
- 与求解器自动微分的边界：XUVDB 提供的是**采样/写入原语**；把 VDB 值直接接入反传图需要
  包一层自定义求导规则（这正是 FastSweeping 等算子不可微的同一边界）。

## 许可证

Apache-2.0（与上游 quadrants、genesis-world 一致），见 [LICENSE](LICENSE)。

## 测试

```
pytest tests/ -q
```

覆盖：树编辑/CSG/稠密互转、粒子 splat（质量守恒、可加性、vec3 速度场、双核函数）、
`union_spheres`（表面/窄带/逐粒子半径/射线命中/与 stamp 复合）、`.xuvdb` 多网格多类型
往返、`.vdb` 头部字节与偏移校验、f32/f64/vec3/fog/level-set 往返、多叶尺寸重分块、
负坐标、惰性值压缩路径、`save()` 拒绝 `.vdb` 后缀、GPU 采样对齐宿主三线性、内核写值
往返、DDA 射线（含空块跳跃、内部出发、tmax 截断、变换偏移）。

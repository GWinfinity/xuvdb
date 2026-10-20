# XUVDB 性能基线

可复现脚本:`python examples/bench_suite.py [cuda]`(预热后计时,射线 host==kernel 奇偶校验内置)。
本页数字来自 1.0.0;改代码后请重跑并更新。

环境:Windows 11,i7-9750H(6C),GTX 1650 4GB,Python 3.12.13,numpy 2.5.3,quadrants 1.3.0。
网格:300 粒子 `union_spheres`,60 叶,32,783 活动 体素,体素 0.05。

## CPU 后端(quadrants x64)

| benchmark | time | rate |
|---|---|---|
| host `sample_linear`(逐点 Python) | 249.8 ms / 2k 点 | 8k 点/s |
| kernel `sample` order=0(nearest) | 1.0 ms / 100k 点 | **96,525k 点/s** |
| kernel `sample` order=1(trilinear) | 2.6 ms / 100k 点 | **38,763k 点/s** |
| kernel `sample` order=2(triquadratic) | 15.7 ms / 100k 点 | 6,387k 点/s |
| kernel `reduce`(active sum/min/max/count) | 0.5 ms | 186,081k 体素/s |
| host DDA(bbox 跳跃,逐射线 Python) | 2,903.5 ms / 1000 射线 | 0.3k 射线/s |
| kernel DDA(批量,一次 launch) | 6.4 ms / 1000 射线 | **156k 射线/s(~450x host)** |
| `VolumeBatch.sample`(2 网格,100k 点) | 3.4 ms | 29,432k 点/s |

## CUDA 后端(GTX 1650)

| benchmark | time | rate |
|---|---|---|
| kernel `sample` order=1 | 1.9 ms / 100k 点 | 52,471k 点/s |
| kernel `sample` order=2 | 10.1 ms / 100k 点 | 9,898k 点/s |
| kernel DDA(批量) | 45.8 ms / 1000 射线 | 22k 射线/s |
| `VolumeBatch.sample`(2 网格,100k 点) | 9.9 ms | 10,069k 点/s |

## 解读(诚实版)

- **内核 vs 宿主**是稳定的大头收益:采样 ~4,800x、射线 ~450x、reduce/gather 全 kernel 化。
- **CUDA 在这个小网格上不敌 CPU 后端**:60 叶/3 万体素的工作量撑不起 launch + 传输开销,
  且这批随机射线长距离 march。CPU 后端(SIMD 编译)在万级叶以下就是最优选择;CUDA 的
  优势区间在更大的网格与批量(0.2.0 时代同脚本在 1000 内向射线上测过 7 us/射线)。
- **与 NanoVDB 的对比是设计级的,不是逐项的**:同为"flat 缓冲 + key 有序 + 每 leaf 缓存
  active bbox 的层级跳跃",本库 DDA 与 NanoVDB ReadAccessor/DDA 同构;真正的 head-to-head
  需要 NanoVDB C++ 构建(本仓库未做,欢迎贡献基准)。定性上:小网格 CPU 后端即可逼近
  NanoVDB CPU 采样器的量级;内核可写值(`write_voxels`)是 NanoVDB 明确不提供的。


## 1.3.3 大规模基准(2026-10-20,AMD MI300X / NVIDIA A10)

测试对象:`pip install xuvdb` 1.3.3(自动拉取 quadrants 1.3.3,一次成功,无编译)。
主测试网格为球体 level set(dx=1/128):2,962 个 16³ 叶、123.5 万活跃体素、活跃数据
4.9 MB。内核测试预热 1 次(JIT 编译)后取 3 次重复的最小值;CPU / GPU 分别在独立进程
运行(quadrants 运行时每个进程单后端)。两台机器跑同一 `bench_xuvdb.py`,数据逐数字
可比。

### 环境

| 项目 | MI300X 实例 | A10 实例 |
|---|---|---|
| GPU | AMD Instinct MI300X(gfx942),192 GB HBM3(带宽 ~5.3 TB/s),ROCm 7.2.3 | NVIDIA A10(Ampere),24 GB GDDR6(600 GB/s),driver 550.54.15,CUDA 13.0 |
| CPU | 23 vCPU(KVM 虚机,宿主显示通用 "Intel Xeon") | Intel Xeon Platinum 8369B @ 2.90 GHz,8 vCPU |
| 内存 | 200 GB | 28 GB(测试峰值 RSS 16.9 GB) |
| OS / Python | Alibaba Cloud Linux 5.10.134 / 3.12.13 | Ubuntu 22.04.5 LTS / 3.12.13 |
| 关键依赖 | quadrants 1.3.3(LLVM 22.1.0 JIT),numpy 2.3.5 | quadrants 1.3.3,numpy 2.5.2 |
| 后端识别 | `arch=amdgpu` 自动识别 MI300X,无需 CUDA | `arch=cuda` 自动识别,无需额外配置 |

### 正确性(先于性能)

| 对照项 | MI300X 实例 | A10 实例 |
|---|---|---|
| 内核采样 CPU vs GPU(20 万点) | nearest/linear/quadratic max diff ≤ 3.8e-6 | nearest **逐位一致(0.0)**,linear ≤ 1.9e-6,quadratic ≤ 3.8e-6 |
| 内核 linear vs 宿主 `VdbGrid.sample_linear` | max diff 1.5e-8 | max diff 1.5e-8 |
| `sdf_normal` CPU vs GPU vs 宿主梯度(窄带内 ~1.5 万点) | 余弦相似度 min = 1.000000,模长 1.00000 | 余弦相似度 min ≥ 0.99999988,模长 1.000000 |
| DDA 光线求交 CPU vs GPU(4096 条) | 命中集合完全一致,t 差 ≤ 1e-15 | 命中集合 100% 一致,t 差 ≤ 8.9e-16 |
| `reduce` 跨后端 | sum=408.4361,count=1,235,034 | 完全相同 |
| 跨机器对照 | — | 采样 checksum(nearest/linear/quadratic/batch)与 MI300X **逐位相同**;ray hits 同为 9,944 |

注意(已在当前树修复):本次基准暴露 `sdf_normal` 的窄带限制——梯度 stencil 跨界处输出无
定义,CPU 与 GPU 方向可能恰好相反(A10 实测 27,683 个有效点中 237 个)。修复后的契约为:
带内中心差分不变;带边界 tap 出界自动转单侧差分;带外返回精确零向量 +
`return_valid` 有效掩码;`extend=k` 就近延拓法线场(详见 README「已知边界」)。

### 内核吞吐(800 万点采样,GPU vs 同机 CPU)

**MI300X 实例(GPU = amdgpu vs CPU = x64,23 vCPU)**:

| 测试项 | CPU | GPU (MI300X) | 加速比 |
|---|---|---|---|
| 采样·最近邻 | 244.8 Mpts/s | 719.0 Mpts/s | **2.9x** |
| 采样·三线性 | 45.4 Mpts/s | 529.5 Mpts/s | **11.7x** |
| 采样·三二次 B 样条 | 14.3 Mpts/s | 309.0 Mpts/s | **21.6x** |
| SDF 法线 | 8.0 Mpts/s | 151.4 Mpts/s | **19.0x** |
| VolumeBatch 跨 8 体积 | 64.9 Mpts/s | 399.5 Mpts/s | **6.2x** |
| 光线求交(262,144 条 DDA) | 1.71 Mrays/s | 3.37 Mrays/s | **2.0x** |
| 写入 400 万体素(含打包) | 40.7 ms | 29.6 ms | 1.4x |
| reduce(123.5 万活跃体素) | 1.0 ms | 5.0 ms | 0.2x(开销主导) |
| `GpuVolume` 打包 / `sync_to_host` 写回 | 11.7 / 16.8 ms | 12.4 / 16.4 ms | ≈1x(纯主机) |

**A10 实例(GPU = CUDA vs CPU = x64,8 vCPU)**:

| 测试项 | CPU | GPU (A10 CUDA) | 加速比 |
|---|---|---|---|
| 采样·最近邻 | 79.2 Mpts/s | 282.8 Mpts/s | **3.6x** |
| 采样·三线性 | 13.2 Mpts/s | 271.5 Mpts/s | **20.6x** |
| 采样·三二次 B 样条 | 4.2 Mpts/s | 229.4 Mpts/s | **55.2x** |
| SDF 法线 | 2.1 Mpts/s | 75.9 Mpts/s | **35.7x** |
| VolumeBatch 跨 8 体积 | 18.0 Mpts/s | 152.3 Mpts/s | **8.4x** |
| 光线求交(262,144 条 DDA) | 0.70 Mrays/s | 2.80 Mrays/s | **4.0x** |
| 写入 400 万体素(含打包) | 78.6 ms | 53.3 ms | 1.5x |
| reduce(123.5 万活跃体素) | 1.5 ms | 10.7 ms | 0.14x(开销主导) |
| `GpuVolume` 打包 / `sync_to_host` 写回 | 15.7 / 17.3 ms | 15.7 / 17.7 ms | ≈1x(纯主机) |

**跨机器 GPU 对比(A10 / MI300X)**:最近邻 39%、三线性 51%、三二次 74%、SDF 法线 50%、
VolumeBatch 38%、光线求交 83%。差距主要由显存带宽决定(600 GB/s vs ~5.3 TB/s):
算术强度越高(三二次 27 角点加权)差距越小,分支密集、带宽不敏感的 DDA 光线求交差距最小。

### 主机端重型管线(纯 NumPy,与计算后端无关,两后端耗时相同)

| 测试项 | 规模 | MI300X 实例 | A10 实例 |
|---|---|---|---|
| 网格→level set 烘焙 | 20,480 tris,dx=1/64 | 13.2 s | 15.3 s |
| 网格→level set 烘焙 | 81,920 tris,dx=1/128 | 67.4 s | 78.6 s |
| 网格→level set 烘焙 | 256,000 tris(环面结),dx=1/64 | 243.9 s | 279.5 s |
| `offset` 窄带偏移 d=0.06(≈8 体素),band=8 | 上述网格 | 97.4 s | 145.0 s |
| `smooth` 5 次迭代 | 同上 | 1.85 s | 2.60 s |
| `level_set_to_mesh`(marching cubes) | 产出 6,298,673 三角形 | 70.5 s | 85.9 s |
| 宿主纯 Python 光线参考 | — | 2.73 ms/条(内核快 ~1.6 万倍) | 3.37 ms/条 |

### 序列化 I/O(同一 123.5 万活跃体素网格)

| 格式 | 文件大小 | 写吞吐(MI300X / A10) | 读吞吐(MI300X / A10) |
|---|---|---|---|
| OpenVDB `.vdb`(zip) | 42.7 MB | 31.3 / 25.5 MB/s | 69.3 / 62.3 MB/s |
| 原生 `.xuvdb` | 50.1 MB | 314 / 313 MB/s | 472 / 457 MB/s |
| 原生 `.xuvdb` + zip | 16.7 MB | 11.5 / 12.1 MB/s | 66.4 / 73.6 MB/s |

### 结论与使用建议

1. **双平台开箱即用**:`pip install xuvdb` 后 MI300X 自动 `arch=amdgpu`、A10 自动
   `arch=cuda`,24 项测试全部跑通,内核结果与 CPU 后端逐位级一致,无需任何额外配置。
2. **GPU 加速与算子算术强度正相关**:三二次采样 21.6x/55.2x、SDF 法线 19.0x/35.7x、
   三线性 11.7x/20.6x、VolumeBatch 6.2x/8.4x、最近邻 2.9x/3.6x——插值越重收益越大;
   A10 因对照 CPU 仅 8 vCPU,相对加速比更高。
3. **绝对吞吐受 kernel 启动 + H2D/D2H 传输限制**(800 万点仅 ~130 MB 数据,
   0.015–0.026 s/调用):更大批量(≥5000 万点)或把多帧点云合并为单次调用可显著提升
   有效吞吐。
4. **光线求交加速比最低**(MI300X 2.0x):DDA 分支密集,且 `GpuVolume.ray_surface_hit`
   每次调用都在 Python 侧重建叶子 bbox(2,962 叶循环),建议缓存。
5. **reduce 小数据留主机**:123.5 万体素仅 4.9 MB,GPU 归约反而慢 5–7 倍
   (kernel 启动 + 往返开销主导)。
6. **主机管线是大规模场景主要瓶颈**(当前为单线程 Python+NumPy,GPU 不加速):
   烘焙 25.6 万三角形 244–280 s、offset 97–145 s。
7. **内存**:800 万点采样测试峰值 RSS ~17 GB(点云与暂存数组为主,网格活跃数据本身仅
   4.9 MB);28 GB 内存机器可完成但余量有限,更大批量建议放到 ≥64 GB 内存的机器。
8. **格式选择**:速度用原生 `.xuvdb`(比 `.vdb` 写快 ~10–12x、读快 ~7x);体积用
   zip(1/3 大小,写慢 ~26–27x)。

使用建议(实测踩坑):

- `VdbGrid.active_voxel_count` / `n_leaves` / `leaf_dim` 是 **property**,不是方法。
- `sdf_normal` 的窄带限制已修复:边界自动单侧差分、带外零向量 + `return_valid` 掩码、
  `extend=k` 法线场延拓(本次基准暴露的问题,修复后跨后端确定性有测试锁定;
  见 README「已知边界」与 tests/test_sdf_normal.py)。
- `GpuVolume` 拓扑在打包时冻结,结构性编辑后需重新打包;写回需 `sync_to_host` 才发布
  到主机树。
- 28 GB 内存机器跑 800 万点批量测试接近上限(峰值 16.9 GB),注意并发进程占内存。

测试产物:基准 `bench_xuvdb.py`(数据 `cpu.jsonl` / `gpu.jsonl` / `cuda_cpu.jsonl` /
`cuda_gpu.jsonl`);正确性 `correctness.py` + `*_samp*.npy` / `*_norm.npy` / `*_rays.npy`。

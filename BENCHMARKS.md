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

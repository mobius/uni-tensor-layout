# tensor-layouts × uni-framework 调研

> 文档时间: 2026-07-14 01:35:34  
> 源仓库:  
> - https://github.com/facebookresearch/tensor-layouts （维护迁出提示: https://github.com/jduprat/tensor-layouts.git）  
> - https://github.com/mobius/uni-framework  
> 本地对照: `/home/joey/Work/uni`

---

## 1. 调研目的

将 **CuTe 布局代数**（纯 Python 可学、可推演、可可视化）引入本机 **Phi + VE + Host** 异构栈，形成「逻辑坐标 → 内存偏移 → 设备分片/向量化」统一描述，并与 uni 的设备发现、NUMA、功率、DAG 调度衔接。

工作区命名 `uni-cute-tensor` 暗示主路径是 **CPU/加速卡异构上的 CuTe 式张量布局**，而非 NVIDIA-only 内核。

---

## 2. tensor-layouts 机制摘要

### 2.1 定位

- **纯 Python** 实现 NVIDIA CuTe layout algebra  
- **No GPU required** — 适合本机无 NVIDIA 卡的现状  
- 版本（调研时）: 0.3.2；`requires-python >= 3.10`；运行时依赖为空（viz/test 为 optional）  
- License: MIT  

### 2.2 核心抽象

| 概念 | 定义 | 本机落地含义 |
|------|------|----------------|
| `Layout(shape, stride)` | 逻辑坐标 → 偏移：`offset = Σ coord_i * stride_i` | 描述 Host/Phi/VE 缓冲中的 tile 形状与跨度 |
| 层次 shape | 嵌套 tuple 表达多级 tile | VE 向量长度 × 核数 × 卡数；Phi 线程 × 核；Host SIMD × cache tile |
| `compose(A, B)` | `C(i)=A(B(i))` | 线程/向量映射再映射到全局缓冲 |
| `complement(L)` | 填补 codomain 空隙 | 与 bank/lane 互补布局、打包空隙分析 |
| `logical_divide` / `product` | 分块 / 复制 | 多设备切分、batch 复制 |
| `Swizzle` | XOR 置换（共享存储 bank 冲突） | GPU 语义为主；可类比到 cache/set 冲突或 VE 访存对齐 |
| `ComposedLayout` | 非线性/带 offset 的精确组合 | 切片、swizzle 链式保持 exact |
| MMA Atoms | 厂商矩阵乘「原子」的 thread-value 布局 | 本机需 **自定义** Host/VE/Phi atoms |

### 2.3 代数四则（教学与工程共用）

1. **compose** — 索引路径组合  
2. **complement** — 未覆盖偏移的补布局  
3. **logical_divide** — 按 tiler 分解 tile/rest  
4. **logical_product** — 按模式复制  

辅助: `flatten`, `coalesce`, `upcast`/`downcast`（bit↔element）, 逆布局, `Tile`, `Tensor`（带存储校验）。

### 2.4 现有 Atoms 与本机关系

| 包 | 硬件目标 | 本机 |
|----|----------|------|
| `atoms_nv` | SM70–SM120 Tensor Core | 仅符号 |
| `atoms_amd` | CDNA1–3 MFMA | 仅符号 |
| `atoms_xe` | Xe-HPC/HPG DPAS | 仅符号 |
| `atoms_amx` | AMX tdp* | 仅符号（CPU 无 AMX） |
| **待建** `atoms_host` | AVX-512 / VNNI | **主路径** |
| **待建** `atoms_ve` | VE 向量寄存器 / NLC GEMM tile | **主路径** |
| **待建** `atoms_phi` | KNC IMCI 512-bit | **主路径** |

### 2.5 可视化与测试

- `tensor_layouts.viz`: matplotlib 绘制 layout / swizzle / MMA  
- `pytest` + 可选 NVIDIA/AMD oracle  
- 本机 Phase 1 应先跑 **无 oracle 的核心测试**，确认代数正确性  

### 2.6 与 CUTLASS/CuTe 的差异注意点

- `ComposedLayout(outer, inner, offset=k)`：**offset 仅关键字参数**（与 CuTe C++/pycute 位置不同，防静默错位）  
- 库倾向 **exactness** 而非不安全归一化  
- 教学实现，性能不是目标；生产内核仍用 ncc/ICC/NLC  

---

## 3. uni-framework 机制摘要

### 3.1 定位

在 **同一台 ESC4000 G4** 上协同：

- Host: 2× Gold 6252  
- 1× Phi 7120P (KNC)  
- 3× NEC VE 1.0  

目标: 最大化互补特征（VE 稠密/带宽，Phi 不规则 x86，Host 调度）。

### 3.2 已完成阶段（上游/本地 uni）

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 硬件验证 | ✅ |
| 1 | uv / ncc / ICC 软件栈 | ✅ |
| 2 | 调度层 7 模块 | ✅ |
| 3 | TC-001~006 基准 | ✅（部分 ⚠️） |
| 4 | SpMV / dataprep / MC | ✅ |

### 3.3 调度层模块（可复用）

```
TaskGraph ── DeviceMgr / NUMABinder / PowerCap
                │
     PhiRunner (ssh/scp) · VERunner (ve_exec) · MPIRunner
```

关键文件（本地 `/home/joey/Work/uni/src/scheduler/`）:

- `devices.py` — 发现 Phi + 3×VE  
- `phi.py` / `ve.py` — 编译与执行  
- `numa.py` — NUMA 亲和  
- `power.py` — 功率封顶  
- `task_graph.py` — asyncio DAG  
- `profiler.py` — 预估 vs 实测  

### 3.4 核心策略（必须继承）

1. **PCIe 最小化** — 数据上卡后闭环  
2. **任务特征匹配** — 稠密→VE，不规则→Phi  
3. **Python 调度** — 非统一源码编译  
4. **uv 优先** — 不污染全局  
5. **Phi I/O via scp**  

### 3.5 已知瓶颈（布局方案必须显式建模）

| 瓶颈 | 数量级 | 布局侧对策 |
|------|--------|------------|
| PCIe vs 卡内带宽 | ~280:1 | `logical_divide` 出 **device-resident tile**；避免小粒度往返 |
| Host 内存 64GB | < 加速器合计 160GB | 流式 layout pipeline |
| 功率 1600W | 满载超标 | atom/任务级 power tag |
| Phi 启动开销 | TC-003 标注严重 | 批量 layout 任务，减少 launch 次数 |
| 三套编译器 | ICC / ncc / gcc | 布局元数据 JSON/codegen，分后端 emit |

---

## 4. 交叉分析：如何「根据 tensor-layouts 机制」用在本机

### 4.1 映射表

| CuTe / tensor-layouts | Host AVX-512 | Phi KNC | VE 1.0 |
|----------------------|--------------|---------|--------|
| 逻辑坐标 (i,j,k) | 矩阵下标 / 批维 | 同左 | 同左 |
| 向量宽度 atom | 8×FP64 / 16×FP32 | 8×FP64 IMCI | 长向量（实现时按 ncc 文档固定 VL） |
| 线程/核 mode | OpenMP 线程 | 61 核 × SMT | 8 核 × 3 卡 |
| Swizzle | 可选 cache-set 启发式 | 同 | HBM bank 启发式（研究项） |
| Tiled MMA | 伪 MMA（FMA 累加 tile） | 同 | NLC DGEMM 外层 tile 描述 |
| complement | 填充/对齐到 cache line | 同 | 对齐到向量长度 |

### 4.2 架构层分工

```
┌─────────────────────────────────────────────────────────┐
│  Host Python (uv env)                                     │
│  tensor-layouts 代数 + 自定义 atoms + viz + codegen 计划   │
└───────────────────────────┬─────────────────────────────┘
                            │ layout 描述 / 分片计划
┌───────────────────────────▼─────────────────────────────┐
│  uni-style scheduler（可 submodule / 路径引用 / 轻量复刻）  │
│  devices · numa · power · task_graph                      │
└───────┬─────────────────────┬───────────────────┬───────┘
        │                     │                   │
   Host kernel           Phi kernel           VE kernel
   (gcc OpenMP+AVX)      (ICC -mmic)           (ncc + NLC)
```

### 4.3 不做什么（范围控制）

- 不重写 CUTLASS C++  
- 不假设 CUDA runtime  
- 不把 AMX/Xe atom 当作本机性能路径  
- 不污染系统 Python  
- 首期不做完整自动 kernel 生成器（先手工内核 + layout 参数驱动）  

---

## 5. 依赖与许可

| 组件 | 许可 | 使用方式 |
|------|------|----------|
| tensor-layouts | MIT | uv 安装 / 或 vendor 子模块（注明版权头） |
| uni-framework | 见上游仓库 | 复用设计与可选代码引用；路径敏感信息脱敏 |
| NLC / MPSS / ICC | 厂商许可 | 本机已装；文档不记录 license key |

---

## 6. 术语与扩展阅读

详见 `docs/glossary.md`。  
关键论文/文档：

- CuTe Layout Representation and Algebra — arXiv:2603.02298  
- Categorical Foundations for CuTe Layouts — arXiv:2601.05972  
- CUTLASS CuTe quickstart（NVIDIA）  
- uni `docs/research/20260601_090918_heterogeneous_system_analysis.md`  

---

## 7. 调研结论

1. **机制可迁移**: Layout 代数与硬件无关；MMA atoms 可按 ISA 扩展。  
2. **本机最优切入点**: Host 上跑 tensor-layouts + 为 VE/Phi/AVX-512 定义 atom 与分片策略，再挂 uni 调度。  
3. **环境门禁已过**: 见同目录 `20260714_013534_hw_env_feasibility.md`。  
4. **下一步**: 架构文档 + 分阶段实施计划（`docs/architecture/`, `docs/plan/`）。

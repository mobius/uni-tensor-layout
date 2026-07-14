# 术语表（Glossary）

> 维护约定: 交互中出现的新术语在此追加；已有词条可补充「本仓库用法」。  
> 首次建立: 2026-07-14

---

## A–C

### Atom（MMA Atom / 计算原子）

在 CuTe / CUTLASS 语境下，指**一次硬件矩阵乘加指令（或等价微内核）所对应的线程与数值布局**。`tensor-layouts` 用 Python 对象描述其 `shape_mnk` 与 A/B/C 的 thread-value layout。  
**本仓库**: 扩展 Host AVX-512、Phi KNC、VE/NLC 三类 atom，不要求与 NVIDIA 指令二进制兼容。

### AVX-512

Intel 512-bit SIMD 指令扩展。Gold 6252（Cascade Lake）支持 F/DQ/CD/BW/VL 等子集。  
**本仓库**: Host 侧向量化与 atom 对齐的主要 ISA。

### AVX512_VNNI

Vector Neural Network Instructions，面向 INT8 等点积加速的 AVX-512 扩展。本机 CPU flags 含 `avx512_vnni`。  
**本仓库**: 可选低精度路径；v1 以 FP64 为主。

### AMX（Advanced Matrix Extensions）

Intel 后续平台（如 Sapphire Rapids）上的矩阵寄存器与 `tdp*` 指令。  
**本机**: **不支持**。`atoms_amx` 仅教学对照。

### AVEO

NEC 提供的 Host 侧 VE 卸载/执行相关运行时接口（VE Offloading）。  
**本仓库**: 数据面可与 uni 的 VE 执行路径对齐。

### Cascade Lake

Intel Xeon 可扩展处理器一代（本机 Gold 6252）。有 AVX-512，**无 AMX**。

### Codomain / Cosize

布局映射的**像空间**范围；`cosize(L)` 常为最大偏移+1（对 ComposedLayout 语义见上游文档）。  
用于判断缓冲分配下界。

### ComposedLayout

`tensor-layouts` 中承载 **非纯仿射** 组合的类型，典型为 `Swizzle ∘ Layout` 或带内部 `offset` 的精确组合。  
构造注意: `offset` 为**仅关键字参数**，与 CuTe C++ 位置参数顺序不同。

### compose

布局代数运算: \(C(i)=A(B(i))\)。B 决定访问 A 的哪些点及顺序。

### complement

给定布局 L，构造在指定 bound 内「补齐」未被 L 覆盖偏移的布局，使与 L 一起覆盖地址空间。

### CuTe

CUTLASS 中的布局与张量核心抽象（C++）。描述逻辑坐标到内存偏移，以及 tiling/MMA 组合。  
**tensor-layouts** 是其纯 Python 教学/推演实现。

### CUTLASS

NVIDIA 的 CUDA 模板库，面向高性能 GEMM 等。本机无 GPU 时不作运行时依赖。

---

## D–L

### DAG 任务图（TaskGraph）

有向无环图描述任务依赖；uni 用 asyncio 做拓扑调度，并可挂功率约束。  
**本仓库**: 经 bridge 复用。

### DGEMM

双精度（FP64）通用矩阵乘 \(C=\alpha AB+\beta C\)。VE 上优先 **NLC** 库实现。

### Hierarchical Layout（层次布局）

shape/stride 为**嵌套元组**，表达多级 tiling（如向量 × 核 × 设备）。

### HBM2

高带宽内存。NEC VE 1.0 单卡约 48GB HBM2，带宽远高于 Host DDR4 与 PCIe。

### IMCI

Intel Many Core Instructions，Knights Corner（Phi 7120P）上的 512-bit 向量 ISA（与后来 AVX-512 相关但不相同）。  
**本仓库**: Phi atom 的向量宽度对齐对象。

### KNC（Knights Corner）

第一代 Xeon Phi 架构（本机 7120P）。

### Layout

\((shape, stride)\) 定义的**坐标→偏移**函数。核心公式:  
`offset = sum(coord_i * stride_i)`（层次结构下按 CuTe 规则递归）。

### logical_divide

将布局按 tiler 分解为 tile 与 rest，是多级分块与多设备切分的核心操作。

### logical_product

按另一布局描述的位置**复制**某布局模式（tiling 的对偶操作之一）。

---

## M–P

### micnativeloadex

在 Host 上加载并运行 MIC native 二进制的工具。Phi 无共享文件系统时配合 scp 使用。

### MMA

Matrix Multiply-Accumulate，矩阵乘加。GPU Tensor Core / 各类矩阵引擎指令的统称。

### MPSS

Intel Manycore Platform Software Stack，驱动与运行 Xeon Phi（KNC）的软件栈。本机 3.8.6。

### NLC

NEC Numeric Library Collection，含 BLAS/LAPACK 等；VE 上 `cblas_dgemm` 高性能路径。

### NUMA

Non-Uniform Memory Access。本机 2 节点；跨节点访存更慢。uni 的 `NUMABinder` 做亲和。

### PCIe

外设互连。本机加速卡多为 Gen3×16；**Host↔Device 带宽远低于卡内 HBM/GDDR**，是异构主瓶颈。

### PowerCap

软件功率封顶，防止 3×VE+Phi+CPU 同时满载超过 PSU。uni 已实现策略，本仓库继承。

### Phi / Xeon Phi 7120P

本机 coprocessor 加速卡：61 核、16GB GDDR5、被动散热，适合不规则控制流与 x86 代码。

---

## R–Z

### Row-major / Column-major

行主序 stride 例 `(N, 1)`；列主序例 `(1, M)`（对 \(M\times N\) 逻辑矩阵）。布局必须显式，不可默认臆测。

### SpMV

Sparse Matrix-Vector multiplication，稀疏矩阵向量乘。uni 示例中 Phi 分块 + VE 并行。

### Swizzle

基于 XOR 的地址置换，经典用途是减少 GPU **shared memory bank conflict**。  
**本仓库**: 可学习；映射到 VE/Phi 时需单独研究，不可直接假设 bank 几何相同。

### tensor-layouts

Meta/社区维护的纯 Python CuTe 布局代数库（PyPI: `tensor-layouts`）。**无需 GPU**。

### Thread-value layout

描述「哪个线程持有矩阵块中哪些元素」的布局，是 atom 的核心输出之一。

### Tile / Tiler

小块；tiler 是用于 divide/compose 的分块描述（可为 shape 或 `Layout`/`Tile`）。

### uni-framework

本机异构协同框架（Phi + 3×VE + Host 调度）。GitHub: mobius/uni-framework；本地常见路径 `/home/joey/Work/uni`。

### upcast / downcast

在 bit 坐标与 element 坐标等不同粒度之间换算布局（CuTe 同名概念）。

### uv

Astral 的 Python 包/环境工具。本项目**首选**用其创建本地 venv，避免污染系统 Python。

### VE / Vector Engine

NEC 向量引擎加速卡。本机 3× VE 1.0，编译器 `ncc`，执行 `ve_exec`，高 HBM 带宽，稠密 FP64 主力。

### VEOS

VE 操作系统/管理栈组件；本机 `ve_exec` 报告 3.6.1。

### VNNI

见 AVX512_VNNI。

---

### ve_exec

NEC 工具：在指定 VE 节点上执行 VE 架构二进制（`ve_exec -N <id> ./prog`）。本仓库 multi-VE 测试通过它对每张卡启动 NLC DGEMM。

### Staging (文件暂存)

Host 将输入矩阵写成 `.bin`，VE 进程读文件、写输出，再由 Host 读回。实现简单、依赖 VE 可见 Host FS；端到端吞吐受文件 I/O 限制，卡内 kernel GFLOPS 仍可接近 NLC 峰值。

### micnativeloadex

在 Host 上加载并运行 MIC native（`.mic`）程序的工具。Phi peak smoke 使用该路径。带文件输入的 DGEMM 则改用 **scp + ssh mic0**，因为卡上进程不能直接 `fopen` Host 路径。

### Comp-CL / CCompL

Intel FlexLM 特性名。ICC 16.0（PSXE 2016）检出 **Comp-CL**；较新的 Parallel Studio 许可组件列表常用 **CCompL**。若 license 仅有 CCompL 而无 Comp-CL，则出现 *No such feature exists (-5,357)*，本项目回退 **k1om-gcc** 交叉编译。

### k1om / K1OM

Knights Corner 的 ELF 机器类型（Intel K1OM）。MPSS 提供 `k1om-mpss-linux-gcc` 交叉工具链，可不依赖 ICC license 生成 `.mic` 可执行文件。

## 文档维护日志

| 日期 | 变更 |
|------|------|
| 2026-07-14 | 初版：覆盖 tensor-layouts / uni / 本机硬件相关词条 |
| 2026-07-14 | 追加 ve_exec、Staging、micnativeloadex（实机测试相关） |

# cpu-cute-tensor 架构设计

> 文档时间: 2026-07-14 01:35:34  
> 状态: v0 草案（实施前）  
> 依赖调研:  
> - `docs/research/20260714_013534_hw_env_feasibility.md`  
> - `docs/research/20260714_013534_tensor_layouts_and_uni_survey.md`

---

## 1. 目标与非目标

### 1.1 目标

1. 在 **无 NVIDIA GPU** 的异构机上，用 **CuTe 布局代数** 统一描述 Host / Phi / VE 上的张量存储与访问。  
2. 复用 **uni-framework** 的设备发现、NUMA、功率、DAG 调度能力。  
3. 提供可运行的：布局推演、可视化、布局驱动分片、至少一条端到端 demo（如 layout-planned DGEMM 或 SpMV）。  
4. 全过程文档化（research / plan / impl / architecture），术语入 `glossary.md`。  
5. 环境隔离（uv 优先），提交前敏感信息 audit。

### 1.2 非目标（v1）

- 完整 CUTLASS 级自动代码生成  
- 跨机多节点 MPI 集群  
- 在本机跑通真实 Tensor Core / AMX 性能  
- 替换 NLC / MKL 手写全部 BLAS  

---

## 2. 逻辑架构

```
                    ┌──────────────────────────────────────┐
                    │           User / CLI / notebooks       │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Application Layer              │
                    │  layout_gemm · layout_spmv · demos     │
                    └──────────────────┬───────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
┌─────────▼─────────┐      ┌───────────▼──────────┐     ┌──────────▼──────────┐
│  Layout Core      │      │  Device Placement    │     │  Scheduler Bridge   │
│  tensor-layouts   │      │  partitioners        │     │  → uni TaskGraph    │
│  + atoms_*        │      │  (host/phi/ve)       │     │  numa/power/devices │
└─────────┬─────────┘      └───────────┬──────────┘     └──────────┬──────────┘
          │                            │                            │
          └────────────────────────────┼────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Backend Emit / Run             │
                    │  host_avx · phi_knc · ve_ncc           │
                    └──────────────────────────────────────┘
```

### 2.1 分层职责

| 层 | 职责 | 技术 |
|----|------|------|
| Layout Core | 代数正确性、atom 定义、可视化 | `tensor-layouts` + 本仓 `src/cpu_cute_tensor/atoms/` |
| Device Placement | 按 shape 做多设备 `logical_divide`、PCIe 成本估计 | 纯 Python |
| Scheduler Bridge | 将 layout plan 变为任务图 | 引用或适配 uni `task_graph` |
| Backend | 编译/执行真实内核 | gcc / ICC-mic / ncc + NLC |
| Apps | 可复现 demo 与基准 | scripts + tests |

---

## 3. 布局模型（设备无关 → 设备相关）

### 3.1 设备无关视图

任何缓冲用：

```text
TensorView = {
  dtype,
  layout: Layout | ComposedLayout,  # from tensor-layouts
  device: host | phi0 | ve{0,1,2},
  base_ptr_or_path,                 # 运行时填充；文档中不出现真实密钥路径
}
```

### 3.2 多级层次 shape 约定（建议）

以矩阵 \(C_{M\times N}\) 为例（列主或行主显式写入 stride）：

```text
shape ≈ (
  (simd_w,  core_tile_m),   # 向量 / 微内核
  (cores,   device_tile),   # 核间
  (devices, batch)          # 多卡 / 批
)
```

具体数字在 Phase 2 用基准校准，架构层只固定 **模式** 不写死最优参数。

### 3.3 自定义 Atoms（v1 定义方向）

| Atom 族 | 语义 | shape_mnk 示例（待校准） | 产出 |
|---------|------|---------------------------|------|
| `HOST_AVX512_8x8xK_F64` | 主机 FMA tile | 与 8-wide ZMM 对齐 | `a/b/c_layout` 线程-值布局 |
| `HOST_VNNI_*` | INT8/BF16 类（若后续需要） | 对齐 VNNI | 可选 |
| `PHI_KNC_8x8xK_F64` | KNC 512-bit FMA tile | 对齐 IMCI | 与 Host 类似但核数/带宽不同 |
| `VE_NLC_DGEMM_TxTxK` | 描述调用 NLC 的外层 tile | 对齐 NLC 高效尺寸 | 强调 **不重写内部**，只描述分块 |
| `VE_VECTOR_1DxC` | 长向量 map/reduce | 1D 为主 | SpMV / axpy |

Atoms **不声称** 与 NVIDIA MMA 二进制兼容；仅复用 CuTe 的 thread-value layout 表达习惯。

---

## 4. 与 uni-framework 的集成方式

### 4.1 推荐集成（默认）

**松耦合路径引用**（避免强绑 monorepo）:

```text
cpu-cute-tensor/
  third_party/ 或 env 配置 UNI_ROOT=/home/joey/Work/uni
  src/cpu_cute_tensor/bridge/uni_adapter.py
```

适配器职责:

- 调用 `devices.discover()`  
- 将 `PlacementPlan` 转为 `TaskGraph` 节点  
- 继承 `PowerCap` / `NUMABinder`  

### 4.2 备选

- git submodule 引入 uni（需清理敏感与大产物）  
- 精简复刻 devices/power 接口（仅当 uni 路径不稳定）  

### 4.3 数据面

| 路径 | 机制 | Layout 角色 |
|------|------|-------------|
| Host↔VE | ve_exec / AVEO / 文件映射（按 uni 现有） | 规划 H2D 块大小与对齐 |
| Host↔Phi | scp + micnativeloadex | 最小化往返；元数据随包 |
| VE↔VE | NEC MPI | AllReduce 布局与 ring 映射 |

---

## 5. 仓库结构（目标态）

```text
cpu-cute-tensor/
├── README.md
├── pyproject.toml              # uv 管理，python>=3.10
├── docs/
│   ├── glossary.md
│   ├── research/               # 时间戳调研
│   ├── plan/                   # 时间戳计划
│   ├── architecture/           # 时间戳架构
│   └── impl/                   # 时间戳实现记录
├── env/                        # uv venv 位置（gitignore）
├── src/cpu_cute_tensor/
│   ├── __init__.py
│   ├── atoms/
│   │   ├── host_avx512.py
│   │   ├── phi_knc.py
│   │   └── ve.py
│   ├── partition/
│   │   ├── multi_device.py
│   │   └── pcie_cost.py
│   ├── bridge/
│   │   └── uni_adapter.py
│   └── backends/
│       ├── host/
│       ├── phi/
│       └── ve/
├── scripts/
│   ├── check_hw.sh             # 实施前复检
│   ├── audit_sensitive.sh
│   └── ...
├── examples/
├── tests/
└── .gitignore
```

`.gitignore` 必须包含: `.venv/`, `env/.venv/`, `*.mic` 大二进制可选策略、`__pycache__`、本地日志、`*.pem`、`.env`。

---

## 6. 关键数据流（端到端 demo）

**Demo A — Layout-planned multi-VE DGEMM（优先）**

1. Python: 用 `Layout` 描述 \(A,B,C\) 与三卡 `logical_divide` 行/列块。  
2. `pcie_cost` 估计仅首尾传输。  
3. `uni_adapter` 生成 3 个 VE 任务（NLC dgemm）+ Host 归约或直接 in-place 分块写回。  
4. 校验: 与 Host numpy/参考实现最大误差。  
5. 记录 GFLOPS 与 PCIe 时间到 `docs/impl/`。

**Demo B — Phi 预处理 + VE 稠密（继承 uni 流水线）**

1. Layout 描述中间特征矩阵（稀疏→密 的 tile）。  
2. Phi 任务写中间缓冲；布局保证 VE 友好对齐。  
3. 强调 **中间数据体积** 指标。

---

## 7. 配置与安全

### 7.1 配置

- 环境变量: `UNI_ROOT`, `CCT_VE_NODES`, `CCT_POWER_CAP_W`  
- 配置文件: `config/default.toml`（无密钥）  

### 7.2 敏感信息策略

禁止进入仓库:

- 设备序列号、MAC、内网 IP 段细节（可用 `mic0`/`ve0` 逻辑名）  
- SSH 私钥、密码、license 文件内容  
- 个人 home 下无关绝对路径中的 token  

提交前: `scripts/audit_sensitive.sh` + 人工 diff 审阅。

---

## 8. 测试策略

| 级别 | 内容 | 硬件需求 |
|------|------|----------|
| L0 | tensor-layouts 代数单测 + 本仓 atoms 形状不变量 | 仅 Host Python |
| L1 | partition 逻辑（mock 设备） | 仅 Host |
| L2 | Host AVX 微内核正确性 | Host |
| L3 | 单 VE / 单 Phi smoke | 对应设备 online |
| L4 | 多设备 + power cap 集成 | 全卡，注意功率 |

CI 若无加速卡: 仅 L0–L1；本机手工 L2–L4。

---

## 9. 演进路线（与 plan 对齐）

| 版本 | 架构交付 |
|------|----------|
| v0 | 本文档 + 环境门禁（当前） |
| v1 | Layout Core + Host atoms + 安装骨架 |
| v2 | VE/Phi atoms + partitioner + uni bridge |
| v3 | 端到端 demo + 基准 + 文档收口 |

---

## 10. 开放问题（下轮研究）

1. VE 长向量的「thread-value」类比如何与 8 核 OpenMP 统一？  
2. 是否需要 swizzle 的 HBM bank 模型，还是仅 alignment 约束？  
3. uni 以路径引用还是 submodule？待 Phase 1 末决定。  
4. 布局序列化格式: JSON vs 小型 DSL？

开放问题关闭时在 `docs/research/` 追加时间戳文档，并回链本文。

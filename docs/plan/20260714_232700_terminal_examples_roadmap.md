# 计划：接近终端的真实案例 Examples（v1.x）

> 文档时间: 2026-07-14 23:27:00  
> 基线: **v1.0.0**（API 冻结、DataPlane、Placement、SpMV/dataprep、TaskGraph 桥）  
> 对齐: [uni-framework](https://github.com/mobius/uni-framework) 应用层（hetero_spmv / hetero_dataprep / examples/*）  
> 状态: **已关闭**（X1+X2 / E1–E5 全套，随 v1.1.0 收尾）

---

## 1. 为什么要做

当前 `examples/` 仍是 **算子/路径验证** 视角：

| 现有 | 角色 |
|------|------|
| `demo_atoms.py` | atom 展示 |
| `demo_partition.py` | 分片 JSON |
| `demo_real_ve.py` | multi-VE DGEMM |
| `demo_hetero_pipeline.py` | Phi scale → VE GEMM |

`apps/spmv_dataprep` 已有「不规则 + 稠密」雏形，但：

- 缺 **可讲故事的终端场景**（科学/工程用户听得懂的 job）  
- 缺 **一键入口**（参数、假数据/真实数据格式、产物、timeline）  
- 缺与 **uni 示例语义** 的对照表（同一类 workload，layout 运行时怎么跑）  
- Host 中规模 OpenBLAS 常赢 → 案例必须 **选对规模与指标**（吞吐/重叠/功耗/正确流水线，而非硬刚单次 wall）

**本阶段目标**：用 3～5 个 **终端向 example**，把 v1.0 能力跑成「可演示、可复现、可解释」的应用故事。

---

## 2. 目标与非目标

### 2.1 目标

1. 每个 example 对应 **一类真实工作流**（不是裸 GEMM）。  
2. 统一骨架：`discover → plan/PowerCap → run → check → timeline/summary`。  
3. **Host-only 可降级**（L0/CI 冒烟）+ **本机全栈可选**（Phi/VE）。  
4. 输出：stdout 表 + 可选 `artifacts/examples/<name>/`（plan.json / timeline.jsonl / metrics.json）。  
5. README / `docs/INDEX` 增加「终端案例」导航；术语入 glossary。

### 2.2 非目标

- 重写 uni 的 C 内核全套（优先 **复用本仓库 backend + 布局调度**；必要时再 port 薄内核）。  
- HTTP/微服务（W3.3 服务模式可后续；本轮默认 **CLI example**）。  
- 真实超大公开数据集入库（用 **合成可复现数据** + 可选外部路径）。  
- 强行宣称全面「相对 Host 加速」——指标按场景定义（见 §5）。

---

## 3. 场景筛选原则

| 原则 | 说明 |
|------|------|
| P1 硬件匹配 | 不规则/预处理 → Host 或 Phi；稠密 GEMM/PCA 型 → VE；PowerCap 默认开 |
| P2 故事完整 | 输入（矩阵/CSR/批次）→ 处理 → 输出（结果 + 误差 + 阶段时间） |
| P3 复用 v1 API | `choose_best_placement` / `execute_plan` / `create_dataplane` / `timeline_scope` / `PowerCap` / TaskGraph 桥 |
| P4 可降级 | 无 Phi/无 VE 时明确 skip 或 host path，exit code 语义统一 |
| P5 可对标 uni | 与 uni `hetero_spmv` / `hetero_dataprep` / multi_task / throughput **叙事对齐**，实现可简化 |

---

## 4. 建议案例组合（按优先级）

### E1 — 批处理稠密回归 / 最小二乘风格（P0，先做）

**故事**: 科学计算常见路径：特征矩阵 \(X\) 经标准化后，对多右端 \(B\) 做 \(C = X^\top X\) 相关或 \(X^\top B\) 投影（用 DGEMM 表达）。

| 项 | 内容 |
|----|------|
| 输入 | 合成 \(X\in\mathbb{R}^{m\times k}\)、\(B\in\mathbb{R}^{k\times n}\)（或多次 batch） |
| 流水线 | Host/Phi **列标准化（scale）** → VE **auto-place DGEMM** |
| 复用 | `run_phi_prep_scale` 或 host scale；`choose_best_placement` + `execute_plan` 或 DataPlane |
| 产物 | `C` 校验、wall、plan 策略、timeline |
| 指标 | 正确性；相对 **固定 3-VE 行切** 的 wall；多 batch 时 thr（b/s） |
| 入口 | `examples/e1_batch_dense_regression.py` |

**为何终端向**: 对应「多批次拟合/投影」而不是 demo GEMM；用户参数 `m,k,n,batches,backend`。

---

### E2 — 稀疏图/网格 × 稠密后处理（P0，对齐 uni SpMV）

**故事**: 稀疏离散算子（CSR SpMV）得到场量/特征，再做稠密变换（模态、滤波器、第二阶段 GEMM）。

| 项 | 内容 |
|----|------|
| 输入 | 合成 CSR（可选：简单 2D 五点 stencil → CSR）+ 多 RHS |
| 流水线 | Host CSR SpMV →（可选 Phi scale）→ VE dense GEMM |
| 复用 | `apps/spmv_dataprep` 增强；`random_csr` + **structured_csr_stencil** |
| 产物 | 相对 dense 参考误差；阶段占比 |
| 指标 | e2e pass；multibatch + overlap 吞吐；与 pure-host 对照表（可解释） |
| 入口 | `examples/e2_sparse_then_dense.py` |

**相对现有**: 现 `spmv_dataprep` 偏库函数；example 要 **CLI + stencil 场景说明 + 默认合理尺寸**。

---

### E3 — 数据清洗 / 标准化 / 低维投影（P1，对齐 uni dataprep）

**故事**: 特征流水线：缺失填充或 clip → 标准化 → 投影 \(Y = X W\)（PCA 风格用固定 \(W\) 或随机正交，避免完整特征分解依赖）。

| 项 | 内容 |
|----|------|
| 输入 | 合成「脏」矩阵（NaN 比例、离群） |
| 流水线 | Host 清洗 → Phi/Host 标准化 → VE GEMM 投影 |
| 复用 | host 清洗（新小函数）；scale；`execute_plan` |
| 产物 | 清洗前后统计、投影后 shape、corr/残差类简单检查 |
| 指标 | 正确性；PowerCap 下设备集；timeline |
| 入口 | `examples/e3_dataprep_project.py` |

**注意**: 完整 PCA 特征分解可 Host；**投影 DGEMM 放 VE** 以体现异构分工。不强制复刻 uni 全 C 内核。

---

### E4 — 多任务 DAG 小作业（P1，对齐 uni multi_task）

**故事**: 一个「作业包」：prep → 两路并行稠密（不同 panel）→ host 归约/校验。

| 项 | 内容 |
|----|------|
| 输入 | 2～3 个矩阵任务 + 依赖边 |
| 流水线 | `task_graph_bridge` 或直接 uni TaskGraph |
| 复用 | PowerCap + Timeline + 每节点 backend |
| 产物 | 每节点 status/wall；DAG 总时间 |
| 指标 | uni backend 可用时 pass；否则 local DAG pass |
| 入口 | `examples/e4_job_dag.py` |

---

### E5 — 持续批吞吐「准服务」CLI（P2，对齐 throughput / W3.3 简化）

**故事**: 固定会话（VE pool 或 AVEO pin），连续投递 N 个 job，报告 jobs/s 与 mean wall。

| 项 | 内容 |
|----|------|
| 输入 | `--seconds` 或 `--jobs`，矩阵尺寸 |
| 流水线 | 常驻 pool/pin + DataPlane |
| 复用 | `AveoSessionPool.pin_all` / `VeWorkerPool`；`PowerCap` |
| 产物 | thr、可选 power N/A |
| 指标 | 不低于 oneshot 冷启动基线的文档化倍数 |
| 入口 | `examples/e5_sustained_jobs.py`（可薄包装 `bench_sustained`） |

---

### 明确不做（本轮）

| 案例 | 原因 |
|------|------|
| 完整 Monte Carlo 金融套件 | 与布局运行时主线弱相关；可后续 |
| 真实 MatrixMarket 大库入库 | 体积/许可证；改为可选路径读取 |
| Web UI | 超出 CLI 范围 |

---

## 5. 验收标准（每个 example 共用）

| ID | 标准 |
|----|------|
| A1 | `python examples/eN_*.py --help` 清晰；默认参数可在本机 **&lt; 60s** 跑完 |
| A2 | 有 `--host-only`（或自动降级）时 L0 可跑通正确性（或 skip 并 exit 0/2 有文档） |
| A3 | 全栈路径：`status=pass` 且 `max_abs_err` 达标 |
| A4 | 写出 `artifacts/examples/<name>/metrics.json`（wall、backend、plan、speedup_note） |
| A5 | README「终端案例」表 + 一行复现命令 |
| A6 | 不引入密钥/序列号；audit 绿 |

**成功叙事（对外）**:

- E1/E2/E3：展示 **异构分工与 plan 可复现**  
- E4：展示 **作业图 + PowerCap**  
- E5：展示 **常驻数据面吞吐**  

不强制所有案例 Host wall 加速比 &gt;1。

---

## 6. 目录与代码约定

```
examples/
  README.md                 # 案例索引与指标说明
  _common.py                # env、discover、artifacts、argparse 公共
  e1_batch_dense_regression.py
  e2_sparse_then_dense.py
  e3_dataprep_project.py
  e4_job_dag.py
  e5_sustained_jobs.py
src/uni_cute_tensor/apps/   # 仅当 example 需可复用逻辑时下沉
  (可选) dataprep_clean.py / stencil_csr.py
artifacts/examples/<name>/  # gitignore 已有 artifacts/
docs/impl/<ts>_examples_e1_e5.md
```

公共行为：

```text
--seed --m --k --n --batches
--devices auto|ve1,ve2,ve3
--backend auto|aveo|pool|host
--power-cap / --no-power-cap
--host-only
--out-dir artifacts/examples/<name>
```

---

## 7. 实施切片（建议 2 个迭代）

### 迭代 X1（P0） — **DONE**

1. ~~`examples/_common.py` + `examples/README.md`~~  
2. ~~**E1** 批处理稠密回归~~  
3. ~~**E2** sparse→dense + **stencil5 CSR**~~  
4. ~~短 impl 纪要 + README 导航~~  
5. ~~L0：`tests/test_examples_smoke.py`~~  

**实现纪要**: `docs/impl/20260714_233200_examples_e1_e2.md`  
**出口**: E1+E2 本机全绿；host-only smoke 3 pass。

### 迭代 X2（P1–P2） — **DONE**

1. ~~**E3** dataprep 投影~~  
2. ~~**E4** job DAG~~  
3. ~~**E5** sustained jobs~~  
4. ~~与 uni 的对照表~~（`examples/README.md`）  
5. Phi 仍默认关（`--phi`）  

**实现纪要**: `docs/impl/20260715_010300_examples_e3_e5.md`  
**出口**: E3–E5 host-only smoke + 本机全栈 pass。

---

## 8. 与现有能力的映射

| 案例 | DataPlane | Placement | Timeline | PowerCap | Phi | TaskGraph |
|------|-----------|-----------|----------|----------|-----|-----------|
| E1 | 可选 | ✓ | ✓ | ✓ | 可选 | — |
| E2 | — | ✓ | ✓ | ✓ | 可选 | 可选 |
| E3 | 可选 | ✓ | ✓ | ✓ | 可选 | — |
| E4 | — | ✓ | ✓ | ✓ | 可选 | ✓ |
| E5 | ✓ pin/pool | 固定或 auto | 汇总 | ✓ | — | — |

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Host 总是更快 → 用户失望 | 文档写清指标；对比 **fixed multi-VE / oneshot**；E5 看 thr |
| Phi  license/离线 | `--host-only` / 自动 fallback |
| Example 膨胀难维护 | `_common.py`；逻辑下沉 `apps/` 仅一份 |
| 与 uni 重复 | **叙事对齐、实现轻量**；README 链到 uni 完整版 |

---

## 10. 默认假设（可改）

| 项 | 默认 |
|----|------|
| 首批实施 | **X1：E1 + E2 + 公共骨架** |
| 默认 backend | `auto`（有 VE → plan+pool/AVEO；否则 host） |
| 数据 | 合成 + seed 可复现 |
| 服务化 | 不做 HTTP，仅 E5 CLI 吞吐 |
| 版本 | examples 合入后可标 **v1.1.0**（非必须） |

---

## 11. 下一步（等你确认后实施）

1. 确认案例优先级是否同意 **E1→E2→E3→E4→E5**。  
2. 确认是否必须 **E2 stencil CSR**（更「终端」）还是随机 CSR 即可。  
3. 确认 Phi 在 example 中默认 **探测启用** 还是 **默认关、--phi 打开**。  
4. ~~X1 + X2 均已实施（E1–E5）~~。

---

## 12. 一页对照：uni vs 本仓库

| uni 应用/示例 | 本计划 example | 差异 |
|---------------|----------------|------|
| hetero_spmv | E2 | 本仓库 CSR Host + dense VE plan；uni 可含 Phi 分块 SpMV 内核 |
| hetero_dataprep | E3 | 清洗+标准化+投影；简化 PCA |
| multi_task | E4 | 小 DAG + PowerCap |
| throughput | E5 | 常驻会话 jobs/s |
| basic 峰值 | 不单独立项 | 已有 bench_* |

**本仓库差异化卖点**: CuTe **PlacementPlan 可序列化** + **auto 策略** + **DataPlane/Timeline** 贯穿 example，而不只是「能跑通异构」。

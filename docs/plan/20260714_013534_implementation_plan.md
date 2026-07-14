# uni-cute-tensor 实施方案

> 文档时间: 2026-07-14 01:35:34  
> 状态: **待用户确认后执行 Phase 1**  
> 前置门禁: `docs/research/20260714_013534_hw_env_feasibility.md` → **PASS**  
> 架构: `docs/architecture/20260714_013534_uni_cute_tensor_architecture.md`

---

## 0. 执行原则

1. **实施动作前**再次运行硬件检查（`scripts/check_hw.sh`，Phase 1 创建）。  
2. **软件安装**: `uv` 本地 venv → conda → podman/docker；禁止污染全局。  
3. **过程文档**: 每次迭代在 `docs/{research,plan,impl,architecture}/` 写 `YYYYMMDD_HHMMSS_*.md`。  
4. **新术语**: 解释并写入 `docs/glossary.md`。  
5. **提交前**: 敏感信息 audit。  
6. **功率**: 多卡满载测试必须启用 PowerCap / 人工确认。

---

## 1. 阶段总览

| Phase | 名称 | 产出 | 依赖硬件 | 预估 |
|-------|------|------|----------|------|
| **0** | 门禁与调研 | 本文档 + research + architecture + glossary | 探测即可 | ✅ 已完成 |
| **1** | 工程骨架与 tensor-layouts | pyproject、uv 环境、依赖安装、冒烟 | 仅 Host | 0.5–1 d |
| **2** | 自定义 Atoms + 分区器 | host/ve/phi atoms、multi_device partition | Host（逻辑） | 1–2 d |
| **3** | uni Bridge | 设备发现对接、TaskGraph 适配 | Phi+VE online | 1–2 d |
| **4** | Backend 内核 | Host 微内核 + VE NLC 包装 + Phi smoke | 对应卡 | 2–4 d |
| **5** | 端到端 Demo + 基准 | layout DGEMM / 流水线、impl 记录 | 全栈 | 1–2 d |
| **6** | 硬化 | 测试矩阵、audit 脚本、README、可选 submodule | — | 1 d |

---

## 2. Phase 0 — 门禁（已完成）

### 完成项

- [x] 探测 CPU / 无 GPU / Phi / VE  
- [x] 对照 uni-framework 与本机一致性  
- [x] 研读 tensor-layouts 能力边界  
- [x] 可行性 **PASS**  
- [x] 文档落盘  

### 门禁结论摘要

- tensor-layouts 代数: **可跑**  
- 异构主路径 Host+Phi+VE: **可跑**  
- NV/AMD/AMX/Xe 实机: **不可跑（仅对照）**  

---

## 3. Phase 1 — 工程骨架与 tensor-layouts

### 3.1 前置检查（实施时必做）

```bash
# 设备
micctrl -s | head
ls /dev/ve0 /dev/ve1 /dev/ve2
# 工具
uv --version
ncc --version
```

任一项失败 → 停止并写 `docs/impl/<ts>_phase1_blocked.md`。

### 3.2 环境（uv 优先）

```bash
cd /mnt/storage/hdd1/uni-cute-tensor
uv venv --python 3.13 env/.venv
source env/.venv/bin/activate
uv pip install "tensor-layouts[viz,test]" pytest ruff
# 验证
python -c "from tensor_layouts import Layout; print(Layout((4,8),(1,4))(2,3))"
pytest --pyargs tensor_layouts  # 或 clone 后测；以包实际导出为准
```

说明:

- 使用 **Python ≥3.10**（推荐 3.13，uv 已缓存；或系统 3.11）。  
- venv 目录 `env/.venv` 必须 gitignore。  
- 不使用 `sudo pip`。

### 3.3 仓库骨架

- `pyproject.toml`（包名 `uni-cute-tensor`，依赖 `tensor-layouts`）  
- `src/uni_cute_tensor/` 空包可导入  
- `scripts/check_hw.sh`  
- `scripts/audit_sensitive.sh`  
- `.gitignore`  
- `README.md`（无敏感信息）  

### 3.4 验收

| ID | 标准 |
|----|------|
| P1-1 | `uv run python -c "import tensor_layouts"` 成功 |
| P1-2 | 简单 Layout 计算 `14`（示例 `(2,3)` → 14） |
| P1-3 | `docs/impl/<ts>_phase1_env.md` 记录版本（python/uv/tensor-layouts） |

### 3.5 回滚

删除 `env/.venv` 即可；无全局污染。

---

## 4. Phase 2 — Atoms 与分区

### 4.1 任务

1. 实现 `atoms/host_avx512.py`: 至少 1 个 FP64 tile atom（shape_mnk + a/b/c layout）  
2. 实现 `atoms/ve.py`: NLC 外层 tile 描述（非重写 DGEMM）  
3. 实现 `atoms/phi_knc.py`: 与 Host 类似的 KNC 向量宽度 atom  
4. `partition/multi_device.py`: 对 `(M,N)` 按设备数 `logical_divide`  
5. 单元测试: 布局 size/cosize、分片不重叠覆盖  

### 4.2 验收

| ID | 标准 |
|----|------|
| P2-1 | Host atom 可打印 shape_mnk 与 c_layout |
| P2-2 | 3 路 VE 行切分：覆盖完整 M 且无交 |
| P2-3 | `docs/impl/<ts>_phase2_atoms.md` + 必要图（可选 viz） |

### 4.3 技术说明（实施时写进 glossary 的已列词）

- Atom = 最小矩阵乘/向量操作的线程-值布局描述  
- logical_divide = 逻辑域分解为 tile + rest  

---

## 5. Phase 3 — uni Bridge

### 5.1 任务

1. 配置 `UNI_ROOT` 默认探测 `/home/joey/Work/uni`  
2. `bridge/uni_adapter.py`: discover devices → 映射 `PlacementPlan`  
3. 将分区结果变为可调度任务描述（函数或 JSON plan）  
4. 接入 power 标志（默认保守）  

### 5.2 验收

| ID | 标准 |
|----|------|
| P3-1 | 发现 1 Phi + 3 VE（或明确降级） |
| P3-2 | 对 mock GEMM plan 生成 ≥3 任务节点 |
| P3-3 | 无设备时 L1 测试仍通过 |

### 5.3 风险

- uni 代码路径变更 → 适配器版本探测  
- 导入 uni 时 sys.path 污染 → 限定路径  

---

## 6. Phase 4 — Backend 内核

### 6.1 Host

- 参考 atom 的 micro-kernel（C 或 Python+numpy 先正确后加速）  
- 可选: OpenMP + 显式 AVX-512 intrinsics（第二迭代）  

### 6.2 VE

- 最小 `ncc` 程序或复用 uni `kernels/ve`  
- 链接 NLC `cblas_dgemm`  
- 由 layout 决定 leading dimension 与 offset  

### 6.3 Phi

- 复用 uni Phi 编译执行路径  
- smoke: 小规模 dgemm/axpy，验证偏移与布局一致  

### 6.4 验收

| ID | 标准 |
|----|------|
| P4-1 | Host 参考 vs 内核 max_abs_err 阈值阈值（FP64 建议 1e-12 相对/绝对按规模） |
| P4-2 | 单 VE NLC 与 Host 参考一致 |
| P4-3 | Phi smoke 通过或文档标注阻塞原因 |
| P4-4 | impl 文档含命令与脱敏日志 |

---

## 7. Phase 5 — 端到端 Demo 与基准

### 7.1 Demo 优先级

1. **Multi-VE layout DGEMM**（算力主力）  
2. **Host layout viz 画廊**（教学）  
3. **Phi→VE 流水线**（展示异构 + layout 对齐）  

### 7.2 基准指标

| 指标 | 说明 |
|------|------|
| 正确性 | max_diff / rel_err |
| 性能 | GFLOPS（卡内） |
| PCIe | H2D/D2H 时间占比 |
| 调度 | 任务 launch 开销 |
| 功率 | 是否触发 cap |

### 7.3 验收

- `examples/` 可一键跑（文档化）  
- `docs/impl/<ts>_phase5_e2e.md`  
- 更新 architecture 若接口有变（新时间戳文档，不覆盖旧文件）  

---

## 8. Phase 6 — 硬化与提交

1. `scripts/audit_sensitive.sh` 扫描: 序列号模式、私钥头、`.env`、内网 IP 等  
2. README 安装与硬件前提  
3. 测试: L0–L1 默认；L2–L4 标记 `@pytest.mark.device`  
4. 提交信息不含机器私有数据  
5. 写 `docs/impl/<ts>_phase6_release.md`  

---

## 9. 环境与依赖矩阵

| 组件 | 安装位置 | 方式 |
|------|----------|------|
| Python 3.13 + venv | `env/.venv` | `uv venv` |
| tensor-layouts | venv | `uv pip install` |
| pytest/ruff/matplotlib | venv | `uv pip install` |
| ncc / ve_exec | 系统 `/opt/nec/ve` | 已装，不重装 |
| MPSS / mic* | 系统 | 已装，不重装 |
| ICC for Phi | uni 既有 podman 方案 | 按需，不装全局 |

---

## 10. 决策记录（本轮）

| ID | 决策 | 理由 |
|----|------|------|
| D1 | 主路径 = Host 代数 + VE/Phi 后端，非 CUDA | 本机无 NVIDIA |
| D2 | 复用 uni 调度，不重写 | Phase 0–4 已验证 |
| D3 | uv + 项目 venv | 用户要求 + 系统 Python 3.6 过旧 |
| D4 | Atoms 自定义三类 | 厂商 atom 无实机 |
| D5 | 文档四分类 + 时间戳 | 用户要求可追溯 |

---

## 11. 下一步（需你确认）

请确认是否进入 **Phase 1**（创建 uv 环境、安装 tensor-layouts、搭仓库骨架）。  

可选偏好（可回复编号）:

1. Python 版本: **3.13（推荐）** / 3.12 / 系统 3.11  
2. uni 集成: **路径引用 UNI_ROOT（推荐）** / submodule / 暂缓到 Phase 3  
3. 首个 demo: **Multi-VE DGEMM（推荐）** / 仅 Host viz / Phi+VE 流水线  

确认前 **不会** 安装软件或写入业务代码（本轮仅文档）。

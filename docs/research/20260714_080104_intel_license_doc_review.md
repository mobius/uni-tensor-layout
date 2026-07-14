# 调研：历史文档中的 Intel License 用法 vs 当前 ICC 失败原因

> 文档时间: 2026-07-14 08:01:04  
> 对照路径: `~/Work/intel_phi`、`~/Work/uni`、`~/parallel_studio.lic`  
> 原则: 不记录 SN/HOSTID/签名内容

---

## 1. 结论（先说清楚）

**你说的「之前就是这么用 license」——和历史文档一致。**  
文档/脚本里的用法从来都是：

1. 安装阶段提供 `parallel_studio.lic`  
2. 运行时只 `source /opt/intel/bin/compilervars.sh intel64`  
3. 然后直接 `icc -mmic ...`（uni 的 `phi.py` 也是如此，**没有**再 export 一堆路径）

**当前失败并不是「没用对路径 / 没按文档挂 license」。**  
License 文件已被 ICC 找到；失败原因是：

| 层级 | 事实 |
|------|------|
| ICC 16.0 请求的 feature | **`Comp-CL`** |
| `parallel_studio.lic` 经 `chklic` 列出的 feature | 有 **`CCompL` / `CCompW`** 等，**没有 `Comp-CL`** |
| FlexLM 错误 | `-5,357` *No such feature exists* for Comp-CL |
| 安装时 silent.cfg | **`ACTIVATION_TYPE=trial_lic`**（试用激活），同时挂了 license 文件路径 |

因此：历史「能编」很大概率是 **试用期 Trusted Storage 仍有效**；  
试用结束后 ICC 落到 `.lic` 文件上，却要不到 `Comp-CL`，就表现为现在这样。

这不是否定「按文档放 `~/parallel_studio.lic`」的流程，而是 **同一文件在安装期/试用期与「仅靠 file 检出 Comp-CL」语义并不相同**。

---

## 2. 历史文档怎么写的

### 2.1 安装 / 评估

`intel_phi/docs/research/2026-05-20_014036_icc_psxe_assessment.md`：

- 包: `parallel_studio_xe_2016.tgz`
- **License: parallel_studio.lic**
- 容器: CentOS 7 (`centos7-phi-dev`)
- 结果: ICC 16.0 可用、`icc -mmic` 验证完成

### 2.2 运行时（从不单独讲 license 环境变量）

`intel_phi/docs/icc-usage.md` / `README.md` / `container/README.md`：

```bash
source /opt/intel/bin/compilervars.sh intel64
icc -std=c99 -mmic -O3 ...
```

### 2.3 uni 调度层（与文档同一模式）

`uni/src/scheduler/phi.py`：

```text
podman exec centos7-phi-dev bash -c '
  source /opt/intel/bin/compilervars.sh intel64 &&
  icc -std=c99 -mmic -O3 -openmp -o /tmp/peak_fp64.mic ...'
```

**没有** `INTEL_LICENSE_FILE=...`，依赖容器内已安装/已激活状态。

### 2.4 silent.cfg（关键细节，容易被忽略）

`intel_phi/psxe_install/silent.cfg` 实际内容：

```text
ACTIVATION_LICENSE_FILE=/tmp/psxe_install/parallel_studio.lic
ACTIVATION_TYPE=trial_lic
```

含义：

- 安装器 **确实引用了** `parallel_studio.lic`（路径形态与「用这个文件」的叙述一致）  
- 但激活类型是 **`trial_lic`（试用）**，不是 `license_file` / `serial_number`  
- 试用成功时，FlexLM **Trusted Storage** 会提供编译器可检出的临时权利（含 ICC 要的 feature 名）

容器内 ISM 数据库时间戳均为 **2026-05-20**（安装日），与评估日志一致。

---

## 3. 当前探测（按文档路径复现）

按文档方式（仅 compilervars + 标准 licenses 目录）复现：

1. 将 `~/parallel_studio.lic` 放入容器 `/opt/intel/licenses/`（及 silent 时代的 `/tmp/psxe_install/`）  
2. `source compilervars.sh intel64`  
3. `icc -mmic ...`

结果：

- License path 列表中 **已包含** `parallel_studio.lic`（文件已被使用）  
- 仍报：`A license for Comp-CL is not available (-5,357)`  
- `chklic` 对同一文件报告：**签名有效**、可列出 `CCompL` 等 feature，**列表中无 `Comp-CL`**

`peak_fp64.mic` 等现网二进制 **Modify: 2026-06-03**，说明 6 月初仍能成功 `icc -mmic` 链接产物——与「试用尚未过期」时间线相容（安装 5-20，约一个月量级试用窗口内）。

---

## 4. 为何「以前能用、现在同样挂 license 却不行」

```text
[2026-05-20 安装]
  silent.cfg: trial_lic + parallel_studio.lic 路径
       │
       ▼
  Trusted Storage 获得试用权利 ──► ICC 检出 Comp-CL 成功
       │
[2026-06-03 左右]
  仍可能在试用期内 ──► uni 成功生成 peak_fp64.mic
       │
[试用结束后 / Trusted Storage 失效]
  ICC 回落到 .lic 文件
       │
       ▼
  .lic 只有 CCompL，没有 Comp-CL ──► -5,357 失败
```

**路径用法没变；变的是「试用权利是否还在」。**

---

## 5. 对 uni-tensor-layout 的含义

| 项 | 说明 |
|----|------|
| 文档流程 | 继续兼容：`INTEL_LICENSE_FILE=$HOME/parallel_studio.lic` → 注入 `/opt/intel/licenses/` |
| 探测 | `try_icc_license()` 正确反映「文件在、但 Comp-CL 不可用」 |
| 回退 | `k1om-gcc` 交叉编译 **不依赖** Comp-CL，已用于实机 DGEMM 正确性 |
| 若要恢复 ICC | 需要：未过期的试用/正式激活 **或** 含 **`Comp-CL`** feature 的许可（或 Intel 侧 feature 别名/新版工具链） |

**不能**仅靠「再 copy 一次同一 `parallel_studio.lic`」让 ICC 16 突然认出 `Comp-CL`——`chklic` 已证明文件里没有该 feature 名。

---

## 6. 建议动作（需你决策）

1. **若目标是 ICC intrinsics / `-openmp` 官方路径**  
   - 用安装器把 `ACTIVATION_TYPE` 改为 `license_file` 重新激活（可能仍因无 Comp-CL 失败）  
   - 或向 license 提供方确认是否应包含 **Comp-CL**（PSXE 2016 Composer 经典 feature 名）  
   - 或重新获得可用的试用/订阅并写入 Trusted Storage  

2. **若目标是继续异构实测**  
   - 保持现状：`k1om-gcc` + scp/ssh 做 Phi 正确性；ICC 预编译 peak 仍可跑  
   - 文档已区分「license 用法正确」vs「feature/试用状态」  

3. **修正我们先前 impl 表述**  
   - 不宜写成「license 用错了」  
   - 应写成「用法与 intel_phi/uni 文档一致；当前阻塞是 Comp-CL vs CCompL + 试用激活过期」

---

## 7. 参考文件（只列路径）

- `intel_phi/psxe_install/silent.cfg`  
- `intel_phi/docs/research/2026-05-20_014036_icc_psxe_assessment.md`  
- `intel_phi/docs/icc-usage.md`  
- `intel_phi/container/README.md`  
- `uni/src/scheduler/phi.py`  
- 本机 `~/parallel_studio.lic`（不入库）

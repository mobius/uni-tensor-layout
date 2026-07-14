# 硬件与软件环境可行性评估

> 文档时间: 2026-07-14 01:35:34 (local) / 2026-07-14T05:35:34Z  
> 主机: g4 (ASUS ESC4000 G4 类机, Rocky Linux 8.10 / el8)  
> 工作区: `/mnt/storage/hdd1/uni-cute-tensor`  
> 目标: 在实施 tensor-layouts × uni-framework 方案前，判定本机是否可支撑

---

## 1. 评估结论（先读）

| 能力层 | 结论 | 说明 |
|--------|------|------|
| **CuTe / tensor-layouts 代数（纯 Python）** | ✅ **可完整支持** | 库声明 *No GPU required*；本机用 `uv` + Python ≥3.10 即可 |
| **Host CPU（AVX-512 / VNNI）布局驱动内核** | ✅ **可支持** | 2× Xeon Gold 6252，含 AVX-512F/BW/DQ/VL + `avx512_vnni` |
| **NEC VE 1.0 ×3 布局驱动内核** | ✅ **可支持** | `/dev/ve0..2` 在线，`ncc` 5.4.1 / `ve_exec` 3.6.1 可用 |
| **Intel Xeon Phi 7120P 布局驱动内核** | ✅ **可支持** | `/dev/mic0` online，MPSS 3.8.6，`micnativeloadex` 可用 |
| **NVIDIA Tensor Core / CUDA 实机 MMA** | ❌ **不可运行** | 无 NVIDIA GPU；`atoms_nv` 仅作教学/对照 |
| **AMD CDNA MFMA 实机** | ❌ **不可运行** | 无 AMD GPU |
| **Intel AMX 实机** | ❌ **不可运行** | Cascade Lake 无 AMX（需 Sapphire Rapids+） |
| **Intel Xe DPAS 实机** | ❌ **不可运行** | 无 Arc / Ponte Vecchio |
| **全加速卡同时满载** | ⚠️ **需功率封顶** | uni 已记录 PSU 1600W vs 满载约 1730W |

**总判**: 本机**完全具备**落地「以 CuTe 布局代数为中心、面向 Host/Phi/VE 异构」方案的硬件与工具链；**不具备** NVIDIA/AMD/AMX/Xe 实机验证。实施方案应以 **CPU-cute + VE/Phi atoms** 为主路径，厂商 GPU atoms 为参考对照。

---

## 2. 主机 CPU

| 项 | 实测值 |
|----|--------|
| 型号 | Intel(R) Xeon(R) Gold 6252 @ 2.10GHz |
| 架构 | Cascade Lake (x86_64) |
| Socket / 核 / 线程 | 2 / 24×2 / 96 |
| NUMA | 2 nodes（node0: 0-23,48-71；node1: 24-47,72-95） |
| L3 | 36608 KB / socket |
| 相关 ISA | `avx2`, `avx512f`, `avx512dq`, `avx512cd`, `avx512bw`, `avx512vl`, `avx512_vnni`, `fma` |
| **AMX** | **无** |
| 主机内存 | ~62 GiB 可用约 55 GiB |
| 内核 | 4.18.0-553.124.1.el8_10.x86_64 |

**对布局方案的含义**:

- Host 侧可把 CuTe 风格 `(shape, stride)` 映射到 **AVX-512 向量宽度**（FP64: 8-wide，FP32: 16-wide，BF16/INT8 VNNI 相关模式）与 **cache-line / NUMA 分块**。
- 不可依赖 AMX tile 寄存器；`atoms_amx` 仅能做符号/可视化对照。

---

## 3. GPU / 显示

| 项 | 实测值 |
|----|--------|
| NVIDIA | 无 `nvidia-smi`，无 CUDA 设备 |
| AMD ROCm | 无 `rocm-smi` |
| 显示控制器 | ASPEED Graphics Family（BMC 管理显示，非算力） |

**对布局方案的含义**: tensor-layouts 的 NVIDIA/AMD MMA atom 与 swizzle 教学可在主机 Python 中跑通；**不能**做实机 Tensor Core 验证。

---

## 4. 算力卡（与 uni-framework 对齐）

### 4.1 Intel Xeon Phi 7120P (KNC)

| 项 | 实测值 |
|----|--------|
| PCIe | `5e:00.0` Co-processor: Intel Xeon Phi SE10/7120 |
| 设备节点 | `/dev/mic0` |
| 状态 | `mic0: online`（linux image, MPSS 3.8.6） |
| 驱动 / MPSS | Driver 3.8.6-1, MPSS 3.8.6 |
| 核心 | 61 active cores @ ~1.238 GHz |
| 内存 | 16 GB GDDR5（ECC on，来自 micinfo / uni 文档） |
| 工具 | `micinfo`, `micctrl`, `micnativeloadex` |
| 序列号 | 已采集于 micinfo（**文档/代码中禁止提交序列号**） |

### 4.2 NEC Vector Engine 1.0 ×3

| 项 | 实测值 |
|----|--------|
| PCIe | `3b:00.0`, `af:00.0`, `d8:00.0` NEC VE 1.0 |
| 设备节点 | `/dev/ve0`, `/dev/ve1`, `/dev/ve2`；`/dev/veslot1..3` |
| VEOS / ve_exec | 3.6.1 |
| 编译器 | ncc/nc++/nfort **5.4.1** |
| 工具链路径 | `/opt/nec/ve/bin/`（含 mpicc、NLC 相关脚本） |

### 4.3 与 uni-framework 一致性

本机配置与 [mobius/uni-framework](https://github.com/mobius/uni-framework) 描述一致：

- 服务器族: ESC4000 G4 + 2× Gold 6252  
- 加速卡: **1× Phi 7120P + 3× VE 1.0**  
- 本地已有工程: `/home/joey/Work/uni`（Phase 0–4 已完成）

---

## 5. 软件环境与隔离策略

| 工具 | 状态 | 策略 |
|------|------|------|
| 系统 Python | 3.6.8（`/usr/bin/python3`） | **不可**直接跑 tensor-layouts（要求 ≥3.10） |
| Python 3.11 | 系统已有 `/usr/bin/python3.11` | 可用 |
| uv | 0.11.14（`~/.local/bin/uv`） | **首选**：项目本地 venv |
| uv 已缓存 Python | cpython-3.13.13 | 可直接 `uv venv --python 3.13` |
| conda | 未安装 | 不依赖 |
| podman | 4.9.4-rhel | 备选：ICC/MPSS 等容器化工具链（与 uni 一致） |
| docker | 未检测 | 次选 |
| gcc | 8.5.0 (RHEL) | Host 侧 C 内核 |
| ncc | 5.4.1 | VE 内核 |
| 磁盘 | `/mnt/storage/hdd1` 约 3.4T 可用 | 工作区充足 |

**环境安装优先级（强制）**:

1. `uv` 创建项目 `.venv` / `env/.venv`（不污染全局 site-packages）  
2. 若 uv 不可用 → conda env  
3. 若需隔离编译器/老 SDK → podman（其次 docker）  
4. **禁止** `sudo pip install` 或向系统 Python 写包

---

## 6. 能力矩阵：tensor-layouts 功能 × 本机

| 模块 | 主机可执行 | 实机加速语义 | 建议用途 |
|------|------------|--------------|----------|
| `Layout` / compose / complement / divide / product | ✅ | N/A（代数） | 核心依赖 |
| `Swizzle` / `ComposedLayout` | ✅ | 可映射到 bank/lane 冲突模型 | 教学 + VE/Phi bank 类比 |
| `viz`（matplotlib） | ✅ | 无 GPU 依赖 | 文档与调试 |
| `atoms_nv` | ✅ 符号 | ❌ 无 GPU | 对照 |
| `atoms_amd` | ✅ 符号 | ❌ | 对照 |
| `atoms_xe` | ✅ 符号 | ❌ | 对照 |
| `atoms_amx` | ✅ 符号 | ❌ 无 AMX | 对照 |
| **自定义 `atoms_host_avx512`** | 待实现 | ✅ Host | 主路径 |
| **自定义 `atoms_ve`** | 待实现 | ✅ VE | 主路径 |
| **自定义 `atoms_phi_knc`** | 待实现 | ✅ Phi | 主路径 |
| Oracle: pycute / CUTLASS C++ | ⚠️ 可装可跳过 | 需 CUDA headers | 可选 CI |
| Oracle: AMD calculator | 可选 clone | 无 GPU | 可选 |

---

## 7. 风险与约束（实施前必须遵守）

1. **PCIe Gen3×16**: 加速器内存带宽 ≫ Host↔Device；布局分块必须以 **卡内闭环** 为默认。  
2. **Host DRAM 64GB vs 加速器 ~160GB**: 大数据集不得整机镜像到 Host；采用流式/分 tile 调度。  
3. **功率**: 与 uni 一致，TaskGraph 层继承 PowerCap；禁止默认四卡同时峰值。  
4. **Phi I/O**: 无共享 FS，文件经 scp/`micnativeloadex`；布局元数据宜小、宜与二进制同打包。  
5. **编程模型分裂**: Host(gcc/icc) / Phi(ICC -mmic) / VE(ncc) 三套 ABI；布局代数统一在 Python，内核分后端生成。  
6. **敏感信息**: mic 序列号、内网 IP、SSH 密钥、license 路径不得进入 git 提交。

---

## 8. 可行性判定签字表

| 检查项 | 结果 |
|--------|------|
| CPU 支持基础方案 | **PASS** |
| GPU 非必需（tensor-layouts） | **PASS**（无 GPU 不阻塞） |
| 算力卡可支撑异构落地 | **PASS**（Phi + 3×VE） |
| 本地隔离 Python 环境可用 | **PASS**（uv + 3.11/3.13） |
| 可立即开始 Phase 1 软件安装 | **PASS**（需用户确认后执行） |

**最终判定**: **可以进入实施方案执行阶段**（建议顺序见 `docs/plan/`）。  
当前仓库仍为空壳；**尚未执行任何软件安装或代码实现**——本文件为实施前门禁记录。

---

## 9. 探测命令备忘（可复现）

```bash
# Host
lscpu; numactl -H; grep -oE 'avx[^ ]*|amx[^ ]*|vnni' /proc/cpuinfo | sort -u
# GPU
nvidia-smi; lspci | grep -iE 'vga|3d|co-processor'
# Phi
micinfo; micctrl -s; ls /dev/mic*
# VE
ls /dev/ve*; ve_exec -V; ncc --version
# Tooling
uv --version; uv python list; podman --version
```

> 复跑时注意：勿把 `micinfo` 序列号、机器私有路径中的凭据写入可公开提交的日志。

# 实现记录 — Intel License 与 Phi 实机 DGEMM

> 文档时间: 2026-07-14 07:02:23

---

## License 结论

> **更正（2026-07-14 文档复核）**：用法与 `intel_phi` / `uni` 历史文档一致  
> （`parallel_studio.lic` + `source compilervars` + `icc -mmic`）。  
> 详细对照见 `docs/research/20260714_080104_intel_license_doc_review.md`。

| 项 | 结果 |
|----|------|
| 用户 license 路径 | `~/parallel_studio.lic`（**不入库**） |
| 文档用法 | 与 intel_phi/uni **一致**（非用错路径） |
| 安装 silent.cfg | `ACTIVATION_TYPE=trial_lic` + license 文件路径 |
| 文件可读 / 已被 ICC 搜索到 | 是 |
| ICC 16 请求 feature | **Comp-CL** |
| `.lic` 内 feature（chklic） | 有 **CCompL** 等，**无 Comp-CL** |
| FlexLM | `-5,357` *No such feature exists* for Comp-CL |
| ICC `-mmic` 当前 | **不可用**（试用 Trusted Storage 失效后，文件无法提供 Comp-CL） |

处理策略:

1. 自动探测 ICC（`try_icc_license()`）— 按文档路径注入 license  
2. 失败则回退 **k1om-mpss-linux-gcc**（与 intel_phi 工具链矩阵中「日常 SDK GCC」一致）  
3. 运行路径: `scp` → `ssh mic0` → `scp` 回传（Phi 进程看不见 Host 路径）

## 新增

| 路径 | 说明 |
|------|------|
| `kernels/phi/dgemm_rect.c` | 矩形 row-major DGEMM，pthread 并行 |
| `backends/phi_dgemm.py` | 编译 / 部署 / 正确性校验 |
| 测试 | `test_phi_dgemm_correctness` |
| 脚本 | `run_real_tests.py` 增加 Phi dgemm 与 license probe |

## 实测结果（本机）

| 用例 | 结果 |
|------|------|
| pytest | **20 passed** |
| multi-VE NLC 384³ 行分片 | pass，err=0，~580–590 GFLOPS/卡 kernel |
| Phi peak smoke | pass，~572 GFLOPS，47.4% theory |
| Phi dgemm 256³ | pass，err ~1e-14，kernel ~0.65 GFLOPS（朴素内核） |

说明: Phi DGEMM 当前是正确性优先的 pthread 朴素实现，未使用 IMCI intrinsics / MKL；峰值仍以 prebuilt peak 二进制为准。

## 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic   # 可选；用于探测 ICC
source env/.venv/bin/activate
pytest -q
python scripts/run_real_tests.py --phi-m 256 --phi-k 256 --phi-n 256
```

## 敏感信息

- license 文件本身 **未** 复制进 git  
- 文档不记录 SN/HOSTID/签名串  

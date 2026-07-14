# 实现记录 — 包名重命名 cpu_cute_tensor → uni_cute_tensor

> 文档时间: 2026-07-14 02:05:32

## 变更

| 旧 | 新 |
|----|-----|
| Python 包目录 `src/cpu_cute_tensor/` | `src/uni_cute_tensor/` |
| import `cpu_cute_tensor.*` | `uni_cute_tensor.*` |
| 发行名 `cpu-cute-tensor` | `uni-cute-tensor` |
| CLI `cct-check-hw` / `cct-demo-partition` | `uct-check-hw` / `uct-demo-partition` |
| 架构文档文件名 `..._cpu_cute_tensor_architecture.md` | `..._uni_cute_tensor_architecture.md` |

文档、脚本、测试、示例中的引用已统一替换。

## 验证

- `import uni_cute_tensor` OK  
- `pytest -q` → 19 passed  

## 未改

- 本地工作区路径 `/mnt/storage/hdd1/cpu-cute-tensor`（磁盘目录名，与 import 无关）  
- GitHub 仓库名 `uni-tensor-layout` 保持不变  

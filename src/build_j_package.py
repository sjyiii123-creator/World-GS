"""Build the J-series results package (J0/J1 real runs + dataset + code).

Contents:
  README.md          core numbers + reproduction commands
  docs/              4 J documents + failure log (F-14)
  code/              3 wmeval modules + 4 scripts + selftest
  data/              actor_sequences dataset (samples/competitions/meta)
  results/           runs/j0_baseline + runs/j1_baseline (incl. cache)
  logs/              selftest output + J1 run output

Weights (88MB) NOT included — curl command documented in README.
Model never inspects images; script only copies files.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j_series_results_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J 系列实验结果包（J0/J1 真实运行）

日期：2026-11-05 | 身份真值：JAAD 官方行人 ID（非 ByteTrack）
复现性：数据集构建 seed=11；阈值只在 train 校准；四套自测 96 项全绿
（logs/selftest_j_output.txt 等）

## 核心数字（test split，n=800）

| 基线 | accuracy | 错误合并率 | 漏关联率 |
|---|---|---|---|
| J0 运动学（全量 15,523 对 acc 0.808） | 0.8150 | 0.1737 | 0.0230 |
| J1 纯外观（cos≥0.6913） | 0.7900 | 0.1338 | 0.1560 |
| **J1 融合（kin ∧ cos）** | **0.8250** | **0.0875** | 0.1790 |

- Mann-Whitney AUC（cos）= **0.8664**；same/diff cos 分布与
  拒识带 [0.60, 0.81]（~49% 真 / ~41% 假）见
  `docs/jepa_experiment_results.md` §2；
- 外观漂移随 gap 单调（same cos 0.825/0.781/0.756 @gap 15/30/45）
  ——J2"预测 gap 后特征"的数据动机；
- J2 触发判据（AUC≥0.8 ✓ 且 merge≤5% ✗）：**未全达，本轮未训练**
  （latent_predictor.py 为可测接口：forward/loss/防泄漏 API）；
- 运行成本：DINOv2-small 8.2ms/帧 @0.29GB VRAM（16GB 富余）。

## 目录导览

```
docs/      jepa_method_analysis（创新核查）/ actor_sequence_dataset（协议+局限）
           jepa_baseline_plan（J0-J3 设计+触发判据）/ jepa_experiment_results（本包主文档）
code/      wmeval/{latent_features,latent_association,latent_predictor}.py
           build_actor_sequences / run_j0_baseline / run_j1_baseline / selftest_j
data/      actor_sequences/（samples.csv 15,523 对 + competitions.jsonl 7,849 + meta.json）
results/   j0_baseline/（pair_results + metrics） j1_baseline/（含特征缓存 feature_cache/）
logs/      selftest_j_output.txt + j1_run_output.txt（完整运行日志）
```

## 复现命令（项目根 /home/moz/World-GS）

```bash
PYTHONPATH=src python src/selftest_j.py                     # 11 项 CPU 自测
python -I src/build_actor_sequences.py                      # 重建数据集（seed=11）
python -I src/run_j0_baseline.py                            # J0 全量
# J1 需先获取权重（88MB，显式代理）：
mkdir -p ckpt/dinov2-small && cd ckpt/dinov2-small
for f in config.json preprocessor_config.json model.safetensors; do
  curl -sS -x http://127.0.0.1:7890 -L -o $f \\
    "https://huggingface.co/facebook/dinov2-small/resolve/main/$f"; done
cd ../.. && python -I src/run_j1_baseline.py                # ~30 分钟 GPU
```

（J1 结果含特征缓存：相同 CropConfig/model 的重跑直接命中缓存。）

## 保护声明

v3.1 实验输出、人工标注表、原始 prompt/视频/生成视频全部未动；
J 系列产物全部位于独立目录（data/results 为副本，原件在项目内）。
J2/J3 无训练/运行结果，未虚构任何数字。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("jepa_method_analysis.md", "actor_sequence_dataset.md",
              "jepa_baseline_plan.md", "jepa_experiment_results.md",
              "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    for f in ("latent_features.py", "latent_association.py",
              "latent_predictor.py"):
        shutil.copy2(ROOT / "src/wmeval" / f, root / "code" / f)
    for f in ("build_actor_sequences.py", "run_j0_baseline.py",
              "run_j1_baseline.py", "selftest_j.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    shutil.copytree(ROOT / "datasets/processed/actor_sequences",
                    root / "data/actor_sequences")

    (root / "results").mkdir()
    shutil.copytree(ROOT / "runs/j0_baseline", root / "results/j0_baseline")
    shutil.copytree(ROOT / "runs/j1_baseline", root / "results/j1_baseline")

    (root / "logs").mkdir()
    for f in ("selftest_j_output.txt", "j1_run_output.txt"):
        p = ROOT / "dist/audit_logs" / f
        if p.exists():
            shutil.copy2(p, root / "logs" / f)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")


if __name__ == "__main__":
    main()

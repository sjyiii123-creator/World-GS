"""E1 minimal experiment: 10 SynPeDAS samples, real + generated sides.

No-argument form (fast-path command shape). Run from project root:
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_e1_smoke.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from wmeval.run_minimal import process_one_video  # noqa: E402

MANIFEST = "datasets/annotations/manifest.csv"
OUTDIR = Path("runs/synpedas_smoke_v3")
WEIGHTS = "ckpt/yolov10x.pt"
NUM_SAMPLES = 10

def main() -> None:
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_video"] and r["generated_video"]:
                rows.append(r)
    rows = rows[:NUM_SAMPLES]
    print(f"[e1] {len(rows)} samples (real+gen each)")

    summary = []
    for r in rows:
        sid = r["sample_id"]
        for side, key in (("real", "source_video"), ("gen", "generated_video")):
            od = OUTDIR / f"{sid}__{side}"
            print(f"[e1] {sid} {side} -> {od}")
            try:
                s = process_one_video(r[key], od, WEIGHTS, imgsz=640,
                                      conf=0.3, target_fps=None,
                                      actor="person", device=0, keyframes=True)
                s["side"] = side
                summary.append(s)
                print(f"[e1] verdict={s['verdict']} "
                      f"person_tracks={s['num_actor_tracks']} "
                      f"consistency={s['traj_consistency']} ({s['elapsed_s']}s)")
            except Exception as e:  # noqa: BLE001
                print(f"[e1] FAILED {sid} {side}: {e}")
                summary.append({"video": r[key], "stem": f"{sid}__{side}",
                                "side": side, "error": str(e)})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for s in summary for k in s})
    with open(OUTDIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(summary)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = [s for s in summary if "error" not in s]
    print(f"\n[e1] done: {len(ok)}/{len(summary)} videos processed")
    for side in ("real", "gen"):
        sub = [s for s in ok if s.get("side") == side]
        if sub:
            with_person = sum(1 for s in sub if s.get("num_actor_tracks", 0) > 0)
            verd: dict = {}
            for s in sub:
                v = s.get("verdict", "?")
                verd[v] = verd.get(v, 0) + 1
            print(f"[e1] {side}: {len(sub)} videos, "
                  f"{with_person} with person tracks, verdicts={verd}")
    print(f"[e1] summary -> {OUTDIR/'summary.csv'}")


if __name__ == "__main__":
    main()

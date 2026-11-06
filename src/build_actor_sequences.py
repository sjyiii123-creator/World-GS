"""J-series: build actor temporal-verification samples from JAAD identity GT.

DATA FACT (measured 2026-11-05): JAAD 2.0 uses CVAT *interpolation*
annotation — bounding boxes exist EVERY frame (0 full-occlusion frames,
0 visibility gaps in the 211 SynPeDAS windows).  Therefore:

  * identity ground truth = JAAD official pedestrian ids (NOT ByteTrack);
  * temporal gaps are CONSTRUCTED by removing delta frames of observation
    from a continuous track (synthetic gap, protocol-flagged) — this
    simulates the tracker-loss/occlusion fragmentation we must repair,
    while labels remain REAL identity;
  * negative pairs / competition candidates are REAL co-occurring
    pedestrians inside the same clip at the candidate time.

Sample types
  A (positive, same_identity)  : anchor [t-11..t] vs candidate
                                 [t+Δ..t+Δ+11], same ped, Δ ∈ {15,30,45}
  B (negative, diff_identity)  : anchor vs another ped's first observation
                                 within [t+Δ-3, t+Δ+15]
  C (competition set)          : anchor + ALL peds observed in the window
                                 (ranking task, includes the true match)

Time: real timestamps from each clip's fps (manifest real_fps — the
211 clips mix 29.97/59.94; frame indices are never compared directly).
Split: JAAD official split_ids (train→train, test→test held-out).

Output: datasets/processed/actor_sequences/{samples.csv,
competitions.jsonl, meta.json}.  Pixel crops are NOT pre-extracted;
feature stage reads frames on demand (bbox+frame stored here).
"""
from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

JAAD = Path("repos/JAAD/annotations")
SPLIT = Path("repos/JAAD/split_ids/default")
MANIFEST = Path("datasets/annotations/manifest.csv")
OUT = Path("datasets/processed/actor_sequences")

ANCHOR_LEN = 12          # frames of anchor history
CAND_LEN = 12            # frames of candidate observation
GAPS = (15, 30, 45)      # synthetic gap (frames) — simulates tracker loss
NEG_WINDOW = (-3, 15)    # candidate search window around t+Δ
MAX_ANCHORS_PER_PED = 2
SEED = 11


def load_split() -> dict:
    sp = {}
    for name in ("train", "val", "test"):
        p = SPLIT / f"{name}.txt"
        if p.exists():
            for line in p.read_text().split():
                sp[line.strip()] = name
    return sp


def ped_tracks(xml_path: Path, f0: int, f1: int) -> dict:
    """{ped_id: [(frame, bbox_xyxy_px, occ), ...]} inside window, sorted."""
    root = ET.parse(xml_path).getroot()
    out: dict = {}
    for box in root.iter("box"):
        pid = next((a.text for a in box.iter("attribute")
                    if a.get("name") == "id"), None)
        if not pid:
            continue
        try:
            fi = int(box.get("frame"))
        except (TypeError, ValueError):
            continue
        if not (f0 <= fi <= f1):
            continue
        bbox = (float(box.get("xtl")), float(box.get("ytl")),
                float(box.get("xbr")), float(box.get("ybr")))
        out.setdefault(pid, []).append(
            (fi, bbox, int(box.get("occluded", "0"))))
    for pid in out:
        out[pid].sort()
    return out


def seg(track, t, length, direction):
    """frames [t, t+length) (forward) or [t-length+1, t] (backward)."""
    if direction < 0:
        lo, hi = t - length + 1, t
    else:
        lo, hi = t, t + length - 1
    return [rec for rec in track if lo <= rec[0] <= hi]


def main() -> None:
    random.seed(SEED)
    split = load_split()
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    samples, competitions = [], []
    n_pos = n_neg = 0

    for r in rows:
        sid = r["sample_id"]
        m = re.match(r"video_(\d{4})_", sid)
        stem = f"video_{m.group(1)}"
        f0, f1 = (int(x) for x in sid.split("_f")[-1].split("-"))
        fps = float(r["real_fps"])
        xp = JAAD / f"{stem}.xml"
        if not xp.exists():
            continue
        tracks = ped_tracks(xp, f0, f1)
        # candidate pool per frame for negatives/competitions
        by_frame = {}
        for pid, tr in tracks.items():
            for fi, bbox, occ in tr:
                by_frame.setdefault(fi, []).append((pid, bbox, occ))

        sp = split.get(stem, "train")
        for pid, tr in tracks.items():
            if len(tr) < ANCHOR_LEN + max(GAPS) + CAND_LEN + 2:
                continue
            frames = [rec[0] for rec in tr]
            # candidate anchor times: interior, leaves room for gap+seg
            lo = frames[0] + ANCHOR_LEN
            hi = frames[-1] - max(GAPS) - CAND_LEN
            if hi <= lo:
                continue
            ts = sorted(random.sample(range(lo, hi + 1),
                                      min(MAX_ANCHORS_PER_PED, hi - lo + 1)))
            for t in ts:
                for delta in GAPS:
                    anchor = seg(tr, t, ANCHOR_LEN, -1)
                    cand = seg(tr, t + delta, CAND_LEN, +1)
                    if len(anchor) < ANCHOR_LEN // 2 or len(cand) < CAND_LEN // 2:
                        continue
                    gap_s = round((t + delta - anchor[-1][0]) / fps, 3)
                    row = {
                        "pair_id": len(samples),
                        "synpedas_sample": sid, "jaad_video": stem,
                        "split": sp, "fps": fps,
                        "anchor_ped_id": pid, "cand_ped_id": pid,
                        "label": "same_identity",
                        "gap_frames": delta, "gap_seconds": gap_s,
                        "synthetic_gap": "yes",
                        "anchor_frames": json.dumps(
                            [[f, list(b), o] for f, b, o in anchor]),
                        "cand_frames": json.dumps(
                            [[f, list(b), o] for f, b, o in cand]),
                    }
                    samples.append(row)
                    n_pos += 1
                    # --- competition set C: all peds near t+delta -----
                    c_lo = t + delta + NEG_WINDOW[0]
                    c_hi = t + delta + NEG_WINDOW[1]
                    seen, cands = set(), []
                    for fi in range(c_lo, c_hi + 1):
                        for cpid, bbox, occ in by_frame.get(fi, []):
                            if cpid not in seen:
                                seen.add(cpid)
                                cands.append({
                                    "cand_ped_id": cpid,
                                    "first_frame": fi,
                                    "bbox": list(bbox),
                                    "is_true_match": cpid == pid})
                    if len(cands) >= 2:
                        competitions.append({
                            "comp_id": len(competitions),
                            "pair_id": row["pair_id"],
                            "synpedas_sample": sid, "split": sp,
                            "anchor_ped_id": pid, "gap_frames": delta,
                            "candidates": cands})
                    # --- negative pair B: one other ped, time-matched ----
                    others = sorted(set(seen) - {pid})
                    if others:
                        q = random.choice(others)
                        qtr = tracks[q]
                        qseg = seg(qtr, t + delta, CAND_LEN, +1)
                        if len(qseg) < CAND_LEN // 2:
                            qseg = seg(qtr, t + delta, CAND_LEN + 6, +1)[:CAND_LEN]
                        if len(qseg) >= CAND_LEN // 2:
                            samples.append({
                                "pair_id": len(samples),
                                "synpedas_sample": sid, "jaad_video": stem,
                                "split": sp, "fps": fps,
                                "anchor_ped_id": pid, "cand_ped_id": q,
                                "label": "different_identity",
                                "gap_frames": delta, "gap_seconds": gap_s,
                                "synthetic_gap": "yes",
                                "anchor_frames": json.dumps(
                                    [[f, list(b), o] for f, b, o in anchor]),
                                "cand_frames": json.dumps(
                                    [[f, list(b), o] for f, b, o in qseg]),
                            })
                            n_neg += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        w.writeheader()
        w.writerows(samples)
    with open(OUT / "competitions.jsonl", "w", encoding="utf-8") as f:
        for c in competitions:
            f.write(json.dumps(c) + "\n")
    meta = {
        "protocol": "synthetic-gap identity verification from JAAD GT",
        "data_fact": "JAAD interpolation annotation has 0 visibility gaps; "
                     "gaps are constructed by observation removal; identity "
                     "labels are REAL JAAD ped ids",
        "anchor_len": ANCHOR_LEN, "cand_len": CAND_LEN, "gaps": GAPS,
        "neg_window": NEG_WINDOW, "seed": SEED,
        "n_positive": n_pos, "n_negative": n_neg,
        "n_competitions": len(competitions),
        "splits": {"train": "JAAD official train", "test": "JAAD official test"},
        "time_basis": "real timestamps = frame / clip fps (29.97/59.94 mix)",
        "pixel_source": "repos/SynPeDAS/JAAD_crossing_moments (clip frame = "
                        "xml frame - window start; same 1920x1080 coords)",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    from collections import Counter
    print(f"[seq] samples={len(samples)} (pos {n_pos} / neg {n_neg}); "
          f"competitions={len(competitions)}")
    print(f"[seq] split distribution: {dict(Counter(s['split'] for s in samples))}")
    print(f"[seq] gap distribution: {dict(Counter(s['gap_frames'] for s in samples))}")
    print(f"[seq] -> {OUT}")


if __name__ == "__main__":
    main()

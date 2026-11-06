"""Generate manual-review contact sheets for suspicious E1 v3 samples.

Pure local OpenCV work: draws detection boxes (from v3 detections.jsonl) on
real/generated frames aligned by normalized time t_norm, and writes contact
sheets + standalone annotated keyframes under runs/manual_review/.

NO image content is ever sent to any model — outputs are file paths only.

Run from project root:
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/make_manual_review.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RUNS = Path("runs/synpedas_smoke_v3")
OUT = Path("runs/manual_review")
OUT.mkdir(parents=True, exist_ok=True)

CELL_W = 640
HDR_H = 34
PAD = 6

# highlight palette for the 0016 id-fragmentation chain
SPECIAL_COLORS = {30: (255, 80, 0), 64: (0, 140, 255)}   # BGR
PERSON_COLOR = (0, 0, 255)
OTHER_COLOR = (110, 110, 110)


def load_dets(run_dir: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for line in (run_dir / "detections.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        by_frame.setdefault(d["frame_index"], []).append(d)
    return by_frame


def get_frame(path: str, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def draw_boxes(frame: np.ndarray, dets: list[dict],
               special: set[int] | None = None) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    special = special or set()
    for d in dets:
        x1, y1, x2, y2 = d["norm_bbox"]
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
        tid = d["track_id"]
        if d["category"] == "person":
            color = SPECIAL_COLORS.get(tid, PERSON_COLOR) if tid in special else PERSON_COLOR
            thick = 3 if tid in special else 2
        else:
            color, thick = OTHER_COLOR, 1
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        label = f"t{tid} {d['category'][:4]} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = int(y1) - th - 4 if int(y1) - th - 4 > 0 else int(y2) + th + 4
        cv2.rectangle(out, (int(x1), ly - 2), (int(x1) + tw + 4, ly + th + 2),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (int(x1) + 2, ly + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def cell(frame: np.ndarray, title: str) -> np.ndarray:
    h = frame.shape[0]
    scale = CELL_W / frame.shape[1]
    resized = cv2.resize(frame, (CELL_W, int(h * scale)))
    ch = resized.shape[0] + HDR_H
    canvas = np.full((ch, CELL_W, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas[HDR_H:, :] = resized
    return canvas


def contact_sheet(rows: list[tuple], out_path: Path) -> None:
    """rows: list of (left_title, left_img, right_title, right_img)."""
    n = len(rows)
    cell_h = max(max(l.shape[0], r.shape[0]) for _, l, _, r in rows)
    W = CELL_W * 2 + PAD * 3
    H = n * (cell_h + PAD) + PAD + 30
    sheet = np.full((H, W, 3), 24, dtype=np.uint8)
    for i, (lt, l, rt, r) in enumerate(rows):
        y = PAD + i * (cell_h + PAD)
        sheet[y:y + l.shape[0], PAD:PAD + CELL_W] = l
        sheet[y:y + r.shape[0], PAD * 2 + CELL_W:PAD * 2 + CELL_W * 2] = r
    cv2.putText(sheet, out_path.stem.replace("_", " ") +
                "  |  LEFT = real JAAD, RIGHT = generated  |  aligned by t_norm",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    ok = cv2.imwrite(str(out_path), sheet,
                     [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(("OK  " if ok else "FAIL") + f" {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")


def standalone(frame: np.ndarray, dets: list[dict], out_path: Path,
               special: set[int] | None = None) -> None:
    if frame is None:
        print(f"FAIL {out_path} (frame unreadable)")
        return
    ann = draw_boxes(frame, dets, special)
    ok = cv2.imwrite(str(out_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(("OK  " if ok else "FAIL") + f" {out_path}")


def t_norm(idx: int, n: int) -> float:
    return idx / max(n - 1, 1)


# =========================== sample 0016 ====================================
def review_0016() -> None:
    sid = "video_0016_0_16_72b_f0-150"
    gen_dir = RUNS / f"{sid}__gen"
    real_dir = RUNS / f"{sid}__real"
    gen_vid = "repos/SynPeDAS/generated_video/video_0016_0_16_72b_f0-150.mp4"
    real_vid = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
    gd, rd = load_dets(gen_dir), load_dets(real_dir)
    g_n, r_n = 93, 150

    # timepoints chosen around the track30->track64 fragmentation chain
    tps = [  # (gen_frame, note)
        (12, "t30 first seen"),
        (20, "early reference"),
        (45, "t64 segment starts"),
        (54, "t64 last before gap"),
        (55, "t30 re-acquires (gap!)"),
        (62, "t30 LAST frame"),
        (63, "t64 takes over (gap=1)"),
        (70, "t64 continues"),
        (92, "last frame"),
    ]
    special = {30, 64}
    rows = []
    for gf, note in tps:
        t = t_norm(gf, g_n)
        rf = min(round(t * (r_n - 1)), r_n - 1)
        gframe = get_frame(gen_vid, gf)
        rframe = get_frame(real_vid, rf)
        if gframe is None or rframe is None:
            print(f"skip t={t:.3f}: frame read failed (gen {gf}, real {rf})")
            continue
        g_ann = draw_boxes(gframe, gd.get(gf, []), special)
        r_ann = draw_boxes(rframe, rd.get(rf, []), special)
        gt = t_norm(gf, g_n) * 5.8125
        rt = t_norm(rf, r_n) * 5.005
        rows.append((
            f"REAL  f{rf}  t={t:.3f} ({rt:.2f}s)", cell(r_ann, ""),
            f"GEN  f{gf}  t={t:.3f} ({gt:.2f}s)  {note}", cell(g_ann, ""),
        ))
    contact_sheet(rows, OUT / "0016_contact_sheet.jpg")

    # standalone annotated keyframes for the chain
    for gf in (12, 45, 54, 55, 62, 63, 92):
        fr = get_frame(gen_vid, gf)
        standalone(fr, gd.get(gf, []),
                   OUT / f"0016_gen_f{gf:03d}_boxes.jpg", special)
    for rf in (19, 88, 89, 100, 101, 149):
        fr = get_frame(real_vid, rf)
        standalone(fr, rd.get(rf, []),
                   OUT / f"0016_real_f{rf:03d}_boxes.jpg", special)

    # alignment record
    rec = {"sample": sid,
           "alignment": "t_norm = frame/(num_frames-1); gen 93f@16fps=5.8125s, real 150f@29.97fps=5.005s",
           "timepoints": [{"gen_frame": gf, "t_norm": round(t_norm(gf, g_n), 4),
                           "real_frame": min(round(t_norm(gf, g_n) * (r_n - 1)), r_n - 1),
                           "note": note} for gf, note in tps]}
    (OUT / "0016_alignment.json").write_text(json.dumps(rec, indent=2))
    print("OK  " + str(OUT / "0016_alignment.json"))


# =========================== sample 0024 ====================================
def review_0024() -> None:
    sid = "video_0024_0_24_99b_f60-210"
    gen_dir = RUNS / f"{sid}__gen"
    real_dir = RUNS / f"{sid}__real"
    gen_vid = f"repos/SynPeDAS/generated_video/{sid}.mp4"
    real_vid = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
    gd, rd = load_dets(gen_dir), load_dets(real_dir)
    g_n, r_n = 93, 150

    # anchor on the real-side track21 single detection at f107
    tps = [  # (real_frame, note)
        (0, "clip start"),
        (48, "mid-clip"),
        (97, "near track21"),
        (106, "track21 prev frame"),
        (107, "track21 ONLY detection"),
        (108, "track21 next frame"),
        (130, "late clip"),
        (149, "last frame"),
    ]
    rows = []
    for rf, note in tps:
        t = t_norm(rf, r_n)
        gf = min(round(t * (g_n - 1)), g_n - 1)
        gframe = get_frame(gen_vid, gf)
        rframe = get_frame(real_vid, rf)
        if gframe is None or rframe is None:
            print(f"skip t={t:.3f}: frame read failed")
            continue
        g_ann = draw_boxes(gframe, gd.get(gf, []))
        r_ann = draw_boxes(rframe, rd.get(rf, []))
        gt = t_norm(gf, g_n) * 5.8125
        rt = t_norm(rf, r_n) * 5.005
        rows.append((
            f"REAL  f{rf}  t={t:.3f} ({rt:.2f}s)  {note}", cell(r_ann, ""),
            f"GEN  f{gf}  t={t:.3f} ({gt:.2f}s)", cell(g_ann, ""),
        ))
    contact_sheet(rows, OUT / "0024_contact_sheet.jpg")

    for rf in (106, 107, 108):
        fr = get_frame(real_vid, rf)
        standalone(fr, rd.get(rf, []), OUT / f"0024_real_f{rf:03d}_boxes.jpg")
    for gf in (0, 30, 66, 92):
        fr = get_frame(gen_vid, gf)
        standalone(fr, gd.get(gf, []), OUT / f"0024_gen_f{gf:03d}_boxes.jpg")

    rec = {"sample": sid,
           "alignment": "t_norm = frame/(num_frames-1); gen 93f@16fps=5.8125s, real 150f@29.97fps=5.005s",
           "timepoints": [{"real_frame": rf, "t_norm": round(t_norm(rf, r_n), 4),
                           "gen_frame": min(round(t_norm(rf, r_n) * (g_n - 1)), g_n - 1),
                           "note": note} for rf, note in tps]}
    (OUT / "0024_alignment.json").write_text(json.dumps(rec, indent=2))
    print("OK  " + str(OUT / "0024_alignment.json"))


if __name__ == "__main__":
    review_0016()
    print("-" * 60)
    review_0024()
    print("done. outputs in", OUT.resolve())

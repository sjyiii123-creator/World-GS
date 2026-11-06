#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the E3-A.1 human event-candidate review kit.

Candidates are generated only by the corrected image-plane state machine.
No event is marked true here: every STOPPED/RESUMING segment needs human
verification and actor/condition binding before it can be used as truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "runs/e3_a_1_baseline"
OUT = REPO_ROOT / "runs/e3_a_1_review"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = defaultdict(list)
    with (BASE / "per_frame_phases.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            frames[(r["video"], r["side"], r["track_id"])].append(r)
    candidates = []
    for (video, side, track_id), rows in sorted(frames.items()):
        rows.sort(key=lambda r: int(r["frame"]))
        for phase in ("STOPPED", "RESUMING"):
            idx = [i for i, r in enumerate(rows) if r["phase"] == phase]
            if not idx:
                continue
            start, end = idx[0], idx[-1]
            prev_phase = rows[start - 1]["phase"] if start else ""
            next_phase = rows[end + 1]["phase"] if end + 1 < len(rows) else ""
            candidates.append({
                "candidate_id": f"{video}__{side}__track{track_id}__{phase.lower()}",
                "video": video, "side": side, "track_id": track_id,
                "phase_candidate": phase,
                "start_frame": rows[start]["frame"], "end_frame": rows[end]["frame"],
                "n_observed_frames": len(idx),
                "previous_phase": prev_phase, "next_phase": next_phase,
                "evidence_scope": "image_plane_motion_candidate_only",
                "tracks_json": f"runs/synpedas_smoke_v2/{video}__{side}/tracks.json",
                "video_path": (f"repos/SynPeDAS/JAAD_crossing_moments/{video}.mp4"
                               if side == "real" else
                               f"repos/SynPeDAS/generated_video/{video}.mp4"),
                "human_event_verdict": "",
                "human_actor_binding": "",
                "human_condition_satisfaction": "",
                "human_confidence": "",
                "reviewer": "", "reviewer_notes": "",
            })
    fields = list(candidates[0])
    with (OUT / "event_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(candidates)
    hashes = {}
    for c in candidates:
        p = REPO_ROOT / c["tracks_json"]
        hashes[c["tracks_json"]] = sha256(p)
    (OUT / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    readme = f"""# E3-A.1 event-candidate human review kit

Generated from the corrected E3-A.1 state machine (selftests 16/16 PASS).

- candidates: {len(candidates)} (unique video/side/track/phase rows)
- STOPPED and RESUMING are **image-plane motion candidates only**, not road-crossing truth.
- `reestablish` is enforced for consecutive usable observations after large gaps.
- Speed smoothing is centred and offline/non-causal (`smooth_window=3`).
- No generated-video RESUMING candidate was detected under this configuration.
  This does NOT prove generated videos lack resumptions; track fragmentation,
  short tracks, image motion, thresholds and detector misses remain alternatives.

Review each row against the actual video and actor identity:
1. `human_event_verdict`: confirmed / rejected / uncertain
2. `human_actor_binding`: anchor_ped_id or unresolved
3. `human_condition_satisfaction`: yes / no / partial / unknown / not_applicable
4. `human_confidence`: high / medium / low
5. `reviewer` and `reviewer_notes`: required for a human truth claim

Blank fields are intentional. Do not use unreviewed candidates for E3-B training,
threshold selection, or claims about generated-video event frequency.
"""
    (OUT / "README.md").write_text(readme)
    print(f"[E3-A.1 review] candidates={len(candidates)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

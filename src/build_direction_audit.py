"""Build the E2 prompt–source–generation direction audit tables.

Deliverables (task brief §七):
  1. datasets/annotations/prompt_source_generation_audit.csv
  2. datasets/annotations/prompt_correction_sidecar.csv (0044 only)

Anti-conflation rules baked in (see wmeval.direction_audit):
  * semantic_direction_reversal stays `unknown` for all 8 samples
    (0035 pixel-sign reversal is an observation, not semantics);
  * 0044 is quarantined as PROMPT_SOURCE_CONFLICT — excluded from clean
    prompt-adherence statistics, prompt never rewritten;
  * actor matches stay unknown (max-detections track != prompted actor);
  * event_required stays unknown pending human protocol review
    (the prompt-side pedestrian-condition FACT is recorded separately);
  * cross-video comparison notes fps/frame incompatibility.

Inputs are READ-ONLY (raw prompts, v3.1 outputs).  External independent
review (WorldGS_E2_Review) conclusions are merged as references.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

from wmeval.direction_audit import (
    SideEvidence, assess_direction, bbox_height_ratio,
    frame_index_comparable, pixel_direction, prompt_direction_semantics,
    scale_trend, sha256_file)

SIDS = ["video_0002_0_2_6b_f60-210", "video_0003_0_3_7b_f0-150",
        "video_0016_0_16_72b_f0-150", "video_0024_0_24_99b_f60-210",
        "video_0030_0_30_134b_f0-150", "video_0035_0_35_163b_f0-150",
        "video_0037_0_37_168b_f0-150", "video_0044_0_44_202b_f60-210"]
PROMPT_DIR = Path("repos/SynPeDAS/video_generation_prompts")
V31 = Path("runs/synpedas_smoke_v3_1")
EXT = Path("WorldGS_E2_Review")
OUT_MAIN = Path("datasets/annotations/prompt_source_generation_audit_v2.csv")
OUT_SIDECAR = Path("datasets/annotations/prompt_correction_sidecar_v2.csv")

# external independent-review statuses (merged as REFERENCE, not truth)
EXT_STATUS = {r["sample_id"]: r for r in csv.DictReader(
    open(EXT / "direction_review_8_samples.csv", encoding="utf-8-sig"))}
EXT_SIDECAR = {r["sample_id"]: r for r in csv.DictReader(
    open(EXT / "prompt_correction_sidecar.csv", encoding="utf-8-sig"))}


def side_ev(sid: str, side: str) -> SideEvidence:
    d = V31 / f"{sid}__{side}"
    tracks = json.loads((d / "tracks.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    persons = [t for t in tracks if t["category"] == "person"]
    if not persons:
        return SideEvidence(None, "unknown", None, "unknown", 0, "none",
                            meta["fps"], meta["num_frames"])
    t = max(persons, key=lambda x: x["num_detections"])
    cs = t["norm_centers"]
    dx = cs[-1][0] - cs[0][0]
    fh = []
    for line in (d / "detections.jsonl").read_text().splitlines():
        if line.strip():
            det = json.loads(line)
            if det["track_id"] == t["track_id"] and det["category"] == "person":
                nb = det["norm_bbox"]
                fh.append((det["frame_index"], nb[3] - nb[1]))
    _, _, ratio = bbox_height_ratio(fh)  # P0-1: TIME-ordered fix
    return SideEvidence(round(dx, 4), pixel_direction(dx), ratio,
                        scale_trend(ratio), len(persons),
                        "max_detections", meta["fps"], meta["num_frames"])


def main() -> None:
    rows = []
    for sid in SIDS:
        prompt_text = (PROMPT_DIR / f"{sid}.txt").read_text(errors="replace")
        sem = prompt_direction_semantics(prompt_text)
        sha = sha256_file(PROMPT_DIR / f"{sid}.txt")
        real, gen = side_ev(sid, "real"), side_ev(sid, "gen")
        # source actor match: never auto-confirmed; 0044 real has exactly 1
        # person track AND external visual review — still 'possible',
        # final confirmation belongs to human review
        src_actor = "possible_single_track_external_visual_review" \
            if real.n_person_tracks == 1 else "unknown"
        gen_actor = "unknown"
        a = assess_direction(sem, real, gen, source_actor_match=src_actor,
                             gen_actor_match=gen_actor,
                             camera_motion_uncertainty="unknown")
        ext = EXT_STATUS.get(sid, {})
        row = {
            "sample_id": sid,
            "raw_prompt_sha256": sha,
            # ---- E2.1 v2 layered protocol fields (P0-5) ----
            "prompt_requirement": f'{sem["direction"]};{sem["semantic_type"]}',
            "prompt_direction_raw": sem["direction"],
            "prompt_semantic_type": sem["semantic_type"],
            "prompt_direction_phrases": "; ".join(sem["matched_phrases"]),
            "prompt_pedestrian_condition": sem["pedestrian_condition"],
            "source_direction_observed": real.pixel_dir,
            "source_bbox_scale_trend": real.scale,
            "source_actor_match": src_actor,
            "generated_direction_observed": gen.pixel_dir,
            "generated_bbox_scale_trend": gen.scale,
            "generated_actor_match": gen_actor,
            "prompt_source_conflict": a.prompt_source_conflict,
            "pixel_horizontal_direction_mismatch": a.pixel_horizontal_mismatch,
            "semantic_direction_reversal": a.semantic_direction_reversal,
            "camera_motion_uncertainty": "unknown",
            "frame_index_comparable": str(frame_index_comparable(
                {"fps": real.fps, "num_frames": real.num_frames},
                {"fps": gen.fps, "num_frames": gen.num_frames})),
            "event_required": "unknown",
            "event_satisfied": "not_assessed",
            "failure_attribution": a.event_adjudication,
            "audit_category": a.category,
            "dx_pixel_norm_real": real.dx_norm if real.dx_norm is not None else "",
            "dx_pixel_norm_gen": gen.dx_norm if gen.dx_norm is not None else "",
            "bbox_height_ratio_real": real.height_ratio if real.height_ratio else "",
            "bbox_height_ratio_gen": gen.height_ratio if gen.height_ratio else "",
            "n_person_tracks_real": real.n_person_tracks,
            "n_person_tracks_gen": gen.n_person_tracks,
            "fps_real": real.fps, "fps_gen": gen.fps,
            "num_frames_real": real.num_frames, "num_frames_gen": gen.num_frames,
            "track_selection_basis": real.track_selection_basis,
            "evidence_notes": " | ".join(a.reasons),
            "external_review_status": ext.get("audit_status", ""),
            "external_review_reference": "WorldGS_E2_Review/direction_review_8_samples.csv",
            # ---- P0-2 candidate/confirmed split ----
            "prompt_source_conflict_candidate": a.prompt_source_conflict_candidate,
            "prompt_source_conflict_confirmed": a.prompt_source_conflict_confirmed,
            "evidence_level": a.conflict_evidence_level,
            # ---- P0-3 two-level agreement ----
            "pixel_direction_agreement": a.pixel_direction_agreement,
            "actor_level_direction_agreement": a.actor_level_direction_agreement,
            # ---- P0-5 protocol layering ----
            "source_observation": f'{real.pixel_dir};scale={real.scale};dx={real.dx_norm}',
            "generated_observation": f'{gen.pixel_dir};scale={gen.scale};dx={gen.dx_norm}',
            "condition_source": "prompt_text+real_video_first~16f(video2world)",
            "conditioning_frames_scope": "likely_first_~16_frames_replayed_then_77_generated",
            "camera_motion_status": a.camera_motion_status,
            "event_applicability": a.event_applicability,
            "event_satisfaction": "not_assessed",
            "review_required": a.review_required,
            "human_review_status": "pending",
        }
        rows.append(row)

    with open(OUT_MAIN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[audit] wrote {len(rows)} rows -> {OUT_MAIN}")

    # ---- sidecar: 0044 only (source-description suggestion, DERIVED) -----
    sid44 = "video_0044_0_44_202b_f60-210"
    ext_side = EXT_SIDECAR[sid44]
    sidecar_row = {
        "sample_id": sid44,
        "original_text": ext_side["original_text"],
        "original_prompt_sha256": sha256_file(PROMPT_DIR / f"{sid44}.txt"),
        "source_descriptive_candidate": ext_side["source_descriptive_candidate"],
        "use_for_original_prompt_compliance": "NO",
        "change_type": "SOURCE_DESCRIPTION_ONLY_DERIVED_NOT_ORIGINAL_PROMPT",
        "source_evidence": "real main-person dx=-0.518 (image_right->image_left), "
                           "150f@29.97; external visual review confirms same "
                           "direction in sampled frames",
        "reason": "Original prompt states left-to-right; source video and "
                  "generated clip both move image-right-to-image-left. Both "
                  "prompt and source kept immutable; conflict recorded in "
                  "prompt_source_generation_audit.csv. Candidate text is a "
                  "RESEARCHER-DERIVED source description, ineligible for "
                  "unmodified-prompt compliance scoring.",
        "never_overwrite_original": "YES",
        "approval": "pending",
    }
    with open(OUT_SIDECAR, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sidecar_row.keys()))
        w.writeheader()
        w.writerow(sidecar_row)
    print(f"[audit] wrote sidecar -> {OUT_SIDECAR}")

    from collections import Counter
    cats = Counter(r["audit_category"] for r in rows)
    print("[audit] category distribution:", dict(cats))


if __name__ == "__main__":
    main()

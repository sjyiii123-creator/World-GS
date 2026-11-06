#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation and event-instance helpers for the E3-VLM review pipeline."""
from __future__ import annotations

from collections import defaultdict

OBSERVATION_STATES = {
    "IMAGE_PLANE_MOVING", "IMAGE_PLANE_STILL", "POSSIBLE_STOP",
    "POSSIBLE_RESUME", "DIRECTION_CHANGE_CANDIDATE", "OCCLUDED",
    "OUT_OF_VIEW", "UNKNOWN",
}
VERDICTS = {"PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"}


def event_segments(per_frame_rows: list[dict]) -> list[dict]:
    """One event_id per contiguous STOPPED/RESUMING segment (never merged)."""
    groups = defaultdict(list)
    for row in per_frame_rows:
        groups[(row["video"], row["side"], str(row["track_id"]))].append(row)
    events = []
    for (video, side, track), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["frame"]))
        current = None
        for row in rows + [None]:
            phase = row["phase"] if row else None
            if current is not None and phase != current["phase"]:
                start, end = current["rows"][0], current["rows"][-1]
                events.append({
                    "event_id": f"{video}__{side}__track{track}__"
                                f"{current['phase'].lower()}__f{start['frame']}",
                    "video": video, "side": side, "track_id": track,
                    "phase": current["phase"],
                    "start_frame": int(start["frame"]), "end_frame": int(end["frame"]),
                    "observed_frames": [int(r["frame"]) for r in current["rows"]],
                })
                current = None
            if phase in ("STOPPED", "RESUMING"):
                if current is None:
                    current = {"phase": phase, "rows": []}
                current["rows"].append(row)
    return events


def condition_verdict(condition_required: str, evidence_status: str,
                      actor_bound: bool = True) -> str:
    """Combine independent condition parse and observation, without guessing."""
    if condition_required not in ("yes", "no", "unknown"):
        raise ValueError("condition_required must be yes/no/unknown")
    if evidence_status not in ("satisfied", "not_satisfied", "insufficient"):
        raise ValueError("invalid evidence_status")
    if condition_required == "no":
        return "NOT_APPLICABLE"
    if condition_required == "unknown" or not actor_bound:
        return "UNKNOWN"
    if evidence_status == "satisfied":
        return "PASS"
    if evidence_status == "not_satisfied":
        return "FAIL"
    return "UNKNOWN"


def validate_observation(record: dict) -> None:
    required = ("sample_id", "source_type", "review_status", "observations",
                "evidence_ids", "unknown_reasons")
    for key in required:
        if key not in record:
            raise ValueError(f"missing observation field: {key}")
    if record["review_status"] != "VLM_PROPOSED_ONLY":
        raise ValueError("VLM output must remain VLM_PROPOSED_ONLY")
    if not record["evidence_ids"]:
        raise ValueError("visual observation requires at least one evidence ID")
    for item in record["observations"]:
        if item.get("state") not in OBSERVATION_STATES:
            raise ValueError(f"invalid state: {item.get('state')}")
        if not item.get("frame_indices"):
            raise ValueError("observation state lacks frame_indices")

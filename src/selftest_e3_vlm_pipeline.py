#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for E3-VLM event segmentation, observation validation and verdicts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_vlm_review_utils as u


def main() -> int:
    tests = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        tests.append((name, ok, detail))

    def rows(phases):
        return [{"video": "v", "side": "real", "track_id": "7",
                 "frame": i, "phase": p} for i, p in enumerate(phases)]

    def t1():
        events = u.event_segments(rows(["CROSSING", "STOPPED", "STOPPED",
                                        "CROSSING", "STOPPED", "CROSSING"]))
        return len(events) == 2 and events[0]["start_frame"] == 1 and events[1]["start_frame"] == 4, \
               f"two independent STOPPED instances: {[e['event_id'] for e in events]}"
    check("V1_separate_event_instances", t1)

    def t2():
        events = u.event_segments(rows(["STOPPED", "RESUMING", "CROSSING"]))
        return [e["phase"] for e in events] == ["STOPPED", "RESUMING"], \
               "STOPPED and RESUMING get separate IDs"
    check("V2_stop_resume_not_merged", t2)

    def t3():
        record = {"sample_id": "v", "source_type": "real",
                  "review_status": "VLM_PROPOSED_ONLY",
                  "observations": [{"state": "UNKNOWN", "frame_indices": [1, 2]}],
                  "evidence_ids": ["e1"], "unknown_reasons": ["test"]}
        u.validate_observation(record)
        record["observations"][0]["state"] = "CROSSING"
        try:
            u.validate_observation(record)
            return False, "invalid CROSSING state was accepted"
        except ValueError:
            return True, "schema enforces evidence IDs and allowed states"
    check("V3_observation_schema", t3)

    def t4():
        got = [u.condition_verdict("no", "not_satisfied"),
               u.condition_verdict("yes", "satisfied"),
               u.condition_verdict("yes", "not_satisfied"),
               u.condition_verdict("yes", "insufficient"),
               u.condition_verdict("yes", "satisfied", actor_bound=False),
               u.condition_verdict("unknown", "satisfied")]
        return got == ["NOT_APPLICABLE", "PASS", "FAIL", "UNKNOWN", "UNKNOWN", "UNKNOWN"], str(got)
    check("V4_condition_verdict_rules", t4)

    n = sum(ok for _, ok, _ in tests)
    for name, ok, detail in tests:
        print(f"{name}: {'PASS' if ok else 'FAIL'} — {detail}")
    print(f"SUMMARY: {n}/{len(tests)} PASS")
    return 0 if n == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())

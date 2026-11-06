"""E2 Phase 8: deterministic, leakage-safe Development/Validation/Test split.

Grouping rule: ALL samples from one source JAAD video stay in the SAME
split (19 source videos contribute 2-6 samples; splitting them would leak).

Contamination rule: the 10 E1 rule-development samples (v1-v3.1 tuning)
are pinned to Development — they can never serve as independent Test.

Determinism: sha256(source_group) hashing with a fixed salt — no RNG.

Stratification (mechanical proxies only, human review pending):
  * condition-clarity proxy: auto_prompt_crossing_phrase present or not
    (from event_protocol.csv) — Validation must contain both strata;
  * the 7 no-person-prompt samples are flagged for the protocol-review
    queue (0024-type "condition ambiguity" stratum).

Output: datasets/annotations/e2_split_proposal.csv — a PROPOSAL: every row
carries review_status=pending; the human may re-assign before freezing.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

MANIFEST = "datasets/annotations/manifest.csv"
PROTOCOL = "datasets/annotations/event_protocol.csv"
OUT = "datasets/annotations/e2_split_proposal.csv"
SALT = "wmeval-e2-split-v1"

E1_DEV_SAMPLES = [
    "video_0002_0_2_6b_f60-210", "video_0003_0_3_7b_f0-150",
    "video_0016_0_16_72b_f0-150", "video_0020_0_20_86b_f381-531",
    "video_0024_0_24_99b_f60-210", "video_0027_0_27_119b_f0-150",
    "video_0030_0_30_134b_f0-150", "video_0035_0_35_163b_f0-150",
    "video_0037_0_37_168b_f0-150", "video_0044_0_44_202b_f60-210",
]

# split targets for the 156 clean source groups (approx fractions)
FRAC_DEV, FRAC_VAL, FRAC_TEST = 0.15, 0.35, 0.50


def source_group(sample_id: str) -> str:
    m = re.match(r"(video_\d{4})_", sample_id)
    return m.group(1) if m else sample_id


def bucket(group: str) -> str:
    h = int(hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest()[:8], 16)
    r = h / 0xFFFFFFFF
    if r < FRAC_DEV:
        return "dev"
    if r < FRAC_DEV + FRAC_VAL:
        return "val"
    return "test"


def main() -> None:
    manifest = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    protocol = {r["sample_id"]: r for r in csv.DictReader(
        open(PROTOCOL, newline="", encoding="utf-8"))}

    groups: dict[str, list[str]] = defaultdict(list)
    for r in manifest:
        groups[source_group(r["sample_id"])].append(r["sample_id"])

    e1_sources = {source_group(s) for s in E1_DEV_SAMPLES}

    rows = []
    for g in sorted(groups):
        for sid in sorted(groups[g]):
            if sid in E1_DEV_SAMPLES:
                split, reason = "dev", "E1 rule development (v1-v3.1 tuning)"
            elif g in e1_sources:
                split, reason = "dev", "shares E1-dev source video (contamination guard)"
            else:
                split = bucket(g)
                reason = {
                    "dev": "held-out source group, deterministic hash",
                    "val": "held-out source group, deterministic hash",
                    "test": "held-out source group, deterministic hash",
                }[split]
            pr = protocol.get(sid, {})
            clarity = ("crossing-phrase" if pr.get("auto_prompt_crossing_phrase")
                       else "person-only" if pr.get("auto_prompt_mentions_person") == "yes"
                       else "no-person-prompt")
            rows.append({
                "sample_id": sid,
                "source_group": g,
                "split": split,
                "stratum_condition_clarity": clarity,
                "reason": reason,
                "review_status": "pending",
            })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["split"] for r in rows)
    g_by_split = defaultdict(set)
    for r in rows:
        g_by_split[r["split"]].add(r["source_group"])
    print(f"[split] wrote {len(rows)} rows -> {OUT}")
    for s in ("dev", "val", "test"):
        cl = Counter(r["stratum_condition_clarity"] for r in rows
                     if r["split"] == s)
        print(f"[split] {s}: {c[s]} samples / {len(g_by_split[s])} source groups "
              f"| clarity strata {dict(cl)}")
    # leakage invariant: no source group in two splits
    seen: dict[str, str] = {}
    leaks = []
    for r in rows:
        prev = seen.setdefault(r["source_group"], r["split"])
        if prev != r["split"]:
            leaks.append(r["source_group"])
    print(f"[split] source groups spanning multiple splits: {len(leaks)}"
          f"{' LEAK!' if leaks else ' (invariant OK)'}")


if __name__ == "__main__":
    main()

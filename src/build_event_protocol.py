"""E2 Phase 3: build the sample-level event protocol CSV.

Design rules (task brief, 2026-11-05):
  * `event_required` and all protocol fields stay `unknown`/empty until a
    human reviews the prompt + generation task definition — nothing is
    auto-filled as `true`.
  * The original video containing a pedestrian NEVER implies the generated
    video must keep it: the condition source is the PROMPT + video2world
    task definition, to be confirmed per sample.
  * Auto-extracted columns are prefixed `auto_` and are machine FACTS
    (regex matches, metadata values), clearly not protocol truth.

Output: datasets/annotations/event_protocol.csv (211 rows).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

MANIFEST = "datasets/annotations/manifest.csv"
OUT = "datasets/annotations/event_protocol.csv"

PERSON_KW = re.compile(
    r"pedestrian|walker|person|people|individuals?|figures?|foot traffic|"
    r"passerby|commuters?|children|child|shopper", re.I)
CROSS_KW = re.compile(
    r"cross(es|ing)? the (street|road|intersection|access road|lane)|"
    r"step(s|ping)? (across|into|off|onto)|"
    r"walk(s|ing)? (across|into|through) the (street|road|intersection)|"
    r"crosswalk|zebra|making (his|her|their|its) way (across|through|into)|"
    r"travers(e|es|ing) the (street|road|crosswalk)", re.I)


def sentence_with(pattern: re.Pattern, text: str) -> str:
    """First sentence containing the pattern, truncated."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if pattern.search(sent):
            s = " ".join(sent.split())
            return s[:220]
    return ""


def main() -> None:
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["sample_id"]
            prompt_path = r.get("prompt_path", "")
            meta_path = r.get("metadata_path", "")
            prompt_text = ""
            if prompt_path and Path(prompt_path).exists():
                prompt_text = Path(prompt_path).read_text(
                    encoding="utf-8", errors="replace")
            meta = {}
            if meta_path and Path(meta_path).exists():
                try:
                    meta = json.loads(Path(meta_path).read_text(
                        encoding="utf-8", errors="replace"))
                except json.JSONDecodeError:
                    meta = {}

            m_person = PERSON_KW.search(prompt_text)
            m_cross = CROSS_KW.search(prompt_text)

            rows.append({
                # ---- human-confirmed protocol fields (NEVER auto-filled) ---
                "sample_id": sid,
                "prompt": prompt_path,
                "generation_type": meta.get("inference_type", "unknown"),
                "event_required": "unknown",
                "event_type": "",
                "actor_category": "",
                "actor_identity": "",
                "required_behavior": "",
                "temporal_constraint": "",
                "spatial_constraint": "",
                "condition_source": (
                    f"prompt:{prompt_path}; metadata:{meta_path}; "
                    f"task={meta.get('inference_type', 'unknown')}; "
                    f"seed={meta.get('seed', 'unknown')}; "
                    f"guidance={meta.get('guidance', 'unknown')}"),
                "constraint_confidence": "unreviewed",
                "protocol_review_status": "pending",
                # ---- machine-extracted evidence (facts, NOT truth) ---------
                "auto_prompt_mentions_person": "yes" if m_person else "no",
                "auto_prompt_person_excerpt":
                    sentence_with(PERSON_KW, prompt_text) if m_person else "",
                "auto_prompt_crossing_phrase":
                    m_cross.group(0) if m_cross else "",
                "auto_prompt_crossing_excerpt":
                    sentence_with(CROSS_KW, prompt_text) if m_cross else "",
                "auto_negative_prompt_present":
                    "yes" if meta.get("negative_prompt") else "no",
                "auto_declared_frames":
                    str(meta.get("num_output_frames", "")),
            })

    fieldnames = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_p = sum(1 for r in rows if r["auto_prompt_mentions_person"] == "yes")
    n_c = sum(1 for r in rows if r["auto_prompt_crossing_phrase"])
    n_both = sum(1 for r in rows if r["auto_prompt_mentions_person"] == "yes"
                 and r["auto_prompt_crossing_phrase"])
    print(f"[protocol] wrote {len(rows)} rows -> {OUT}")
    print(f"[protocol] auto evidence: person-mention {n_p}/{len(rows)}, "
          f"crossing-phrase {n_c}/{len(rows)}, both {n_both}, "
          f"no-person {len(rows) - n_p}")
    print("[protocol] ALL event_required=unknown — pending human review")


if __name__ == "__main__":
    main()

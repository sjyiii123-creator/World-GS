"""Map SynPeDAS sample IDs to JAAD 2.0 ground-truth annotations.

KEY INSIGHT (verified 2026-11-05): the SynPeDAS sample_id middle segment IS
a JAAD pedestrian id::

    video_0016_0_16_72b_f0-150
            ^^^^^^^^^^    ^^^^^^
            JAAD ped id   frame window in the JAAD clip
            0_<video>_<ped#>  ('b' suffix = behavior-annotated pedestrian)

So JAAD_2.0's per-pedestrian attributes (crossing info) and per-frame
annotations (bbox + occlusion 0/1/2) can be attached to every SynPeDAS
sample directly by id + frame window.

IMPORTANT SEMANTICS (E2 discipline): this produces **real-side weak GT**
(crossing behaviour of the ORIGINAL clip).  It is NOT a label for the
generated video and MUST NOT be copied into human-truth columns; it
supports cross-checking manual annotation and occlusion-attribution audit.

Run after `jaad_annotations` clone:
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_jaad_gt_evidence.py

Output: datasets/annotations/jaad_gt_evidence.csv
"""
from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

JAAD = Path("repos/JAAD")
PROTOCOL = Path("datasets/annotations/event_protocol.csv")
OUT = Path("datasets/annotations/jaad_gt_evidence.csv")

SID_RE = re.compile(r"^video_(\d{4})_(0_\d+_\d+[bp])_f(\d+)-(\d+)$")


def parse_sid(sid: str):
    m = SID_RE.match(sid)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def find_track(ann_root, ped_id):
    """Locate the element for a pedestrian id.

    Two layouts (verified 2026-11-05 against actual files):
    * annotations/<clip>.xml — CVAT: <track><box frame occluded outside>
      with <attribute name="id">PED_ID</attribute> inside each box.
    * annotations_attributes/<clip>_attributes.xml — flat:
      <ped_attributes><pedestrian id="PED_ID" crossing=".." .../>.
    """
    for el in ann_root.iter():
        if el.tag.split("}")[-1] == "pedestrian" and el.get("id") == ped_id:
            return el
    for box in ann_root.iter("box"):
        for a in box.iter("attribute"):
            if a.get("name") == "id" and (a.text or "").strip() == ped_id:
                return box
    return None


def extract_attrs(xml_path: Path, ped_id: str) -> dict:
    """Per-pedestrian attributes (crossing, demographics, direction...)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {"_parse_error": str(xml_path)}
    for ped in root.iter("pedestrian"):
        if ped.get("id") == ped_id:
            return dict(ped.attrib)
    return {}


def frame_stats(xml_path: Path, ped_id: str, f0: int, f1: int) -> dict:
    """Per-frame bbox/occlusion/outside stats for a ped inside the window."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {"ann_parse_error": 1}
    boxes = []
    for box in root.iter("box"):
        pid = next((a.text for a in box.iter("attribute")
                    if a.get("name") == "id"), None)
        if pid != ped_id:
            continue
        try:
            fi = int(box.get("frame"))
        except (TypeError, ValueError):
            continue
        if f0 <= fi <= f1:
            boxes.append((fi, box.get("occluded", "0"), box.get("outside", "0")))
    if not boxes:
        return {"ann_track_found": 0}
    occ = [int(o) for _, o, _ in boxes]
    out = [int(o) for _, _, o in boxes]
    return {
        "ann_track_found": 1,
        "gt_frames_in_window": len(boxes),
        "gt_window_f0": min(f for f, _, _ in boxes),
        "gt_window_f1": max(f for f, _, _ in boxes),
        "gt_occ_partial_frames": sum(1 for o in occ if o == 1),
        "gt_occ_full_frames": sum(1 for o in occ if o == 2),
        "gt_outside_frames": sum(1 for o in out if o == 1),
    }


def main() -> None:
    if not JAAD.exists():
        raise SystemExit(f"JAAD repo missing: {JAAD} — run jaad_annotations first")
    protocol = list(csv.DictReader(open(PROTOCOL, newline="", encoding="utf-8")))
    ann_dir = JAAD / "annotations"
    attr_dir = JAAD / "annotations_attributes"
    files = sorted(ann_dir.glob("*.xml")) + sorted(attr_dir.glob("*.xml"))
    print(f"[jaad-gt] annotation files found: {len(files)} "
          f"(annotations={len(list(ann_dir.glob('*.xml')))}, "
          f"attributes={len(list(attr_dir.glob('*.xml')))})")

    rows, misses = [], []
    for r in protocol:
        sid = r["sample_id"]
        parsed = parse_sid(sid)
        if not parsed:
            misses.append((sid, "unparsed sample id"))
            continue
        vid, ped_id, f0, f1 = parsed
        stem = f"video_{vid}"
        row = {"sample_id": sid, "jaad_video": stem, "jaad_ped_id": ped_id,
               "jaad_frame_window": f"{f0}-{f1}"}
        attr_file = next((p for p in attr_dir.glob(f"{stem}*")), None)
        ann_file = next((p for p in ann_dir.glob(f"{stem}*")), None)
        if attr_file is not None:
            row.update({f"attr_{k}": v for k, v in
                        extract_attrs(attr_file, ped_id).items()})
        else:
            row["attr_file_found"] = "no"
        if ann_file is not None:
            row.update(frame_stats(ann_file, ped_id, f0, f1))
        else:
            row["ann_file_found"] = "no"
        row["semantics"] = "REAL-side weak GT (original clip behaviour); " \
                           "NOT generated-video truth; do not copy into " \
                           "human-truth columns"
        rows.append(row)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    matched = sum(1 for r in rows if r.get("ann_track_found") == "1"
                  or any(k.startswith("attr_") for k in r))
    print(f"[jaad-gt] wrote {len(rows)} rows -> {OUT}")
    print(f"[jaad-gt] peds matched in JAAD annotations: {matched}/{len(rows)}")
    if misses:
        print(f"[jaad-gt] unparsed ids: {misses[:5]}")


if __name__ == "__main__":
    main()

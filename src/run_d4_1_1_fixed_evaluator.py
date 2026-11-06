#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D4.1.1 fixed evaluator — per-anchor Top-1, seven unified methods, hard invariants.

What was broken (previous inline re-evaluation)
-----------------------------------------------
The inline scripts aggregated ALL candidate rows of a source video and ranked
them at video level, so one anchor did not produce one decision: the same
video-level top candidate was reused/recounted.  That is how a 23-anchor video
(video_0310) could report 325 "correct" associations — 325 is the number of
candidate ROWS of that video, not decisions.  Coverage was ranked over
candidate rows and per-source stats had no valid unit.

This module (reads the frozen audit CSV only — no models, no training)
----------------------------------------------------------------------
* Groups candidates by (split, source, anchor_id).  Every anchor yields
  exactly ONE Top-1 decision per method; candidates are never mixed across
  anchors.
* Seven methods, one scoring direction (higher = more trustworthy):
      kin        = -kin
      static     = static_cos
      kin_static = static_cos - kin
      b4         = b4_cos
      kin_b4     = b4_cos - kin
      d4         = d4_cos
      kin_d4     = d4_cos - kin
* Deterministic, content-based tie-breaking (numeric-aware ped order), so
  results never depend on CSV row order (checked as invariant I4).
* Non-finite method scores are disqualified from winning (effective -inf)
  without removing the candidate from any method's set (no silent skipping).
* Conflicting duplicate candidate keys are a hard error; fully identical
  duplicates are collapsed and reported.
* Statistical invariants (I1..I8) are enforced; on violation the evaluator
  stops and writes failure details instead of impossible numbers.
* Fixed-coverage (risk) evaluation operates on ANCHOR decisions; the
  denominator is the anchor count, never the candidate count.
* LOSO excludes ALL anchors of the left-out source and re-aggregates the
  remaining anchors (per method, plus deltas vs kin / kin_static / kin_b4).
* An EXTERNAL independent recomputation (REFERENCE_CHECKS below) is used
  strictly as a regression assertion; those numbers are never emitted as
  evaluation output.
* A run manifest keyed on run_id = sha256(code, frozen input, config) refuses
  to re-evaluate the same experiment twice (per output dir), and runs on the
  frozen input are additionally registered in a cross-output-dir registry
  (.formal_run_registry.json); formal results are published atomically only
  after every check passes.

CLI (from the repo root):
    python3 -I src/run_d4_1_1_fixed_evaluator.py \
        [--csv runs/d4_1_reevaluation/audit_candidates.csv] \
        [--out-dir runs/d4_1_1_integrity] \
        [--reference-check on|off] [--force] [--force-run-id RUN_ID]

Release gate: formal result files are staged in a temp dir and published
atomically ONLY after all invariants AND the reference regression pass; on
failure only diagnostics (failure reason + per-anchor details) are saved,
and the manifest is never marked completed.  --force may only retry a
NON-completed run; overwriting a completed formal result requires the
explicit confirmation --force-run-id <recorded run_id>.

Exit codes: 0 ok; 2 invariant violation; 3 reference mismatch; 4 run guard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------------- constants ---------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "runs/d4_1_reevaluation/audit_candidates.csv"
DEFAULT_OUT = REPO_ROOT / "runs/d4_1_1_integrity"
# Cross-output-dir registry for runs on the FROZEN input: the formal
# experiment may be evaluated at most once per run_id, regardless of
# --out-dir.  Test runs on synthetic CSVs never touch this file.
FORMAL_REGISTRY_PATH = DEFAULT_OUT / ".formal_run_registry.json"

GROUP_KEY_FIELDS = ("split", "source", "anchor_id")
SCORE_FIELDS = ("kin", "static_cos", "b4_cos", "d4_cos")
REQUIRED_CSV_FIELDS = ("anchor_id", "split", "source", "ped", "is_true") + SCORE_FIELDS

METHODS = ("kin", "static", "kin_static", "b4", "kin_b4", "d4", "kin_d4")
BASELINE_METHODS = ("kin", "kin_static", "kin_b4")

METHOD_FORMULAS = {
    "kin": "-kin",
    "static": "static_cos",
    "kin_static": "static_cos - kin",
    "b4": "b4_cos",
    "kin_b4": "b4_cos - kin",
    "d4": "d4_cos",
    "kin_d4": "d4_cos - kin",
}

FIXED_COVERAGE = 0.5
CURVE_COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

EXIT_OK = 0
EXIT_INVARIANT = 2
EXIT_REFERENCE = 3
EXIT_RUN_GUARD = 4

# Formal result files — staged and atomically published ONLY on a fully
# passing run; never written on a failed one.
FORMAL_FILES = (
    "seven_method_results.json",
    "per_anchor_predictions.csv",
    "per_source_results.csv",
    "loso_relative_gains.json",
    "risk_coverage.json",
    "invariant_checks.json",
    "reference_check.json",
)

# External independent recomputation of the val split — used ONLY as
# regression assertions (run_reference_check); never written as results.
REFERENCE_CHECKS = {
    "split": "val",
    "candidate_rows": 588,
    "anchors": 60,
    "sources": 15,
    "video_0310": {
        "n_anchors": 23,
        "correct": {"kin": 12, "kin_static": 15, "kin_b4": 15, "kin_d4": 14},
    },
    "top1_correct": {"kin": 39, "kin_static": 39, "kin_b4": 38, "kin_d4": 42},
    "coverage_50": {
        "accepted": 30,
        "wrong_among_accepted": {"kin": 8, "kin_static": 7, "kin_d4": 9},
    },
}


# ------------------------------- exceptions --------------------------------

class EvaluatorError(Exception):
    """Base class for evaluator failures."""


class DuplicateCandidateError(EvaluatorError):
    """Conflicting duplicate candidate keys in the input."""


class InvariantViolation(EvaluatorError):
    """A statistical invariant failed; no results may be published."""


class ReferenceMismatch(EvaluatorError):
    """Computed results disagree with the external reference recomputation."""


class RunGuardError(EvaluatorError):
    """The same code version was already evaluated (run-discipline guard)."""


# -------------------------------- helpers ----------------------------------

def numeric_key(text: str) -> Tuple[int, float, str]:
    """Deterministic, content-based sort key for id-ish strings.

    Numeric ids order numerically; non-numeric ids order lexicographically
    after all numeric ones.  Never depends on CSV row order.
    """
    t = str(text).strip()
    try:
        v = float(t)
        if math.isfinite(v):
            return (0, v, "")
    except ValueError:
        pass
    return (1, 0.0, t)


def parse_bool(value: str, *, field: str = "is_true") -> bool:
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise EvaluatorError(f"unparseable boolean for {field}: {value!r}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def build_config(args) -> dict:
    """Experiment configuration identity (feeds run_id together with the
    code hash and the input hash)."""
    return {
        "methods": list(METHODS),
        "method_formulas": dict(METHOD_FORMULAS),
        "score_direction": "higher_is_better",
        "tie_break": "numeric_aware_ped_ascending",
        "nonfinite_policy": "effective_minus_inf",
        "duplicate_policy": "conflicting_error_identical_collapse",
        "fixed_coverage": FIXED_COVERAGE,
        "curve_coverages": list(CURVE_COVERAGES),
        "loso_baselines": list(BASELINE_METHODS),
        "reference_check_mode": args.reference_check,
    }


def guard_override(args, prev: dict, run_id: str) -> bool:
    """Whether the single-execution guard may be overridden.

    --force may only retry a NON-completed run of the SAME experiment.
    Overwriting a completed formal result — or replacing a different
    experiment recorded in the same output dir — requires the explicit
    confirmation --force-run-id <recorded run_id>.
    """
    prev_id = str(prev.get("run_id", ""))
    if prev_id == run_id:
        if prev.get("status") == "completed":
            return args.force_run_id == run_id
        return bool(args.force) or args.force_run_id == run_id
    return args.force_run_id == prev_id


def registry_lookup(path: Path, run_id: str) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for entry in data.get("runs", []):
        if entry.get("run_id") == run_id:
            return entry
    return None


def registry_record(path: Path, run_id: str, out_dir: Path, status: str) -> None:
    """Append/update one experiment entry in a run registry (one entry per
    run_id; history of other run_ids is preserved)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    entries = [e for e in data.get("runs", []) if e.get("run_id") != run_id]
    entries.append({"run_id": run_id, "out_dir": str(out_dir), "status": status,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"runs": entries})


def group_sort_key(group_key: Tuple[str, str, str]):
    split, source, anchor_id = group_key
    return (split, source, numeric_key(anchor_id))


# -------------------------------- data model -------------------------------

@dataclass(frozen=True)
class Candidate:
    split: str
    source: str
    anchor_id: str
    ped: str
    is_true: bool
    kin: float
    static_cos: float
    b4_cos: float
    d4_cos: float
    row_index: int  # original CSV line; diagnostics only, never affects results

    @property
    def group_key(self) -> Tuple[str, str, str]:
        return (self.split, self.source, self.anchor_id)

    @property
    def candidate_key(self) -> Tuple[str, str, str, str]:
        return (self.split, self.source, self.anchor_id, self.ped)

    @property
    def tie_break_key(self) -> Tuple[int, float, str]:
        return numeric_key(self.ped)


@dataclass
class Decision:
    """One anchor's Top-1 decision under one method."""
    split: str
    source: str
    anchor_id: str
    method: str
    chosen_ped: str
    is_true: bool
    raw_score: float
    effective_score: float
    score_finite: bool
    n_candidates: int
    n_tied_at_top: int
    margin: Optional[float]  # top1 - top2 effective score, when both finite

    @property
    def group_key(self) -> Tuple[str, str, str]:
        return (self.split, self.source, self.anchor_id)

    def order_key(self):
        return (self.split, self.source, numeric_key(self.anchor_id))

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "source": self.source,
            "anchor_id": self.anchor_id,
            "method": self.method,
            "chosen_ped": self.chosen_ped,
            "is_true": int(self.is_true),
            "top1_score": round(self.raw_score, 6) if self.score_finite else None,
            "top1_score_finite": int(self.score_finite),
            "margin": round(self.margin, 6) if self.margin is not None else None,
            "n_candidates": self.n_candidates,
            "n_tied_at_top": self.n_tied_at_top,
        }


# ------------------------------ loading / pre-checks ------------------------

def load_candidates(csv_path: Path) -> List[Candidate]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_CSV_FIELDS if c not in fieldnames]
        if missing:
            raise EvaluatorError(f"{csv_path}: missing columns {missing}")
        out: List[Candidate] = []
        for i, raw in enumerate(reader):
            line = i + 2

            def _num(name: str) -> float:
                try:
                    return float(raw[name])
                except (TypeError, ValueError):
                    raise EvaluatorError(
                        f"{csv_path}:{line}: bad numeric for {name}: {raw.get(name)!r}"
                    ) from None

            ped = (raw.get("ped") or "").strip()
            if not ped:
                raise EvaluatorError(f"{csv_path}:{line}: empty ped")
            for name in GROUP_KEY_FIELDS:
                if not (raw.get(name) or "").strip():
                    raise EvaluatorError(f"{csv_path}:{line}: empty {name}")
            out.append(Candidate(
                split=raw["split"].strip(),
                source=raw["source"].strip(),
                anchor_id=raw["anchor_id"].strip(),
                ped=ped,
                is_true=parse_bool(raw["is_true"]),
                kin=_num("kin"),
                static_cos=_num("static_cos"),
                b4_cos=_num("b4_cos"),
                d4_cos=_num("d4_cos"),
                row_index=line,
            ))
    if not out:
        raise EvaluatorError(f"{csv_path}: no candidate rows")
    return out


def check_duplicates(cands: Sequence[Candidate]):
    """Return (unique_candidates, report).

    A duplicate candidate key is (split, source, anchor_id, ped).  Duplicates
    with CONFLICTING values are a hard error (refuse to evaluate).  Fully
    identical duplicates are collapsed to one row — content-identical, so
    which copy is kept cannot change any result.
    """
    by_key: Dict[tuple, Candidate] = {}
    identical_dups: List[Tuple[Tuple[str, str, str, str], int, int]] = []
    conflicts: List[Tuple[Tuple[str, str, str, str], int, int]] = []
    for c in cands:
        k = c.candidate_key
        prev = by_key.get(k)
        if prev is None:
            by_key[k] = c
            continue
        same = (prev.is_true == c.is_true
                and prev.kin == c.kin
                and prev.static_cos == c.static_cos
                and prev.b4_cos == c.b4_cos
                and prev.d4_cos == c.d4_cos)
        if same:
            identical_dups.append((k, prev.row_index, c.row_index))
        else:
            conflicts.append((k, prev.row_index, c.row_index))
    if conflicts:
        detail = "; ".join(f"{k}: lines {a} vs {b}" for k, a, b in conflicts[:10])
        raise DuplicateCandidateError(
            "conflicting duplicate candidate keys (split,source,anchor_id,ped) — "
            f"refusing to evaluate: {detail}" + (" ..." if len(conflicts) > 10 else ""))
    unique = sorted(by_key.values(),
                    key=lambda c: (group_sort_key(c.group_key), c.tie_break_key))
    report = {
        "conflicting_duplicate_keys": len(conflicts),
        "identical_duplicate_rows_collapsed": len(identical_dups),
        "identical_examples": [
            {"key": list(map(str, k)), "kept_line": a, "dropped_line": b}
            for k, a, b in identical_dups[:20]
        ],
    }
    return unique, report


def nonfinite_inventory(cands: Sequence[Candidate]) -> dict:
    inv = {f: 0 for f in SCORE_FIELDS}
    rows_with_nonfinite = 0
    for c in cands:
        hit = False
        for f in SCORE_FIELDS:
            if not math.isfinite(getattr(c, f)):
                inv[f] += 1
                hit = True
        if hit:
            rows_with_nonfinite += 1
    return {"cells_per_field": inv, "rows_with_nonfinite": rows_with_nonfinite}


# --------------------------------- scoring ----------------------------------

def method_score(method: str, c: Candidate) -> float:
    """Unified scoring: higher = more trustworthy, for ALL seven methods."""
    kin, s, b, d = c.kin, c.static_cos, c.b4_cos, c.d4_cos
    if method == "kin":
        return -kin
    if method == "static":
        return s
    if method == "kin_static":
        return s - kin
    if method == "b4":
        return b
    if method == "kin_b4":
        return b - kin
    if method == "d4":
        return d
    if method == "kin_d4":
        return d - kin
    raise EvaluatorError(f"unknown method {method!r}")


def effective_score(raw: float) -> float:
    """Non-finite raw scores never win a ranking slot over a finite score."""
    return raw if math.isfinite(raw) else float("-inf")


# ------------------------------ grouping / Top-1 ----------------------------

def group_candidates(cands: Sequence[Candidate]) -> Dict[Tuple[str, str, str], List[Candidate]]:
    """Group by (split, source, anchor_id) — the unit of evaluation."""
    groups: Dict[Tuple[str, str, str], List[Candidate]] = defaultdict(list)
    for c in cands:
        groups[c.group_key].append(c)
    return dict(groups)


def decide_anchor(group_cands: Sequence[Candidate], method: str) -> Decision:
    """One anchor -> ONE Top-1 decision under `method`.

    Ranking: max effective score; ties broken by ascending numeric-aware ped
    key (content-based, order-independent).  Non-finite scores rank last.
    """
    if not group_cands:
        raise InvariantViolation("decide_anchor called with an empty group")
    scored = [(effective_score(method_score(method, c)), c.tie_break_key, c) for c in group_cands]
    ranked = sorted(scored, key=lambda t: (-t[0], t[1]))
    top_eff, _, best = ranked[0]
    raw = method_score(method, best)
    n_tied = sum(1 for eff, _, _ in ranked if eff == top_eff)
    margin = None
    if len(ranked) > 1 and math.isfinite(top_eff) and math.isfinite(ranked[1][0]):
        margin = top_eff - ranked[1][0]
    return Decision(
        split=best.split, source=best.source, anchor_id=best.anchor_id,
        method=method, chosen_ped=best.ped, is_true=best.is_true,
        raw_score=raw, effective_score=top_eff,
        score_finite=math.isfinite(raw),
        n_candidates=len(group_cands), n_tied_at_top=n_tied, margin=margin,
    )


def evaluate_method(groups: Dict[Tuple[str, str, str], List[Candidate]], method: str) -> List[Decision]:
    """All decisions for one method, in deterministic anchor order."""
    return [decide_anchor(g, method)
            for _, g in sorted(groups.items(), key=lambda kv: group_sort_key(kv[0]))]


def results_projection(decisions_by_method: Dict[str, List[Decision]]) -> dict:
    """JSON-able projection of all decisions (used for order-invariance checks)."""
    return {m: [d.to_dict() for d in decisions_by_method[m]] for m in sorted(decisions_by_method)}


# ------------------------------- aggregations -------------------------------

def summarize_decisions(decisions: Sequence[Decision]) -> dict:
    by_split: Dict[str, dict] = defaultdict(
        lambda: {"n_anchors": 0, "n_correct": 0, "n_top1_nonfinite": 0})
    for d in decisions:
        s = by_split[d.split]
        s["n_anchors"] += 1
        s["n_correct"] += int(d.is_true)
        s["n_top1_nonfinite"] += int(not d.score_finite)
    out = {}
    for split, s in sorted(by_split.items()):
        n = s["n_anchors"]
        out[split] = {
            "n_anchors": n,
            "n_correct": s["n_correct"],
            "top1_accuracy": round(s["n_correct"] / n, 6) if n else None,
            "n_top1_nonfinite": s["n_top1_nonfinite"],
        }
    return out


def per_source_stats(groups: Dict[Tuple[str, str, str], List[Candidate]],
                     decisions_by_method: Dict[str, List[Decision]]) -> List[dict]:
    """Per (split, source): n_anchors, n_candidates and correct counts."""
    sources: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {"n_anchors": 0, "n_candidates": 0,
                 "correct": {m: 0 for m in METHODS}})
    for gk, g in groups.items():
        rec = sources[(gk[0], gk[1])]
        rec["n_anchors"] += 1
        rec["n_candidates"] += len(g)
    for m, decisions in decisions_by_method.items():
        for d in decisions:
            sources[(d.split, d.source)]["correct"][m] += int(d.is_true)
    rows = []
    for (split, source), rec in sorted(sources.items()):
        row = {"split": split, "source": source,
               "n_anchors": rec["n_anchors"], "n_candidates": rec["n_candidates"]}
        for m in METHODS:
            row[f"{m}_correct"] = rec["correct"][m]
        rows.append(row)
    return rows


def assert_source_bound(per_source_rows: Sequence[dict]) -> None:
    """Invariant: correct_count <= n_anchors for every source and method."""
    bad = []
    for r in per_source_rows:
        for m in METHODS:
            if r.get(f"{m}_correct", 0) > r["n_anchors"]:
                bad.append(f"{r.get('split')}/{r['source']}: {m} "
                           f"{r.get(f'{m}_correct')} > {r['n_anchors']} anchors")
    if bad:
        raise InvariantViolation(
            "per-source correct count exceeds anchor count: " + "; ".join(bad))


# ----------------------------- coverage / risk ------------------------------

def rank_by_confidence(decisions: Sequence[Decision]) -> List[Decision]:
    """Deterministic cross-anchor ranking: Top-1 score (confidence)
    descending, ties broken by anchor key ascending."""
    return sorted(decisions, key=lambda d: (-d.effective_score, d.order_key()))


def coverage_block(decisions: Sequence[Decision], coverage: float) -> Optional[dict]:
    """Fixed-coverage risk over ANCHOR decisions (never candidate rows).

    Confidence = the method's own Top-1 score (higher = more confident).
    accepted = round-half-up(coverage * n_anchors).  Denominator is the
    anchor count.  Ties at the accept boundary are resolved deterministically
    (anchor key ascending) and flagged.
    """
    n = len(decisions)
    if n == 0:
        return None
    accepted_n = int(math.floor(coverage * n + 0.5))
    ranked = rank_by_confidence(decisions)
    accepted = ranked[:accepted_n]
    accepted_correct = sum(1 for d in accepted if d.is_true)
    total_correct = sum(1 for d in decisions if d.is_true)
    wrong = accepted_n - accepted_correct
    boundary_tie = False
    if 0 < accepted_n < n:
        boundary_tie = (ranked[accepted_n - 1].effective_score
                        == ranked[accepted_n].effective_score)
    return {
        "coverage": coverage,
        "n_anchors": n,
        "accepted": accepted_n,
        "accepted_correct": accepted_correct,
        "accepted_wrong": wrong,
        "wrong_among_accepted": wrong,
        "wrong_among_accepted_fraction": round(wrong / accepted_n, 6) if accepted_n else None,
        "top1_accuracy": round(total_correct / n, 6),
        "boundary_tie": boundary_tie,
    }


# ----------------------------------- LOSO -----------------------------------

def loso_analysis(groups: Dict[Tuple[str, str, str], List[Candidate]]) -> dict:
    """Leave-one-source-out over ANCHOR decisions.

    Each fold excludes ALL anchors of the left-out source and re-aggregates
    the remaining anchors per method.  Deltas are reported vs the three
    baselines.  NOTE: LOSO inside one split is a within-dataset stability
    check, NOT an independent cross-domain test.
    """
    out: Dict[str, dict] = {}
    for split in sorted({gk[0] for gk in groups}):
        split_groups = {k: v for k, v in groups.items() if k[0] == split}
        sources = sorted({k[1] for k in split_groups})
        folds = []
        for s in sources:
            keep = {k: v for k, v in split_groups.items() if k[1] != s}
            n_total, n_keep = len(split_groups), len(keep)
            methods = {}
            for m in METHODS:
                dec = evaluate_method(keep, m)
                correct = sum(1 for d in dec if d.is_true)
                methods[m] = {
                    "n_anchors": n_keep,
                    "n_correct": correct,
                    "accuracy": round(correct / n_keep, 6) if n_keep else None,
                }
            deltas = {}
            for m in METHODS:
                deltas[m] = {}
                for b in BASELINE_METHODS:
                    key = f"delta_vs_{b}"
                    deltas[m][key] = (round(methods[m]["accuracy"] - methods[b]["accuracy"], 6)
                                      if n_keep else None)
            folds.append({
                "left_out_source": s,
                "n_anchors_total": n_total,
                "n_anchors_remaining": n_keep,
                "methods": methods,
                "deltas": deltas,
            })
        summary: Dict[str, dict] = {}
        for m in METHODS:
            for b in BASELINE_METHODS:
                key = f"delta_vs_{b}"
                vals = [f["deltas"][m][key] for f in folds
                        if f["deltas"][m][key] is not None]
                summary.setdefault(m, {})[key] = {
                    "mean": round(sum(vals) / len(vals), 6) if vals else None,
                    "min": min(vals) if vals else None,
                    "max": max(vals) if vals else None,
                    "folds_negative": sum(1 for v in vals if v < 0),
                    "folds_zero": sum(1 for v in vals if v == 0),
                    "folds_positive": sum(1 for v in vals if v > 0),
                }
        out[split] = {"folds": folds, "summary": summary}
    return out


# -------------------------------- invariants --------------------------------

def check_invariants(groups, decisions_by_method, unique_cands,
                     duplicate_report, nonfinite) -> Tuple[List[dict], List[dict]]:
    """Enforce the statistical invariants; return (all_checks, failed_checks)."""
    checks: List[dict] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL",
                       "detail": detail})
        return ok

    n_groups = len(groups)

    # I1: exactly one Top-1 decision per anchor per method
    ok, det = True, []
    for m, decisions in decisions_by_method.items():
        if len(decisions) != n_groups or {d.group_key for d in decisions} != set(groups):
            ok = False
            det.append(f"{m}: {len(decisions)} decisions vs {n_groups} anchors")
    record("I1_one_top1_decision_per_anchor", ok,
           "; ".join(det) or f"7 methods x {n_groups} anchors, decision keys == group keys")

    # I2: per-source correct <= n_anchors
    ps = per_source_stats(groups, decisions_by_method)
    try:
        assert_source_bound(ps)
        record("I2_source_correct_le_anchors", True,
               f"{len(ps)} (split, source) rows, all 7 methods checked")
    except InvariantViolation as exc:
        record("I2_source_correct_le_anchors", False, str(exc))

    # I3: per-split total correct <= split anchor count
    bad3 = []
    for split in sorted({gk[0] for gk in groups}):
        n = sum(1 for gk in groups if gk[0] == split)
        for m in METHODS:
            c = sum(1 for d in decisions_by_method[m]
                    if d.split == split and d.is_true)
            if c > n:
                bad3.append(f"{split}/{m}: {c} > {n}")
    record("I3_split_correct_le_anchors", not bad3,
           "; ".join(bad3) or "all splits x 7 methods checked")

    # I4: row-order invariance — full re-run on the reversed candidate list
    rev_groups = group_candidates(list(reversed(unique_cands)))
    rev_decisions = {m: evaluate_method(rev_groups, m) for m in METHODS}
    same = all(
        [d.to_dict() for d in rev_decisions[m]] == [d.to_dict() for d in decisions_by_method[m]]
        for m in METHODS)
    record("I4_row_order_invariance", same,
           "pipeline re-run on reversed candidate list; all 7 methods' decisions identical"
           if same else "decisions differ under row reversal")

    # I5: no duplicate candidate keys
    record("I5_no_duplicate_candidate_keys",
           duplicate_report["conflicting_duplicate_keys"] == 0,
           f"conflicting=0; identical rows collapsed="
           f"{duplicate_report['identical_duplicate_rows_collapsed']}")

    # I6: a non-finite score never wins while a finite one exists
    ok6, det6 = True, []
    for m, decisions in decisions_by_method.items():
        for d in decisions:
            if not d.score_finite:
                g = groups[d.group_key]
                if any(math.isfinite(method_score(m, c)) for c in g):
                    ok6 = False
                    det6.append(f"{m}/{d.group_key}")
    record("I6_nonfinite_disqualified", ok6,
           "; ".join(det6) or (f"data rows with non-finite cells: {nonfinite['rows_with_nonfinite']}; "
                               "no non-finite decision beat a finite candidate"))

    # I7: per-method provenance — every decision is the argmax of THAT method's
    # own scores (independent max-based re-check, not the sort used to decide)
    ok7, det7 = True, []
    for m, decisions in decisions_by_method.items():
        for d in decisions:
            g = groups[d.group_key]
            chosen_tie = numeric_key(d.chosen_ped)
            for c in g:
                eff = effective_score(method_score(m, c))
                if eff > d.effective_score or (eff == d.effective_score
                                               and c.tie_break_key < chosen_tie):
                    ok7 = False
                    if len(det7) < 5:
                        det7.append(f"{m}/{d.group_key}: ped {c.ped} beats chosen {d.chosen_ped}")
    record("I7_method_provenance_argmax", ok7,
           "; ".join(det7) or "every decision re-verified as the argmax of its own method score")

    # I8: consistent candidate sets across methods (no silent skipping)
    counts: Dict[tuple, set] = defaultdict(set)
    for m, decisions in decisions_by_method.items():
        for d in decisions:
            counts[d.group_key].add(d.n_candidates)
    bad8 = [k for k, v in counts.items() if len(v) != 1]
    record("I8_consistent_candidate_sets", not bad8,
           f"n_candidates per anchor identical across all 7 methods ({n_groups} anchors)"
           if not bad8 else f"anchors with inconsistent sets: {bad8[:5]}")

    failed = [c for c in checks if c["status"] == "FAIL"]
    return checks, failed


# ----------------------------- reference check ------------------------------

def run_reference_check(groups, decisions_by_method, coverage_results) -> Tuple[dict, List[dict]]:
    """Regression-assert the val split against the EXTERNAL independent
    recomputation (REFERENCE_CHECKS).  Reference numbers are assertions only —
    they are never emitted as evaluation output."""
    ref = REFERENCE_CHECKS
    split = ref["split"]
    checks: List[dict] = []

    def cmp(name, kind, expected, actual, method=None, source=None):
        checks.append({"name": name, "kind": kind, "method": method,
                       "source": source, "expected": expected, "actual": actual,
                       "pass": expected == actual})

    sg = {k: v for k, v in groups.items() if k[0] == split}
    cmp("val_candidate_rows", "count", ref["candidate_rows"],
        sum(len(g) for g in sg.values()))
    cmp("val_anchors", "count", ref["anchors"], len(sg))
    cmp("val_sources", "count", ref["sources"], len({k[1] for k in sg}))

    v310 = {k: v for k, v in sg.items() if k[1] == "video_0310"}
    cmp("video_0310_anchors", "count", ref["video_0310"]["n_anchors"], len(v310))
    for m, expected in ref["video_0310"]["correct"].items():
        actual = sum(1 for d in decisions_by_method[m]
                     if d.split == split and d.source == "video_0310" and d.is_true)
        cmp(f"video_0310_{m}_correct", "source", expected, actual, method=m,
            source="video_0310")

    for m, expected in ref["top1_correct"].items():
        actual = sum(1 for d in decisions_by_method[m]
                     if d.split == split and d.is_true)
        cmp(f"val_top1_{m}", "top1", expected, actual, method=m)

    for m, expected in ref["coverage_50"]["wrong_among_accepted"].items():
        blk = coverage_results[split][m]["coverage_0.5"]
        cmp(f"val_coverage50_{m}_wrong", "coverage", expected,
            blk["wrong_among_accepted"], method=m)
    for m in ref["coverage_50"]["wrong_among_accepted"]:
        blk = coverage_results[split][m]["coverage_0.5"]
        cmp(f"val_coverage50_{m}_accepted", "coverage",
            ref["coverage_50"]["accepted"], blk["accepted"], method=m)

    failed = [c for c in checks if not c["pass"]]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checked_split": split,
        "note": ("Reference values are an external independent recomputation, "
                 "used ONLY as regression assertions; never emitted as "
                 "evaluation output."),
        "checks": checks,
    }
    return result, failed


def build_mismatch_details(failed_checks, groups, decisions_by_method,
                           coverage_results) -> dict:
    """On reference mismatch: report the specific anchors, candidates, scores."""
    out = {}
    for ch in failed_checks:
        m = ch.get("method")
        entry = {"name": ch["name"], "expected": ch["expected"],
                 "actual": ch["actual"]}
        if m:
            if ch["kind"] == "coverage":
                ranked = sorted([d for d in decisions_by_method[m]
                                 if d.split == REFERENCE_CHECKS["split"]],
                                key=lambda d: (-d.effective_score, d.order_key()))
                entry["ranked_anchor_confidence"] = [
                    {"anchor": list(d.group_key), "chosen_ped": d.chosen_ped,
                     "is_true": int(d.is_true),
                     "confidence": d.raw_score if d.score_finite else None}
                    for d in ranked]
            else:
                ds = [d for d in decisions_by_method[m]
                      if d.split == REFERENCE_CHECKS["split"]
                      and (not ch.get("source") or d.source == ch["source"])]
                entry["anchor_decisions_with_candidates"] = [
                    {"anchor": list(d.group_key), "chosen_ped": d.chosen_ped,
                     "is_true": int(d.is_true),
                     "top1_score": d.raw_score if d.score_finite else None,
                     "candidates": [
                         {"ped": c.ped, "is_true": int(c.is_true), "kin": c.kin,
                          "static_cos": c.static_cos, "b4_cos": c.b4_cos,
                          "d4_cos": c.d4_cos}
                         for c in groups[d.group_key]]}
                    for d in ds]
        out[ch["name"]] = entry
    return out


# ------------------------------ output writers ------------------------------

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_outputs(out_dir: Path, *, groups, decisions_by_method, summaries,
                  per_source, coverage_results, loso, checks, ref_result,
                  metadata) -> None:
    """Write the formal output files into `out_dir` (a staging directory in
    the release-gate flow; the caller moves them into place atomically after
    every check has passed).  The run log and manifest are owned by main()."""
    # seven_method_results.json
    write_json(out_dir / "seven_method_results.json", {
        "metadata": metadata,
        "dataset": metadata["dataset_stats"],
        "splits": {split: {m: summaries[m].get(split) for m in METHODS}
                   for split in sorted({gk[0] for gk in groups})},
    })

    # per_anchor_predictions.csv — one row per (anchor, method)
    with (out_dir / "per_anchor_predictions.csv").open("w", newline="",
                                                       encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "source", "anchor_id", "method", "n_candidates",
                    "chosen_ped", "is_true", "top1_score", "top1_score_finite",
                    "margin", "n_tied_at_top"])
        for m in METHODS:
            for d in decisions_by_method[m]:
                w.writerow([d.split, d.source, d.anchor_id, m, d.n_candidates,
                            d.chosen_ped, int(d.is_true),
                            f"{d.raw_score:.6f}" if d.score_finite else "",
                            int(d.score_finite),
                            f"{d.margin:.6f}" if d.margin is not None else "",
                            d.n_tied_at_top])

    # per_source_results.csv
    with (out_dir / "per_source_results.csv").open("w", newline="",
                                                   encoding="utf-8") as f:
        w = csv.writer(f)
        header = (["split", "source", "n_anchors", "n_candidates"]
                  + [f"{m}_correct" for m in METHODS])
        w.writerow(header)
        for r in per_source:
            w.writerow([r["split"], r["source"], r["n_anchors"], r["n_candidates"]]
                       + [r[f"{m}_correct"] for m in METHODS])

    # loso_relative_gains.json
    write_json(out_dir / "loso_relative_gains.json", {
        "definition": {
            "unit": "anchor decisions (one Top-1 per anchor per method)",
            "fold": "exclude ALL anchors of the left-out source, then "
                    "re-aggregate the remaining anchors",
            "deltas": "accuracy(method) - accuracy(baseline) on the remaining "
                      f"anchors, baselines = {list(BASELINE_METHODS)}",
            "note": "LOSO within one split is a within-dataset stability "
                    "check, NOT an independent cross-domain test.",
        },
        "splits": loso,
    })

    # risk_coverage.json
    write_json(out_dir / "risk_coverage.json", {
        "definition": {
            "unit": "anchor decisions (never candidate rows)",
            "confidence": "the method's own Top-1 score (higher = more "
                          "confident)",
            "tie_break": "anchor key (split, source, numeric anchor_id) "
                         "ascending",
            "accepted_rule": "round-half-up(coverage * n_anchors); the "
                             "denominator is the anchor count, never the "
                             "candidate count",
            "fixed_coverage": FIXED_COVERAGE,
        },
        "splits": coverage_results,
    })

    # invariant_checks.json + reference_check.json (formal outputs; published
    # only on a passing run; the run log and manifest are written by main)
    write_json(out_dir / "invariant_checks.json", checks)
    write_json(out_dir / "reference_check.json", ref_result)


def dataset_stats(groups, unique_cands, nonfinite, dup_report) -> dict:
    stats = {}
    for split in sorted({gk[0] for gk in groups}):
        sg = {k: v for k, v in groups.items() if k[0] == split}
        true_per_anchor = [sum(1 for c in g if c.is_true) for g in sg.values()]
        stats[split] = {
            "candidate_rows": sum(len(g) for g in sg.values()),
            "anchors": len(sg),
            "sources": len({k[1] for k in sg}),
            "anchors_without_true": sum(1 for t in true_per_anchor if t == 0),
            "anchors_with_multi_true": sum(1 for t in true_per_anchor if t > 1),
            "nonfinite_rows": sum(
                1 for c in unique_cands if c.split == split
                and any(not math.isfinite(getattr(c, f)) for f in SCORE_FIELDS)),
        }
    return stats


# ----------------------------------- main -----------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="D4.1.1 fixed evaluator (per-anchor Top-1, 7 methods)")
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="audit candidates CSV (default: frozen re-evaluation CSV)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help="output directory (default: runs/d4_1_1_integrity)")
    ap.add_argument("--reference-check", choices=("on", "off"), default="on",
                    help="on: assert val results against the external "
                         "independent recomputation (default)")
    ap.add_argument("--force", action="store_true",
                    help="retry a NON-completed run of the same experiment; "
                         "never overwrites a completed formal result")
    ap.add_argument("--force-run-id", default=None, metavar="RUN_ID",
                    help="explicit confirmation to deliberately replace the "
                         "recorded experiment RUN_ID (required to overwrite "
                         "a completed formal result)")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    code_sha = code_sha256()
    config = build_config(args)
    config_sha = sha256_text(json.dumps(config, sort_keys=True))
    log_lines: List[str] = []

    try:
        input_sha = sha256_file(csv_path)
    except OSError as exc:
        sys.stderr.write(f"[load-fail] cannot read csv {csv_path}: {exc}\n")
        return EXIT_INVARIANT
    run_id = sha256_text(f"{code_sha}\n{input_sha}\n{config_sha}")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest = {
        "run_id": run_id,
        "code_version": {"sha256": code_sha,
                         "file": str(Path(__file__).resolve())},
        "input": {"csv": str(csv_path), "sha256": input_sha},
        "config": config,
        "config_sha256": config_sha,
        "reference_check": {"mode": args.reference_check, "status": "pending"},
        "argv": list(sys.argv[1:]),
        "started_at": started,
        "status": "running",
    }
    log_lines.append(f"started {started} run_id={run_id[:12]} "
                     f"code={code_sha[:12]} input={input_sha[:12]} "
                     f"config={config_sha[:12]} csv={csv_path} "
                     f"reference_check={args.reference_check}")

    # ---- single-execution guard: run_id = hash(code, input, config) ------
    prev = read_manifest(manifest_path)
    override = prev is not None and guard_override(args, prev, run_id)
    if prev is not None and not override:
        prev_id = str(prev.get("run_id", "unknown"))
        if prev_id == run_id:
            msg = (f"[run-guard] experiment {run_id[:12]} (identical code "
                   f"sha256 + input sha256 + config) already ran, status="
                   f"{prev.get('status')}. Refusing to re-run the same "
                   f"experiment.")
            if prev.get("status") == "completed":
                msg += (f" To deliberately replace the completed formal "
                        f"result, confirm with --force-run-id {run_id}.")
            else:
                msg += (" Investigate the failure and change the code, or "
                        "pass --force to retry.")
        else:
            msg = (f"[run-guard] a DIFFERENT experiment (run_id "
                   f"{prev_id[:12]}, status={prev.get('status')}) is already "
                   f"recorded in this output dir. Refusing to mix experiments "
                   f"or silently reuse results. Use a fresh --out-dir, or "
                   f"deliberately replace it with --force-run-id {prev_id}.")
        sys.stderr.write(msg + "\n")
        return EXIT_RUN_GUARD
    if prev is not None and override:
        # Deliberate, confirmed replacement: archive the previous run's
        # formal files so nothing can be mistaken for the new run's output.
        prev_id = str(prev.get("run_id", "unknown"))
        archive = out_dir / f"superseded_{prev_id[:12]}"
        archive.mkdir(exist_ok=True)
        moved = [n for n in FORMAL_FILES + ("evaluation_run_log.txt",)
                 if (out_dir / n).exists()]
        for n in moved:
            os.replace(out_dir / n, archive / n)
        log_lines.append(
            f"deliberate replacement of recorded run {prev_id} "
            f"(status={prev.get('status')}): archived {len(moved)} previous "
            f"output files to {archive.name}/")

    # ---- frozen-input registry: formal experiment runs once, ever ---------
    # Guards the FROZEN input across output directories: the same
    # (code, input, config) formal experiment cannot be re-evaluated into a
    # different --out-dir.  Synthetic/test CSVs never reach this path.
    is_formal_input = csv_path.resolve() == DEFAULT_CSV.resolve()
    if is_formal_input:
        entry = registry_lookup(FORMAL_REGISTRY_PATH, run_id)
        if (entry is not None and entry.get("status") == "completed"
                and args.force_run_id != run_id):
            sys.stderr.write(
                f"[run-guard] formal experiment {run_id[:12]} on the frozen "
                f"input was already evaluated (status=completed, out_dir="
                f"{entry.get('out_dir')}). Refusing cross-directory "
                f"re-evaluation; confirm with --force-run-id {run_id}.\n")
            return EXIT_RUN_GUARD

    atomic_write_json(manifest_path, manifest)
    for leftover in out_dir.glob(".staging*"):
        shutil.rmtree(leftover, ignore_errors=True)
    if is_formal_input:
        registry_record(FORMAL_REGISTRY_PATH, run_id, out_dir, "running")

    def finish(status: str, exit_code: int) -> int:
        manifest.update({"status": status, "finished_at":
                         time.strftime("%Y-%m-%dT%H:%M:%S"), "exit_code": exit_code})
        atomic_write_json(manifest_path, manifest)
        if is_formal_input:
            registry_record(FORMAL_REGISTRY_PATH, run_id, out_dir, status)
        (out_dir / "evaluation_run_log.txt").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8")
        return exit_code

    # ------------------------------- load ----------------------------------
    try:
        cands = load_candidates(csv_path)
        unique, dup_report = check_duplicates(cands)
        nonfinite = nonfinite_inventory(unique)
        groups = group_candidates(unique)
    except (EvaluatorError, OSError) as exc:
        log_lines.append(f"LOAD FAIL: {exc}")
        sys.stderr.write(f"[load-fail] {exc}\n")
        return finish("load_failed", EXIT_INVARIANT)

    manifest["input"]["rows_loaded"] = len(cands)
    manifest["input"]["rows_after_dedup"] = len(unique)
    log_lines.append(
        f"loaded {len(cands)} rows -> {len(unique)} unique candidates, "
        f"{len(groups)} anchors; duplicates: conflicting=0 identical_collapsed="
        f"{dup_report['identical_duplicate_rows_collapsed']}; non-finite rows="
        f"{nonfinite['rows_with_nonfinite']}")

    # ----------------------------- evaluate --------------------------------
    decisions_by_method = {m: evaluate_method(groups, m) for m in METHODS}
    summaries = {m: summarize_decisions(decisions_by_method[m]) for m in METHODS}

    # ---------------------------- invariants -------------------------------
    checks, failed = check_invariants(groups, decisions_by_method, unique,
                                      dup_report, nonfinite)
    for c in checks:
        log_lines.append(f"invariant {c['name']}: {c['status']} — {c['detail']}")
    if failed:
        atomic_write_json(out_dir / "invariant_checks.json", checks)
        sys.stderr.write("[invariant-violation] "
                         + "; ".join(f"{c['name']}: {c['detail']}" for c in failed)
                         + "\nNo formal results are published.\n")
        log_lines.append("RESULT: invariant violation — diagnostics only, "
                         "no formal results published")
        return finish("invariant_violation", EXIT_INVARIANT)

    # ------------------------------ coverage -------------------------------
    coverage_results: Dict[str, dict] = {}
    for split in sorted({gk[0] for gk in groups}):
        coverage_results[split] = {}
        for m in METHODS:
            dec = [d for d in decisions_by_method[m] if d.split == split]
            coverage_results[split][m] = {
                "coverage_0.5": coverage_block(dec, FIXED_COVERAGE),
                "curve": [coverage_block(dec, c) for c in CURVE_COVERAGES],
            }

    # -------------------------------- LOSO ---------------------------------
    loso = loso_analysis(groups)
    per_source = per_source_stats(groups, decisions_by_method)

    # --------------------------- reference check ---------------------------
    ref_result = {"status": "skipped",
                  "reason": "--reference-check off (selftest mode)"}
    ref_failed: List[dict] = []
    if args.reference_check == "on":
        ref_result, ref_failed = run_reference_check(groups, decisions_by_method,
                                                     coverage_results)
        log_lines.append(f"reference regression: {ref_result['status']} "
                         f"({sum(1 for c in ref_result['checks'] if c['pass'])}"
                         f"/{len(ref_result['checks'])} checks pass)")
    manifest["reference_check"]["status"] = ref_result["status"]

    # ---- release gate: on failure save ONLY diagnostics, never results ---
    if ref_failed:
        atomic_write_json(out_dir / "invariant_checks.json", checks)
        atomic_write_json(out_dir / "reference_check.json", ref_result)
        details = build_mismatch_details(ref_failed, groups,
                                         decisions_by_method, coverage_results)
        atomic_write_json(out_dir / "reference_mismatch_details.json", details)
        sys.stderr.write(
            "[reference-mismatch] computed results disagree with the external "
            "independent recomputation:\n  "
            + "\n  ".join(f"{c['name']}: expected {c['expected']}, got {c['actual']}"
                          for c in ref_failed)
            + "\nDetails (specific anchors, candidates, scores) written to "
              "reference_mismatch_details.json. Stopping: no formal results "
              "are published.\n")
        log_lines.append("RESULT: reference mismatch — diagnostics only, "
                         "no formal results published")
        print("[D4.1.1] RESULT: FAIL (reference mismatch) — diagnostics "
              "only; no formal evaluation results published")
        return finish("reference_mismatch", EXIT_REFERENCE)

    # ---- publish: stage formal files, then move them in atomically -------
    # Formal files are written to a staging dir first; only after every
    # invariant AND the reference regression have passed are they moved into
    # place (os.replace is atomic per file).  The run log and manifest are
    # written last, so a failed or partially published run is never marked
    # completed.
    metadata = {
        "evaluator": "src/run_d4_1_1_fixed_evaluator.py",
        "code_sha256": code_sha,
        "run_id": run_id,
        "csv": {"path": str(csv_path), "sha256": input_sha,
                "rows_loaded": len(cands), "rows_after_dedup": len(unique)},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": dict(METHOD_FORMULAS),
        "score_direction": "higher = more trustworthy (all 7 methods)",
        "tie_break": "max effective score, then ascending numeric-aware ped "
                     "key (content-based; row-order independent)",
        "nonfinite_policy": "non-finite method scores are disqualified "
                            "(effective -inf) but stay in the candidate set",
        "duplicate_policy": "conflicting duplicate candidate keys are a hard "
                            "error; identical duplicates collapse to one row",
        "reference_check": "external independent recomputation used only as "
                           "regression assertions, never as output",
        "dataset_stats": dataset_stats(groups, unique, nonfinite, dup_report),
    }
    staging = out_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    write_outputs(staging, groups=groups, decisions_by_method=decisions_by_method,
                  summaries=summaries, per_source=per_source,
                  coverage_results=coverage_results, loso=loso, checks=checks,
                  ref_result=ref_result, metadata=metadata)
    published = [n for n in FORMAL_FILES if (staging / n).exists()]
    for name in published:
        os.replace(staging / name, out_dir / name)
    shutil.rmtree(staging, ignore_errors=True)
    log_lines.append("published formal outputs atomically: " + ", ".join(published))

    # ----------------------------- summary ---------------------------------
    val = REFERENCE_CHECKS["split"]
    val_sum = {m: summaries[m].get(val) for m in METHODS}
    cov50 = {m: coverage_results.get(val, {}).get(m, {}).get("coverage_0.5")
             for m in METHODS}
    print(f"[D4.1.1] run_id {run_id[:12]} | code {code_sha[:12]} | csv "
          f"{csv_path.name} (sha {input_sha[:12]}, {len(cands)} rows, "
          f"{len(groups)} anchors)")
    print(f"[D4.1.1] pre-checks: conflicting duplicate keys=0, identical "
          f"collapsed={dup_report['identical_duplicate_rows_collapsed']}, "
          f"non-finite rows={nonfinite['rows_with_nonfinite']}")
    print(f"[D4.1.1] splits: " + " | ".join(
        f"{s} {metadata['dataset_stats'][s]['anchors']}a/"
        f"{metadata['dataset_stats'][s]['candidate_rows']}r"
        for s in sorted(metadata["dataset_stats"])))
    print(f"[D4.1.1] invariants: "
          f"{sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)} PASS"
          " (incl. row-order invariance)")
    if any(val_sum.values()):
        print("[D4.1.1] val Top-1 (n=%d): " % (val_sum["kin"] or {}).get("n_anchors", 0)
              + " | ".join(f"{m} {val_sum[m]['n_correct']}" for m in METHODS
                           if val_sum[m]))
        print("[D4.1.1] val coverage@%.2f (accepted=%s): " % (FIXED_COVERAGE,
              (cov50["kin"] or {}).get("accepted"))
              + " | ".join(f"{m} wrong {cov50[m]['wrong_among_accepted']}"
                           for m in METHODS if cov50[m]))
    if args.reference_check == "on":
        n_pass = sum(1 for c in ref_result["checks"] if c["pass"])
        print(f"[D4.1.1] reference regression ({val}): "
              f"{n_pass}/{len(ref_result['checks'])} "
              f"{'PASS' if not ref_failed else 'FAIL'}")
    print(f"[D4.1.1] outputs -> {out_dir}")

    print("[D4.1.1] RESULT: PASS")
    log_lines.append("RESULT: PASS")
    return finish("completed", EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-tests for the D4.1.1 fixed evaluator — run BEFORE the evaluation.

T1  two anchors, same source        -> two independent Top-1 decisions
T2  same anchor_id, different source-> groups must not mix
T3  score directions                -> kin asc (smaller better), cos desc,
                                       joint = cos - kin (higher better)
T4  duplicate candidates            -> conflicting keys raise; identical rows
                                       collapse (no inflated correct counts)
T5  per-source stats                -> correct_count <= n_anchors, violation
                                       raises
T6  fixed coverage                  -> 60 anchors @ 0.5 -> exactly 30 accepted
T7  row-order invariance            -> seeded shuffle changes nothing
T8  seven-method isolation          -> each method's result comes from its own
                                       scores (not one dict copied 7 times)
T9  run-discipline guard            -> same experiment refuses to re-run
T10 reference-mismatch release gate -> exit 3, diagnostics only, NO formal
                                       result files published
T11 run_id identity                 -> run_id = hash(code, input, config);
                                       same triple refused, changed input is
                                       a DIFFERENT experiment (never silently
                                       reused)

Usage:  python3 -I src/selftest_d4_1_1_evaluator.py   (from the repo root)

A summary is printed and written to runs/d4_1_1_integrity/selftest_results.txt.
Exit code 0 only if all tests pass.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d4_1_1_fixed_evaluator as ev  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "d4_1_1_integrity"

TESTS = []


def register(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def mkc(split="val", source="v1", anchor="a1", ped="p1", is_true=False,
        kin=0.5, static=0.5, b4=0.5, d4=0.5, row_index=0):
    return ev.Candidate(split=split, source=source, anchor_id=anchor, ped=ped,
                        is_true=is_true, kin=kin, static_cos=static,
                        b4_cos=b4, d4_cos=d4, row_index=row_index)


# ------------------------------- T1 ----------------------------------------
@register("T1_two_anchors_same_source")
def t1():
    rows = [
        mkc(anchor="a1", ped="p1", is_true=True, kin=0.2, static=0.55),
        mkc(anchor="a1", ped="p2", is_true=False, kin=0.9, static=0.95),
        mkc(anchor="a2", ped="p3", is_true=True, kin=0.3, static=0.50),
        mkc(anchor="a2", ped="p4", is_true=False, kin=0.4, static=0.99),
    ]
    groups = ev.group_candidates(rows)
    dec_kin = ev.evaluate_method(groups, "kin")
    dec_static = ev.evaluate_method(groups, "static")
    kin_pick = {d.anchor_id: d.chosen_ped for d in dec_kin}
    static_pick = {d.anchor_id: d.chosen_ped for d in dec_static}
    ok = (len(groups) == 2 and len(dec_kin) == 2
          and kin_pick == {"a1": "p1", "a2": "p3"}
          and all(d.is_true for d in dec_kin)
          and static_pick == {"a1": "p2", "a2": "p4"}
          and not any(d.is_true for d in dec_static))
    return ok, (f"kin picks p1/p3 (both true), static independently picks "
                f"p2/p4 (both false); 2 decisions for 2 anchors, no mixing")


# ------------------------------- T2 ----------------------------------------
@register("T2_same_anchor_id_different_sources")
def t2():
    rows = [
        mkc(source="v1", anchor="a1", ped="pA", is_true=False, kin=0.9),
        mkc(source="v1", anchor="a1", ped="pB", is_true=True, kin=0.1),
        mkc(source="v2", anchor="a1", ped="q1", is_true=True, kin=0.2),
        mkc(source="v2", anchor="a1", ped="q2", is_true=False, kin=0.8),
    ]
    groups = ev.group_candidates(rows)
    dec = ev.evaluate_method(groups, "kin")
    got = {(d.source, d.anchor_id): d.chosen_ped for d in dec}
    ok = (len(groups) == 2
          and got == {("v1", "a1"): "pB", ("v2", "a1"): "q1"})
    return ok, ("v1/a1 -> pB, v2/a1 -> q1; the global best (pB, kin=0.1) is "
                "not leaked into v2's group; 2 groups")


# ------------------------------- T3 ----------------------------------------
@register("T3_score_directions")
def t3():
    small_kin = mkc(ped="x", kin=0.1, static=0.5, b4=0.5, d4=0.5)
    big_kin = mkc(ped="x", kin=0.5, static=0.5, b4=0.5, d4=0.5)
    big_cos = mkc(ped="x", kin=0.3, static=0.9, b4=0.9, d4=0.9)
    small_cos = mkc(ped="x", kin=0.3, static=0.5, b4=0.5, d4=0.5)
    joint_a = mkc(ped="ja", kin=0.20, static=0.80, b4=0.80, d4=0.80)
    joint_b = mkc(ped="jb", kin=0.50, static=0.85, b4=0.85, d4=0.85)
    ok = (ev.method_score("kin", small_kin) > ev.method_score("kin", big_kin)
          and all(ev.method_score(m, big_cos) > ev.method_score(m, small_cos)
                  for m in ("static", "b4", "d4"))
          and all(ev.method_score(m, joint_a) > ev.method_score(m, joint_b)
                  for m in ("kin_static", "kin_b4", "kin_d4"))
          and all(ev.method_score(m, joint_b) > ev.method_score(m, joint_a)
                  for m in ("static", "b4", "d4")))
    groups = ev.group_candidates([joint_a, joint_b])
    d_joint = {m: ev.evaluate_method(groups, m)[0].chosen_ped
               for m in ("kin_static", "kin_b4", "kin_d4")}
    d_pure = {m: ev.evaluate_method(groups, m)[0].chosen_ped
              for m in ("static", "b4", "d4")}
    ok = (ok
          and d_joint == {"kin_static": "ja", "kin_b4": "ja", "kin_d4": "ja"}
          and d_pure == {"static": "jb", "b4": "jb", "d4": "jb"})
    return ok, ("kin: smaller distance wins; cos: larger wins; joint "
                "cos - kin: 0.80-0.20 beats 0.85-0.50, but pure cos flips")


# ------------------------------- T4 ----------------------------------------
@register("T4_duplicate_candidates")
def t4():
    base = dict(split="val", source="v1", anchor="a1", ped="p1", is_true=True,
                kin=0.2, static=0.9, b4=0.9, d4=0.9)
    conflicting = [mkc(**base, row_index=1), mkc(**{**base, "kin": 0.7}, row_index=2)]
    raised = False
    try:
        ev.check_duplicates(conflicting)
    except ev.DuplicateCandidateError:
        raised = True
    identical = [mkc(**base, row_index=1), mkc(**base, row_index=2)]
    unique, report = ev.check_duplicates(identical)
    groups = ev.group_candidates(unique)
    dec = ev.evaluate_method(groups, "kin")
    ok = (raised and len(unique) == 1 and dec[0].is_true
          and report["identical_duplicate_rows_collapsed"] == 1
          and report["conflicting_duplicate_keys"] == 0)
    return ok, (f"conflicting duplicate key raises "
                f"DuplicateCandidateError={raised}; identical duplicate "
                f"collapses to 1 row (correct count not inflated)")


# ------------------------------- T5 ----------------------------------------
@register("T5_source_stats_bound")
def t5():
    rows = []
    for a, (kin_true, kin_false) in (("a1", (0.1, 0.9)), ("a2", (0.1, 0.9)),
                                     ("a3", (0.9, 0.1))):
        rows.append(mkc(anchor=a, ped="t", is_true=True, kin=kin_true))
        rows.append(mkc(anchor=a, ped="f", is_true=False, kin=kin_false))
    groups = ev.group_candidates(rows)
    dec = {m: ev.evaluate_method(groups, m) for m in ev.METHODS}
    ps = ev.per_source_stats(groups, dec)
    row = ps[0]
    ok = (row["n_anchors"] == 3 and row["kin_correct"] == 2
          and all(row[f"{m}_correct"] <= row["n_anchors"] for m in ev.METHODS))
    raised = False
    try:
        ev.assert_source_bound([{"split": "val", "source": "vX", "n_anchors": 3,
                                 **{f"{m}_correct": 4 for m in ev.METHODS}}])
    except ev.InvariantViolation:
        raised = True
    return ok and raised, ("v1: 3 anchors, kin_correct=2 (<= 3); a fabricated "
                           "correct=4 > 3 raises InvariantViolation")


# ------------------------------- T6 ----------------------------------------
@register("T6_fixed_coverage")
def t6():
    # 60 anchors x 4 candidates = 240 rows.  Every anchor's Top-1 (ped p0)
    # carries the anchor's confidence; anchor ranks 0..11 are wrong and all
    # sit inside the top-30 ANCHOR decisions.  A candidate-row implementation
    # would accept 120 rows, so this fixture discriminates the anchor-count
    # denominator from the P0-forbidden candidate-row denominator.
    rows = []
    for i in range(60):
        top_true = not (i < 12)
        conf = 1.0 - i * 0.001
        for k in range(4):
            rows.append(mkc(source="v1", anchor=f"a{i:02d}", ped=f"p{k}",
                            is_true=(top_true and k == 0),
                            static=conf - k * 0.0001))
    groups = ev.group_candidates(rows)
    dec = ev.evaluate_method(groups, "static")
    blk = ev.coverage_block(dec, 0.5)
    ranked = ev.rank_by_confidence(dec)
    accepted = ranked[:blk["accepted"]]
    ok = (len(rows) == 240 and len(dec) == 60
          and all(d.n_candidates == 4 for d in dec)
          and blk["accepted"] == 30
          and blk["accepted_wrong"] == 12
          and blk["accepted_correct"] == 18
          and blk["wrong_among_accepted"] == 12
          and blk["wrong_among_accepted_fraction"] == round(12 / 30, 6)
          and blk["coverage"] == 0.5
          and blk["top1_accuracy"] == round(48 / 60, 6)
          and blk["boundary_tie"] is False
          and len(accepted) == 30
          and len({d.anchor_id for d in accepted}) == 30)
    return ok, (f"60 anchors x 4 candidates (240 rows) @ 0.5 -> accepted="
                f"{blk['accepted']} DISTINCT anchors (a row-based "
                f"implementation would accept 120); "
                f"wrong_among_accepted={blk['wrong_among_accepted']}, "
                f"accepted_correct={blk['accepted_correct']}")


# ------------------------------- T7 ----------------------------------------
@register("T7_row_order_invariance")
def t7():
    rng = random.Random(20261106)
    rows = []
    n = 0
    for src in ("v1", "v2", "v3"):
        for a in range(6):
            for k in range(4):
                rows.append(mkc(source=src, anchor=f"a{a}", ped=f"p{k}",
                                is_true=(k == 0),
                                kin=round(rng.uniform(0.0, 1.0), 4),
                                static=round(rng.uniform(0.0, 1.0), 4),
                                b4=round(rng.uniform(0.0, 1.0), 4),
                                d4=round(rng.uniform(0.0, 1.0), 4),
                                row_index=n))
                n += 1

    def full(row_list):
        unique, _ = ev.check_duplicates(row_list)
        groups = ev.group_candidates(unique)
        dec = {m: ev.evaluate_method(groups, m) for m in ev.METHODS}
        proj = ev.results_projection(dec)
        ps = ev.per_source_stats(groups, dec)
        cov = {m: {c: ev.coverage_block(dec[m], c)
                   for c in (0.3, 0.5, 0.8)}
               for m in ev.METHODS}
        return json.dumps({"proj": proj, "per_source": ps, "coverage": cov},
                          sort_keys=True)

    a = full(rows)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    b = full(shuffled)
    return a == b, (f"{len(rows)} rows / 18 anchors x 7 methods; original vs "
                    f"seeded-shuffle results identical: {a == b}")


# ------------------------------- T8 ----------------------------------------
@register("T8_seven_method_isolation")
def t8():
    # One anchor, 7 candidates, hand-tuned so each method has a DIFFERENT
    # argmax (verified by hand): kin->c1, static->c2, kin_static->c3,
    # b4->c4, kin_b4->c5, d4->c6, kin_d4->c7.  Only c7 is the true match.
    rows = [
        mkc(ped="c1", kin=0.05, static=0.20, b4=0.20, d4=0.20, is_true=False),
        mkc(ped="c2", kin=0.30, static=0.97, b4=0.30, d4=0.30, is_true=False),
        mkc(ped="c3", kin=0.10, static=0.95, b4=0.10, d4=0.10, is_true=False),
        mkc(ped="c4", kin=0.35, static=0.30, b4=0.98, d4=0.30, is_true=False),
        mkc(ped="c5", kin=0.08, static=0.25, b4=0.96, d4=0.25, is_true=False),
        mkc(ped="c6", kin=0.40, static=0.35, b4=0.35, d4=0.99, is_true=False),
        mkc(ped="c7", kin=0.06, static=0.10, b4=0.10, d4=0.97, is_true=True),
    ]
    expected = {"kin": "c1", "static": "c2", "kin_static": "c3", "b4": "c4",
                "kin_b4": "c5", "d4": "c6", "kin_d4": "c7"}
    groups = ev.group_candidates(rows)
    got = {m: ev.evaluate_method(groups, m)[0].chosen_ped for m in ev.METHODS}
    corrects = {m: int(ev.evaluate_method(groups, m)[0].is_true)
                for m in ev.METHODS}
    ok = (got == expected
          and len(set(got.values())) == 7
          and corrects == {m: (1 if m == "kin_d4" else 0) for m in ev.METHODS})
    return ok, (f"winners {got}; only kin_d4 is correct -> the seven results "
                f"provably come from seven different scorings, not one dict "
                f"copied into seven names")


# ------------------------------- T9 ----------------------------------------
@register("T9_run_discipline_guard")
def t9():
    tmp = Path(tempfile.mkdtemp(prefix="d411_selftest_"))
    try:
        csv_path = tmp / "mini.csv"
        fields = ["anchor_id", "split", "source", "ped", "is_true", "kin",
                  "static_cos", "b4_cos", "d4_cos", "tier", "delta_s"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for a in range(4):
                for k, (kin, cos) in enumerate([(0.1, 0.9), (0.5, 0.7),
                                                (0.9, 0.99)]):
                    w.writerow({"anchor_id": f"{a:03d}", "split": "val",
                                "source": "vT", "ped": f"p{k}",
                                "is_true": int(k == 0), "kin": kin,
                                "static_cos": cos, "b4_cos": cos,
                                "d4_cos": cos, "tier": 0, "delta_s": 1.0})
        script = str(Path(__file__).resolve().parent
                     / "run_d4_1_1_fixed_evaluator.py")
        cmd = [sys.executable, "-I", script, "--csv", str(csv_path),
               "--out-dir", str(tmp / "run1"), "--reference-check", "off"]
        r1 = subprocess.run(cmd, capture_output=True, text=True)
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        guard = (r1.returncode == 0
                 and r2.returncode == ev.EXIT_RUN_GUARD
                 and "refus" in (r2.stderr + r2.stdout).lower())
        out1 = tmp / "run1"
        required = ("seven_method_results.json", "per_anchor_predictions.csv",
                    "per_source_results.csv", "loso_relative_gains.json",
                    "risk_coverage.json", "invariant_checks.json",
                    "reference_check.json", "run_manifest.json",
                    "evaluation_run_log.txt")
        files_ok = all((out1 / n).exists() for n in required)
        n_rows = (sum(1 for _ in (out1 / "per_anchor_predictions.csv").open())
                  - 1) if (out1 / "per_anchor_predictions.csv").exists() else -1
        cov = (json.loads((out1 / "risk_coverage.json").read_text())
               if (out1 / "risk_coverage.json").exists() else {})
        cov_ok = (cov.get("splits", {}).get("val", {}).get("kin", {})
                  .get("coverage_0.5", {}).get("accepted") == 2)
        return guard and files_ok and n_rows == 28 and cov_ok, (
            f"CLI run 1 exit={r1.returncode}, re-run exit={r2.returncode} "
            f"(guard refused: {guard}); required files present={files_ok}; "
            f"per-anchor rows={n_rows} (4 anchors x 7 methods); "
            f"coverage@0.5 accepted=2 of 4 anchors in risk_coverage.json "
            f"(a row-based implementation would report 6): {cov_ok}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------- T10 ---------------------------------------
@register("T10_reference_mismatch_release_gate")
def t10():
    # Manufactured reference mismatch: clean synthetic data passes every
    # invariant, but its val numbers cannot match the external reference
    # recomputation -> the release gate must publish NOTHING formal.
    tmp = Path(tempfile.mkdtemp(prefix="d411_t10_"))
    try:
        csv_path = tmp / "mini_refail.csv"
        fields = ["anchor_id", "split", "source", "ped", "is_true", "kin",
                  "static_cos", "b4_cos", "d4_cos", "tier", "delta_s"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for a in range(4):
                for k, (kin, cos) in enumerate([(0.1, 0.9), (0.5, 0.7),
                                                (0.9, 0.99)]):
                    w.writerow({"anchor_id": f"{a:03d}", "split": "val",
                                "source": "vT", "ped": f"p{k}",
                                "is_true": int(k == 0), "kin": kin,
                                "static_cos": cos, "b4_cos": cos,
                                "d4_cos": cos, "tier": 0, "delta_s": 1.0})
        script = str(Path(__file__).resolve().parent
                     / "run_d4_1_1_fixed_evaluator.py")
        out = tmp / "run"
        # default --reference-check on -> mismatch guaranteed
        r = subprocess.run([sys.executable, "-I", script, "--csv",
                            str(csv_path), "--out-dir", str(out)],
                           capture_output=True, text=True)
        formal = ["seven_method_results.json", "per_anchor_predictions.csv",
                  "per_source_results.csv", "loso_relative_gains.json",
                  "risk_coverage.json"]
        no_formal = not any((out / n).exists() for n in formal)
        man = (json.loads((out / "run_manifest.json").read_text())
               if (out / "run_manifest.json").exists() else {})
        refc = (json.loads((out / "reference_check.json").read_text())
                if (out / "reference_check.json").exists() else {})
        ok = (r.returncode == 3
              and man.get("status") == "reference_mismatch"
              and man.get("reference_check", {}).get("status") == "FAIL"
              and man.get("exit_code") == 3
              and refc.get("status") == "FAIL"
              and (out / "reference_mismatch_details.json").exists()
              and (out / "invariant_checks.json").exists()
              and no_formal
              and not list(out.glob(".staging*")))
        return ok, (f"exit={r.returncode}; manifest={man.get('status')} "
                    f"(ref check {man.get('reference_check', {}).get('status')}); "
                    f"formal files published={not no_formal}; diagnostics kept: "
                    f"reference_check.json, reference_mismatch_details.json, "
                    f"invariant_checks.json, run log, manifest")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------- T11 ---------------------------------------
@register("T11_run_id_experiment_identity")
def t11():
    tmp = Path(tempfile.mkdtemp(prefix="d411_t11_"))
    try:
        def write_csv(path, kin_first):
            fields = ["anchor_id", "split", "source", "ped", "is_true", "kin",
                      "static_cos", "b4_cos", "d4_cos", "tier", "delta_s"]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for a in range(4):
                    for k, (kin, cos) in enumerate([(kin_first, 0.9), (0.5, 0.7),
                                                    (0.9, 0.99)]):
                        w.writerow({"anchor_id": f"{a:03d}", "split": "val",
                                    "source": "vT", "ped": f"p{k}",
                                    "is_true": int(k == 0), "kin": kin,
                                    "static_cos": cos, "b4_cos": cos,
                                    "d4_cos": cos, "tier": 0, "delta_s": 1.0})

        csv_a, csv_b = tmp / "a.csv", tmp / "b.csv"
        write_csv(csv_a, kin_first=0.1)
        write_csv(csv_b, kin_first=0.2)  # different input content
        script = str(Path(__file__).resolve().parent
                     / "run_d4_1_1_fixed_evaluator.py")
        base = [sys.executable, "-I", script, "--reference-check", "off"]
        out_a, out_b = tmp / "runA", tmp / "runB"
        r1 = subprocess.run(base + ["--csv", str(csv_a), "--out-dir", str(out_a)],
                            capture_output=True, text=True)
        r2 = subprocess.run(base + ["--csv", str(csv_a), "--out-dir", str(out_a)],
                            capture_output=True, text=True)
        # changed input into the SAME dir -> must be refused as a DIFFERENT
        # experiment, not silently mixed or reused
        r3 = subprocess.run(base + ["--csv", str(csv_b), "--out-dir", str(out_a)],
                            capture_output=True, text=True)
        # changed input in a FRESH dir -> new experiment, runs normally
        r4 = subprocess.run(base + ["--csv", str(csv_b), "--out-dir", str(out_b)],
                            capture_output=True, text=True)
        man_a = json.loads((out_a / "run_manifest.json").read_text())
        man_b = json.loads((out_b / "run_manifest.json").read_text())
        keys = ("run_id", "code_version", "input", "config", "config_sha256",
                "reference_check", "started_at", "finished_at", "status")
        complete = all(k in man_b for k in keys)
        distinct = (man_a["run_id"] != man_b["run_id"]
                    and man_a["input"]["sha256"] != man_b["input"]["sha256"]
                    and man_a["code_version"]["sha256"]
                    == man_b["code_version"]["sha256"])
        # cross-output-dir registry (frozen-input single-evaluation protection)
        reg = tmp / "registry.json"
        ev.registry_record(reg, "rid_1", tmp / "dirA", "completed")
        ev.registry_record(reg, "rid_2", tmp / "dirB", "running")
        ev.registry_record(reg, "rid_1", tmp / "dirA", "reference_mismatch")
        entries = json.loads(reg.read_text())["runs"]
        reg_ok = (ev.registry_lookup(reg, "rid_1")["status"]
                  == "reference_mismatch"
                  and ev.registry_lookup(reg, "rid_2")["status"] == "running"
                  and ev.registry_lookup(reg, "rid_missing") is None
                  and len(entries) == 2
                  and {e["run_id"] for e in entries} == {"rid_1", "rid_2"})
        ok = (r1.returncode == 0
              and r2.returncode == 4 and "same experiment" in r2.stderr
              and r3.returncode == 4 and "DIFFERENT experiment" in r3.stderr
              and r4.returncode == 0 and man_b["status"] == "completed"
              and distinct and complete and reg_ok)
        return ok, (f"same (code,input,config) re-run refused (exit={r2.returncode}); "
                    f"changed input in same dir refused as DIFFERENT experiment "
                    f"(exit={r3.returncode}); changed input in fresh dir ran as a "
                    f"new experiment (exit={r4.returncode}), run_id "
                    f"{man_a['run_id'][:8]}.. != {man_b['run_id'][:8]}..; manifest "
                    f"records code/input/config/reference-check/start/end/status: "
                    f"{complete}; cross-dir registry upsert/lookup semantics: "
                    f"{reg_ok}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------- main -------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["D4.1.1 evaluator selftest",
             f"run at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
             f"evaluator code sha256: {ev.code_sha256()[:12]}", ""]
    n_pass = 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"exception: {type(exc).__name__}: {exc}"
        n_pass += int(ok)
        line = f"{name}: {'PASS' if ok else 'FAIL'} — {detail}"
        print(line)
        lines.append(line)
    summary = f"SUMMARY: {n_pass}/{len(TESTS)} PASS"
    print(summary)
    lines += ["", summary]
    (OUT_DIR / "selftest_results.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    print(f"[selftest] results -> {OUT_DIR / 'selftest_results.txt'}")
    return 0 if n_pass == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

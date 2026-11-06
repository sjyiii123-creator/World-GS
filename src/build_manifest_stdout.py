"""Standalone wrapper: build the SynPeDAS manifest and print CSV to stdout.

Read-only w.r.t. the filesystem (writes to /dev/stdout); the actual CSV is
persisted separately.  Run from the project root:

    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_manifest_stdout.py
"""
import sys

sys.path.insert(0, "src")

from wmeval.build_manifest import build_manifest  # noqa: E402

build_manifest("repos/SynPeDAS", "/dev/stdout", probe_videos=True)

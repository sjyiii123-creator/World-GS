"""Entry wrapper for wmeval.run_minimal in isolated-mode script form.

Usage (from project root):
    PYTHONPATH-independent: extra CLI args are forwarded verbatim.
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_minimal_entry.py \
        --videos <path> --weights ckpt/yolov10x.pt --outdir runs/smoke_one
"""
import sys

sys.path.insert(0, "src")

from wmeval.run_minimal import main  # noqa: E402

main()

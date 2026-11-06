"""wmeval — World-Model event-level EValuation toolkit.

Uncertainty-aware safety-critical event consistency verification for
generative driving world models (master's thesis research framework).

Modules
-------
video_io      A. video reading (OpenCV, configurable fps/res, timestamps)
detect_track  B. object detection + multi-object tracking (YOLO + ByteTrack)
traj_metrics  faithful NumPy port of DrivingGen official trajectory metrics
event_checks  C. event constraint verification (presence/region/time/order/disappearance)
uncertainty   D. uncertainty-aware adjudication (Valid / Invalid / Uncertain)
build_manifest SynPeDAS sample manifest builder
run_minimal   minimal end-to-end experiment entry point
"""

__version__ = "0.1.0"

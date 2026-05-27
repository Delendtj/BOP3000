import argparse
import os
import sys
import cv2
import numpy as np
import supervision as sv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from functions.system.hardware_detector import HardwareDetector
from trackers import ByteTrackTracker, OCSORTTracker
from trackers.core.sort.tracker import SORTTracker

# Tracker choices: "bytetrack", "ocsort", or "sort"
TRACKER_LEFT = "ocsort"
TRACKER_RIGHT = "bytetrack"

AVAILABLE_TRACKERS = {
    "bytetrack": {
        "class": ByteTrackTracker,
        "kwargs": {
            "lost_track_buffer": 30,
            "track_activation_threshold": 0.7,
            "minimum_consecutive_frames": 2,
            "minimum_iou_threshold": 0.1,
            "high_conf_det_threshold": 0.6,
        },
    },
    "ocsort": {
        "class": OCSORTTracker,
        "kwargs": {
            "lost_track_buffer": 30,
            "minimum_consecutive_frames": 3,
            "minimum_iou_threshold": 0.3,
            "direction_consistency_weight": 0.2,
            "high_conf_det_threshold": 0.6,
            "delta_t": 3,
        },
    },
    "sort": {
        "class": SORTTracker,
        "kwargs": {
            "lost_track_buffer": 30,
            "track_activation_threshold": 0.25,
            "minimum_consecutive_frames": 3,
            "minimum_iou_threshold": 0.3,
        },
    },
}

# Model config (mirrors main.py defaults)
CONFIG = {
    "Model_OV_path": "models/best_openvino_model",
    "Model_PT_path": "models/1280.pt",
    "Tensor_engine_path": "models/1280.engine",
    "USE_FP16": True,
    "IMGSZ": 1280,
}

# Force backend: "openvino", "pytorch", or "cuda". Set to None to auto-detect.
FORCE_BACKEND = "openvino"

INFERENCE_CONFIG = {
    "conf": 0.2,
    "iou": 0.5,
    "max_det": 100,
    "imgsz": 1280,
    "half": False,
    "device": None,
    "verbose": False,
}

DISPLAY_PANEL_SIZE = (960, 540)
WINDOW_NAME = "Tracker Compare"


def build_display_panel(frame, title, panel_size):
    panel_w, panel_h = panel_size
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    if frame is not None:
        src_h, src_w = frame.shape[:2]
        scale = min(panel_w / src_w, panel_h / src_h)
        resized_w = max(1, int(src_w * scale))
        resized_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame, (resized_w, resized_h))
        x = (panel_w - resized_w) // 2
        y = (panel_h - resized_h) // 2
        panel[y:y + resized_h, x:x + resized_w] = resized

    cv2.putText(
        panel,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return panel


def make_tracker(kind, frame_rate):
    kind = kind.lower().strip()
    spec = AVAILABLE_TRACKERS.get(kind)
    if spec is None:
        options = ", ".join(sorted(AVAILABLE_TRACKERS.keys()))
        raise ValueError(f"Unknown tracker kind: {kind}. Available: {options}")
    kwargs = dict(spec["kwargs"])
    kwargs["frame_rate"] = frame_rate
    return spec["class"](**kwargs)


def annotate_frame(frame, tracks):
    if frame is None:
        return None
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(frame, tracks)
    labels = []
    if len(tracks) > 0:
        for tid in tracks.tracker_id:
            labels.append(f"ID {tid}")
    if labels:
        annotated = label_annotator.annotate(annotated, tracks, labels=labels)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Compare two trackers side-by-side on the same video.")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    detector = HardwareDetector(CONFIG)
    if FORCE_BACKEND:
        detector.hardware_type = FORCE_BACKEND
    model = detector.initialize_model()

    tracker_left = make_tracker(TRACKER_LEFT, fps)
    tracker_right = make_tracker(TRACKER_RIGHT, fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(
                frame,
                conf=INFERENCE_CONFIG["conf"],
                iou=INFERENCE_CONFIG["iou"],
                max_det=INFERENCE_CONFIG["max_det"],
                imgsz=INFERENCE_CONFIG["imgsz"],
                half=INFERENCE_CONFIG["half"],
                device=INFERENCE_CONFIG["device"],
                verbose=INFERENCE_CONFIG["verbose"],
            )[0]

            detections = sv.Detections.from_ultralytics(result)

            left_tracks = tracker_left.update(detections)
            right_tracks = tracker_right.update(detections)

            left_annotated = annotate_frame(frame.copy(), left_tracks)
            right_annotated = annotate_frame(frame.copy(), right_tracks)

            left_panel = build_display_panel(left_annotated, f"{TRACKER_LEFT.upper()}", DISPLAY_PANEL_SIZE)
            right_panel = build_display_panel(right_annotated, f"{TRACKER_RIGHT.upper()}", DISPLAY_PANEL_SIZE)
            canvas = np.hstack([left_panel, right_panel])

            cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

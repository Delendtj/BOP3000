import cv2
import os
import time
import tkinter as tk

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from collections import defaultdict, Counter

# Main program functions
from functions.roi import load_roi, save_roi, select_roi
from functions.tracker import Tracker
from hardware_detector import HardwareDetector

config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_PT_path': "models/1280.pt",
    'Tensor_engine_path': "models/1280.engine",
    'USE_FP16': True,
    'IMGSZ': 1280,
}

DATA_PATH = "../videos/DJI_CUT.MP4"
CONF_THRESHOLD = 0.3
FRAME_SKIP = 1
OCR_FRAMES = 3          # collect votes for N frames before deciding
# INSERT ACTUAL FORMULA HERE
NUMBER_OF_THREADS = 2

INFERENCE_CONFIG = {
    'conf': CONF_THRESHOLD,
    'iou': 0.5,
    'max_det': 300,
    'imgsz': 1280,
    'half': False, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

ROI_PATH = os.path.join("img", "detection_roi.json")

detector = HardwareDetector(config)
model = detector.initialize_model()

# Screen resolution for window sizing
root = tk.Tk()
system_width = root.winfo_screenwidth()
system_height = root.winfo_screenheight()
root.destroy()

# Open video
cap = cv2.VideoCapture(DATA_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, preview_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read initial frame for ROI.")

roi = load_roi(ROI_PATH)
if roi is None:
    roi = select_roi(preview_frame)
    if roi is not None:
        save_roi(ROI_PATH, roi)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.namedWindow('Yolo vision', cv2.WINDOW_NORMAL)

# THE TRACKER
tracker = Tracker(OCR_FRAMES, CONF_THRESHOLD, roi)

prev_frame_time = None
fps_ema = 0.0
frame_count = 0
helmet_saved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        result = model(
            frame,
            conf=INFERENCE_CONFIG['conf'],
            iou=INFERENCE_CONFIG['iou'],
            max_det=INFERENCE_CONFIG['max_det'],
            imgsz=INFERENCE_CONFIG['imgsz'],
            half=INFERENCE_CONFIG['half'],
            device=INFERENCE_CONFIG['device'],
            verbose=INFERENCE_CONFIG['verbose']
        )[0]

        detections = sv.Detections.from_ultralytics(result)


        tracker.track_detection(detections, frame)

    else:
        pass

    # Annotate frames
    annotated = tracker.annotate(frame)

    if roi is not None:
        cv2.polylines(
            annotated,
            [np.array(roi, dtype=np.int32)],
            True,
            (0, 255, 255),
            2,
        )

    if roi is not None:
        cv2.polylines(
            annotated,
            [np.array(roi, dtype=np.int32)],
            True,
            (0, 255, 255),
            2,
        )

    now = time.perf_counter()
    if prev_frame_time is not None:
        elapsed = now - prev_frame_time
        if elapsed > 0:
            fps_inst = 1.0 / elapsed
            fps_ema = fps_inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * fps_inst)
    prev_frame_time = now

    fps_text = f"FPS: {fps_ema:.1f}" if fps_ema > 0 else "FPS: --"
    cv2.putText(
        annotated,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    display_frame = cv2.resize(annotated, (1920, 1080))
    cv2.imshow('Yolo vision', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("r"):
        new_roi = select_roi(frame)
        if new_roi is not None:
            roi = new_roi
            save_roi(ROI_PATH, roi)

cap.release()
cv2.destroyAllWindows()
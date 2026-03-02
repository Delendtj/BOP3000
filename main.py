import cv2
import os
import time
import tkinter as tk

import numpy as np
import supervision as sv
from functions.Inference_roi import (
    keep_detections_inside_roi,
    shift_detections_to_full_frame,
)
from functions.roi import load_roi, roi_inside_roi, save_roi, select_roi
from functions.tracker import Tracker
from hardware_detector import HardwareDetector
from pipeline.async_pipeline import AsyncFramePipeline

config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_PT_path': "models/1280.pt",
    'Tensor_engine_path': "models/1280.engine",
    'USE_FP16': True,
    'IMGSZ': 1280,
}

DATA_PATH = "DJI_20260228140513_0010_D.MP4"
CONF_THRESHOLD = 0.3
FRAME_SKIP = 1
OCR_FRAMES = 3          # collect votes for N frames before deciding
# INSERT ACTUAL FORMULA HERE
NUMBER_OF_THREADS = 2

INFERENCE_CONFIG = {
    'conf': CONF_THRESHOLD,
    'iou': 0.5,
    'max_det': 100,
    'imgsz': 1280,
    'half': False, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

YOLO_ROI_PATH = os.path.join("img", "yolo_roi.json")
OCR_ROI_PATH = os.path.join("img", "ocr_roi.json")
LEGACY_ROI_PATH = os.path.join("img", "detection_roi.json")

detector = HardwareDetector(config)
model = detector.initialize_model()


def select_ocr_roi_inside_yolo(frame, yolo_roi):
    if frame is None or yolo_roi is None:
        return None

    while True:
        candidate = select_roi(frame, window_name="OCR ROI Selector")
        if candidate is None:
            return None
        if roi_inside_roi(candidate, yolo_roi):
            return candidate
        print("OCR ROI must be inside YOLO ROI. Draw again or press Esc to cancel.")

# Screen resolution for window sizing
root = tk.Tk()
system_width = root.winfo_screenwidth()
system_height = root.winfo_screenheight()
root.destroy()

# Open video
preview_cap = cv2.VideoCapture(DATA_PATH)
fps = preview_cap.get(cv2.CAP_PROP_FPS)
total_frames = int(preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, preview_frame = preview_cap.read()
preview_cap.release()
if not ret:
    raise RuntimeError("Could not read initial frame for ROI.")

yolo_roi = load_roi(YOLO_ROI_PATH)
if yolo_roi is None:
    yolo_roi = load_roi(LEGACY_ROI_PATH)
if yolo_roi is None:
    yolo_roi = select_roi(preview_frame, window_name="YOLO ROI Selector")
    if yolo_roi is not None:
        save_roi(YOLO_ROI_PATH, yolo_roi)
        save_roi(LEGACY_ROI_PATH, yolo_roi)

ocr_roi = load_roi(OCR_ROI_PATH)
if ocr_roi is not None and yolo_roi is not None and not roi_inside_roi(ocr_roi, yolo_roi):
    print("Loaded OCR ROI is outside YOLO ROI. Please redraw OCR ROI.")
    ocr_roi = None

if ocr_roi is None and yolo_roi is not None:
    selected_ocr = select_ocr_roi_inside_yolo(preview_frame, yolo_roi)
    if selected_ocr is not None:
        ocr_roi = selected_ocr
        save_roi(OCR_ROI_PATH, ocr_roi)

cv2.namedWindow('Yolo vision', cv2.WINDOW_NORMAL)

# THE TRACKER
tracker = Tracker(OCR_FRAMES, CONF_THRESHOLD, ocr_roi, frame_rate=fps)
pipeline = AsyncFramePipeline(
    source=DATA_PATH,
    frame_skip=FRAME_SKIP,
    queue_size=3,
    inference_roi=yolo_roi,
)
pipeline.start()

prev_frame_time = None
fps_ema = 0.0
frame_count = 0
helmet_saved = False

try:
    while True:
        item = pipeline.read(timeout=0.5)
        if item is None:
            if pipeline.stop_event.is_set():
                break
            continue

        frame = item.frame
        inference_frame = item.inference_frame if item.inference_frame is not None else frame
        inference_offset = item.inference_offset

        frame_count += 1

        result = model(
            inference_frame,
            conf=INFERENCE_CONFIG['conf'],
            iou=INFERENCE_CONFIG['iou'],
            max_det=INFERENCE_CONFIG['max_det'],
            imgsz=INFERENCE_CONFIG['imgsz'],
            half=INFERENCE_CONFIG['half'],
            device=INFERENCE_CONFIG['device'],
            verbose=INFERENCE_CONFIG['verbose']
        )[0]

        detections = sv.Detections.from_ultralytics(result)
        if yolo_roi is not None:
            detections = shift_detections_to_full_frame(detections, inference_offset)
            detections = keep_detections_inside_roi(detections, yolo_roi)

        tracker.track_detection(detections, frame)

        # Annotate frames
        annotated = tracker.annotate(frame)

        if yolo_roi is not None:
            cv2.polylines(
                annotated,
                [np.array(yolo_roi, dtype=np.int32)],
                True,
                (0, 255, 255),
                2,
            )
        if ocr_roi is not None:
            cv2.polylines(
                annotated,
                [np.array(ocr_roi, dtype=np.int32)],
                True,
                (0, 200, 0),
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

        display_frame = cv2.resize(annotated, (1280, 720))
        cv2.imshow('Yolo vision', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            new_yolo_roi = select_roi(frame, window_name="YOLO ROI Selector")
            if new_yolo_roi is not None:
                yolo_roi = new_yolo_roi
                pipeline.set_inference_roi(yolo_roi)
                save_roi(YOLO_ROI_PATH, yolo_roi)
                save_roi(LEGACY_ROI_PATH, yolo_roi)
                if ocr_roi is not None and not roi_inside_roi(ocr_roi, yolo_roi):
                    print("Current OCR ROI is outside updated YOLO ROI. Press 'o' to redraw OCR ROI.")
                    ocr_roi = None
                    tracker.set_roi(None)
        if key == ord("o"):
            if yolo_roi is None:
                print("Define YOLO ROI first (press 'r').")
            else:
                new_ocr_roi = select_ocr_roi_inside_yolo(frame, yolo_roi)
                if new_ocr_roi is not None:
                    ocr_roi = new_ocr_roi
                    tracker.set_roi(ocr_roi)
                    save_roi(OCR_ROI_PATH, ocr_roi)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

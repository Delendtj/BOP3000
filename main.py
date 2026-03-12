from multiprocessing.spawn import freeze_support

import cv2
import os
import time
import tkinter as tk

import numpy as np
import supervision as sv

from collections import deque

from functions.BBExtractor import extract_helmet_box
from functions.Inference_roi import (
    keep_detections_inside_roi,
    shift_detections_to_full_frame,
)
from functions.homography import (
    associate_close_helmet_crops,
    build_close_crops,
    load_homography,
    select_close_frame, project_point,
)
from functions.ocr_worker import OCRWorker
from functions.roi import load_roi, roi_inside_roi, save_roi, select_roi
from functions.tracker import Tracker
from hardware_detector import HardwareDetector
from pipeline.async_pipeline import AsyncFramePipeline
from utilities.benchmark import OCRThroughputStats
from utilities.downscale_to_1080p import downscale_to_1080p

# Keep TensorRT TF32 behavior stable between engine build and execution contexts.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_PT_path': "models/1280.pt",
    'Tensor_engine_path': "models/1280.engine",
    'USE_FP16': True,
    'IMGSZ': 1280,
}

DATA_PATH = "../videos/DJI_CUT.MP4"
CLOSE_SOURCE = "../videos/canon_1.mp4"
CONF_THRESHOLD = 0.5
FRAME_SKIP = 1
OCR_VOTE = 3          # collect votes for N frames before deciding
MAX_SYNC_DELTA = 0.05
CLOSE_CROP_SIZE = 140
CLOSE_MATCH_MAX_DIST = 120
HELMET_CLASS_ID = 0
PERSON_CLASS_ID = 1
HOMOGRAPHY_PATH = os.path.join("img", "homography.json")

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


def main():
    detector = HardwareDetector(config)
    model = detector.initialize_model()

    # Screen resolution for window sizing
    root = tk.Tk()
    root.destroy()

    # Open video
    preview_cap = cv2.VideoCapture(DATA_PATH)
    fps = preview_cap.get(cv2.CAP_PROP_FPS)
    ret, preview_frame = preview_cap.read()
    preview_cap.release()

    # Downscales if it isn't 1080p
    preview_frame = downscale_to_1080p(preview_frame)

    if not ret:
        raise RuntimeError("Could not read initial frame for ROI.")

    # ROI
    yolo_roi = load_roi(YOLO_ROI_PATH)
    if yolo_roi is None:
        yolo_roi = select_roi(preview_frame, window_name="YOLO ROI Selector")
        if yolo_roi is not None:
            save_roi(YOLO_ROI_PATH, yolo_roi)

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
    cv2.namedWindow('Close vision', cv2.WINDOW_NORMAL)

    # THE TRACKER
    tracker = Tracker(OCR_VOTE, CONF_THRESHOLD, ocr_roi, frame_rate=fps)

    # Initialize threads and processes
    pipeline = AsyncFramePipeline(
        source=DATA_PATH,
        frame_skip=FRAME_SKIP,
        queue_size=3,
        inference_roi=yolo_roi,
    )
    pipeline_close = AsyncFramePipeline(
        source=CLOSE_SOURCE,
        frame_skip=FRAME_SKIP,
        queue_size=3,
        inference_roi=None,
    )

    ocr_worker = OCRWorker()
    ocr_worker.start()
    pipeline.start()
    pipeline_close.start()

    close_buffer = deque(maxlen=32)
    try:
        homography = load_homography(HOMOGRAPHY_PATH)
    except FileNotFoundError:
        homography = None
        print("Homography missing, cross-camera crops disabled.")

    # Section for initializing benchmark classes here:
    ocr_bench = OCRThroughputStats(log_every_sec=2.0)

    prev_frame_time = None
    fps_ema = 0.0
    frame_count = 0
    paused = False
    last_display_frame = None

    print("Controls: Esc=quit, Space=pause/resume, r=redraw YOLO ROI, o=redraw OCR ROI")

    try:
        while True:
            # Pause logic (With Space Bar)
            if paused and last_display_frame is not None:
                paused_frame = last_display_frame.copy()
                cv2.putText(
                    paused_frame,
                    "PAUSED (Space to resume)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow('Yolo vision', paused_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    paused = False
                continue

            while True:
                close_item = pipeline_close.read(timeout=0.01)

                if close_item is None:
                    break
                close_buffer.append((close_item.ts, close_item.frame.copy()))

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

            # Run the Tracker
            tracker.track_detection(detections)

            # Extract better crop from helmet detections and give to worker queue
            close_frame = None
            if homography is not None:
                close_frame = select_close_frame(close_buffer, item.ts, MAX_SYNC_DELTA)
            if close_frame is None and len(close_buffer) > 0:
                close_frame = close_buffer[-1][1]

            if ocr_roi is not None:
                helmet_tracks = tracker.get_non_confirmed_helmet_tracks()
                helmet_in_roi = keep_detections_inside_roi(helmet_tracks, ocr_roi)
                close_vis = None
                if close_frame is not None:
                    close_vis = close_frame.copy()

                # This whole if contains everything with homography association for helmet
                if close_frame is not None and homography is not None:
                    close_result = model(
                        close_frame,
                        conf=INFERENCE_CONFIG['conf'],
                        iou=INFERENCE_CONFIG['iou'],
                        max_det=INFERENCE_CONFIG['max_det'],
                        imgsz=INFERENCE_CONFIG['imgsz'],
                        half=INFERENCE_CONFIG['half'],
                        device=INFERENCE_CONFIG['device'],
                        verbose=INFERENCE_CONFIG['verbose']
                    )[0]
                    close_detections = sv.Detections.from_ultralytics(close_result)
                    close_helmets = close_detections[close_detections.class_id == HELMET_CLASS_ID] # Get only helmets

                    # Draw helmets on close frame
                    if close_vis is not None and len(close_helmets) > 0:
                        for bbox in close_helmets.xyxy:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(close_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Visualize homography with red dots
                    if close_vis is not None and len(tracker.people_tracks) > 0:
                        for bbox in tracker.people_tracks.xyxy:
                            center_x = float((bbox[0] + bbox[2]) / 2.0)
                            center_y = float(bbox[3])
                            projected = project_point(homography, center_x, center_y)
                            if projected is None:
                                continue
                            px, py = int(projected[0]), int(projected[1])
                            if 0 <= px < close_vis.shape[1] and 0 <= py < close_vis.shape[0]:
                                cv2.circle(close_vis, (px, py), 5, (0, 0, 255), -1)

                    helmet_crops = associate_close_helmet_crops(
                        close_helmets,
                        tracker.people_tracks,
                        close_frame,
                        homography,
                        CLOSE_MATCH_MAX_DIST,
                    )
                else:
                    # OLD: Without homography
                    helmet_crops = extract_helmet_box(helmet_in_roi, frame)

                for h in helmet_crops:
                    ocr_worker.submit(h)

            # Evaluate if the detection is good enough to be set for tracks
            helmets_res = ocr_worker.drain_results()
            tracker.check_for_ocr(helmets_res)

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

            if ocr_bench.should_log(now):
                stats = ocr_worker.get_stats()
                print(ocr_bench.format_line(stats))

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
            last_display_frame = display_frame
            cv2.imshow('Yolo vision', display_frame)
            if close_frame is not None:
                close_display = cv2.resize(close_vis if close_vis is not None else close_frame, (960, 540))
                cv2.imshow('Close vision', close_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                paused = True
                continue
            if key == ord("r"):
                new_yolo_roi = select_roi(frame, window_name="YOLO ROI Selector")
                if new_yolo_roi is not None:
                    yolo_roi = new_yolo_roi
                    pipeline.set_inference_roi(yolo_roi)
                    save_roi(YOLO_ROI_PATH, yolo_roi)
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
        ocr_worker.stop()
        pipeline.stop()
        pipeline_close.stop()
        cv2.destroyAllWindows()

# Main guard needed for multiprocessing
if __name__ == '__main__':
    freeze_support()
    main()

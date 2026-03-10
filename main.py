from multiprocessing.spawn import freeze_support

import cv2
import os
import time
import tkinter as tk
from tkinter import simpledialog

import numpy as np
import supervision as sv

from functions.BBExtractor import extract_helmet_box
from functions.Inference_roi import (
    keep_detections_inside_roi,
    shift_detections_to_full_frame,
)
from functions.lap_panel import render_lap_panel
from functions.ocr_worker import OCRWorker
from functions.roi import load_line, load_roi, roi_inside_roi, save_line, save_roi, select_line, select_roi
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

data_path = "DJI_20260228140513_0010_D.MP4"
conf_threshold = 0.5
frame_skip = 1
ocr_vote = 3          # collect votes for N frames before deciding

inference_config = {
    'conf': conf_threshold,
    'iou': 0.5,
    'max_det': 100,
    'imgsz': 1280,
    'half': True, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

yolo_roi_path = os.path.join("data", "yolo_roi.json")
ocr_roi_path = os.path.join("data", "ocr_roi.json")
finish_line_path = os.path.join("data", "finish_line.json")
display_width = 1920
display_height = 1080
lap_panel_width = 340
lap_panel_height = 720
vision_window_name = "SpeedSkate"
lap_window_name = "Lap Count"

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
def prompt_total_laps():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        while True:
            total_laps = simpledialog.askinteger(
                "Race Laps",
                "Enter total laps for this race:",
                minvalue=1,
                parent=root,
            )
            if total_laps is not None:
                return int(total_laps)
    finally:
        root.destroy()

def main():
    detector = HardwareDetector(config)
    model = detector.initialize_model()

    total_laps = prompt_total_laps()

    # Open video
    preview_cap = cv2.VideoCapture(data_path)
    fps = preview_cap.get(cv2.CAP_PROP_FPS)
    ret, preview_frame = preview_cap.read()
    preview_cap.release()

    # Downscales if it isn't 1080p
    preview_frame = downscale_to_1080p(preview_frame)

    if not ret:
        raise RuntimeError("Could not read initial frame for ROI.")

    # ROI
    yolo_roi = load_roi(yolo_roi_path)
    if yolo_roi is None:
        yolo_roi = select_roi(preview_frame, window_name="YOLO ROI Selector")
        if yolo_roi is not None:
            save_roi(yolo_roi_path, yolo_roi)

    ocr_roi = load_roi(ocr_roi_path)
    if ocr_roi is not None and yolo_roi is not None and not roi_inside_roi(ocr_roi, yolo_roi):
        print("Loaded OCR ROI is outside YOLO ROI. Please redraw OCR ROI.")
        ocr_roi = None

    if ocr_roi is None and yolo_roi is not None:
        selected_ocr = select_ocr_roi_inside_yolo(preview_frame, yolo_roi)
        if selected_ocr is not None:
            ocr_roi = selected_ocr
            save_roi(ocr_roi_path, ocr_roi)

    finish_line = load_line(finish_line_path)

    cv2.namedWindow(vision_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(vision_window_name, display_width, display_height)
    cv2.namedWindow(lap_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lap_window_name, lap_panel_width, lap_panel_height)

    # THE TRACKER
    tracker = Tracker(
        ocr_vote,
        conf_threshold,
        ocr_roi,
        frame_rate=fps,
        finish_line=finish_line,
        total_laps=total_laps,
    )

    # Initialize threads and processes
    pipeline = AsyncFramePipeline(
        source=data_path,
        frame_skip=frame_skip,
        queue_size=3,
        inference_roi=yolo_roi,
    )

    ocr_worker = OCRWorker()
    ocr_worker.start()
    pipeline.start()

    # Section for initializing benchmark classes here:
    ocr_bench = OCRThroughputStats(log_every_sec=2.0)

    prev_frame_time = None
    fps_ema = 0.0
    frame_count = 0
    paused = False
    last_display_frame = None
    last_lap_panel = None
    last_lap_panel_key = None

    print("Controls: Esc=quit, Space=pause/resume, r=redraw YOLO ROI, o=redraw OCR ROI, f=redraw finish line")

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
                cv2.imshow(vision_window_name, paused_frame)
                if last_lap_panel is not None:
                    cv2.imshow(lap_window_name, last_lap_panel)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    paused = False
                continue

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
                conf=inference_config['conf'],
                iou=inference_config['iou'],
                max_det=inference_config['max_det'],
                imgsz=inference_config['imgsz'],
                half=inference_config['half'],
                device=inference_config['device'],
                verbose=inference_config['verbose']
            )[0]

            detections = sv.Detections.from_ultralytics(result)
            if yolo_roi is not None:
                detections = shift_detections_to_full_frame(detections, inference_offset)
                detections = keep_detections_inside_roi(detections, yolo_roi)

            # Run the Tracker
            tracker.track_detection(detections)
            tracker.update_lap_counts()

            # Extract better crop from helmet detections and give to worker queue
            if ocr_roi is not None:
                helmet_tracks = tracker.get_non_confirmed_helmet_tracks()
                helmet_in_roi = keep_detections_inside_roi(helmet_tracks, ocr_roi)
                # Submit crops into OCR worker
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
            if finish_line is not None:
                start_pt = tuple(int(v) for v in finish_line[0])
                end_pt = tuple(int(v) for v in finish_line[1])
                cv2.line(annotated, start_pt, end_pt, (0, 165, 255), 3)
                cv2.circle(annotated, start_pt, 6, (0, 255, 255), -1)
                cv2.circle(annotated, end_pt, 6, (0, 140, 255), -1)
                label_x = int((start_pt[0] + end_pt[0]) / 2)
                label_y = int((start_pt[1] + end_pt[1]) / 2) - 10
                cv2.putText(
                    annotated,
                    "FINISH",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
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

            lap_rows = tracker.get_active_lap_counts()
            lap_panel_key = (
                finish_line is not None,
                int(tracker.total_laps),
                tuple(
                    (row["track_id"], row["lap_count"], row["predicted"])
                    for row in lap_rows
                ),
            )
            if lap_panel_key != last_lap_panel_key:
                last_lap_panel = render_lap_panel(
                    lap_panel_height,
                    lap_panel_width,
                    lap_rows,
                    finish_line is not None,
                    tracker.total_laps,
                )
                last_lap_panel_key = lap_panel_key

            last_display_frame = annotated
            cv2.imshow(vision_window_name, annotated)
            if last_lap_panel is not None:
                cv2.imshow(lap_window_name, last_lap_panel)

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
                    save_roi(yolo_roi_path, yolo_roi)
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
                        save_roi(ocr_roi_path, ocr_roi)
            if key == ord("f"):
                new_finish_line = select_line(frame, window_name="Finish Line Selector")
                if new_finish_line is not None:
                    finish_line = new_finish_line
                    tracker.set_finish_line(finish_line)
                    save_line(finish_line_path, finish_line)
    finally:
        ocr_worker.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

# Main guard needed for multiprocessing
if __name__ == '__main__':
    freeze_support()
    main()

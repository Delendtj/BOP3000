from multiprocessing.spawn import freeze_support

import cv2
import os
import time
import tkinter as tk
from tkinter import simpledialog

import numpy as np
import supervision as sv

from collections import deque

from functions.ocr.helmet_crop import extract_helmet_box
from functions.detection.roi_inference import (
    keep_detections_inside_roi,
    shift_detections_to_full_frame,
)
from functions.spatial.homography import (
    associate_close_helmets_to_wide_helmet_tracks,
    load_homography,
    map_close_point_to_wide_distorted,
    select_close_frame,
)
from functions.tracking.association import (
    bbox_top_center_xyxy,
    match_close_helmets_to_people,
)
from functions.detection.close_inference import run_close_inference
from functions.spatial.rink_projection import project_bboxes_to_rink_canvas
from functions.visualization.lap_panel import render_lap_panel
from functions.ocr.ocr_worker import OCRWorker
from functions.spatial.roi import (
    load_line,
    load_roi,
    roi_inside_roi,
    save_line,
    save_roi,
    select_line,
    select_roi,
    validate_finish_line,
)
from functions.tracking.assignment import hungarian_assign
from functions.tracking.tracker import Tracker
from functions.visualization.visualization import (
    build_rink_canvas,
    compose_display_canvas,
    compute_window_layout,
    draw_bboxes,
    draw_match_lines,
    draw_rink_match_lines,
    draw_rink_points,
    get_screen_size,
)
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

WIDE_SOURCE = "../videos/wide_cam.mp4"
CLOSE_SOURCE = "../videos/close_cam.mp4"
CONF_THRESHOLD = 0.2
FRAME_SKIP = 1
OCR_VOTE = 3          # collect votes for N frames before deciding
MAX_SYNC_DELTA = 0.05
CLOSE_MATCH_MAX_DIST = 120
CLOSE_HELMET_PERSON_MAX_DIST = 80
CLOSE_HELMET_PERSON_MAX_BELOW_RATIO = 0.08
HELMET_CLASS_ID = 0
PERSON_CLASS_ID = 1
SYNC_MISS_LOG_INTERVAL = 2.0
HOMOGRAPHY_PATH = os.path.join("img", "homography.json")
RINK_WIDE_H_PATH = os.path.join("img", "homography_wide.json")
RINK_CLOSE_H_PATH = os.path.join("img", "homography_close.json")

INFERENCE_CONFIG = {
    'conf': CONF_THRESHOLD,
    'iou': 0.5,
    'max_det': 100,
    'imgsz': 1280,
    'half': True, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

finish_line_path = os.path.join("data", "finish_line.json")
YOLO_ROI_PATH = os.path.join("img", "yolo_roi.json")
OCR_ROI_PATH = os.path.join("img", "ocr_roi.json")
CLOSE_YOLO_ROI_PATH = os.path.join("img", "close_yolo_roi.json")
CLOSE_OCR_ROI_PATH = os.path.join("img", "close_ocr_roi.json")
# IIHF 60m x 30m rink -> half-extents: x in [-15, 15], y in [-30, 30]
RINK_BOUNDS = (-15.0, 15.0, -30.0, 30.0)
RINK_GOAL_LINE_OFFSET = 26.0
RINK_RED_LINES = (0.0, -RINK_GOAL_LINE_OFFSET, RINK_GOAL_LINE_OFFSET)
RINK_MATCH_MAX_DIST = 1.5
multi_cam_window_name = "Multi-cam view"
rink_window_name = "Rink view"
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
    screen_width, screen_height = get_screen_size()
    window_layout = compute_window_layout(screen_width, screen_height)
    display_panel_size = window_layout["display_panel_size"]
    rink_canvas_size = window_layout["rink_canvas_size"]
    lap_panel_width, lap_panel_height = window_layout["lap_panel_size"]

    # Open video
    preview_cap = cv2.VideoCapture(WIDE_SOURCE)
    fps = preview_cap.get(cv2.CAP_PROP_FPS)
    wide_ret, preview_frame = preview_cap.read()
    preview_cap.release()

    preview_frame = downscale_to_1080p(preview_frame)
    if not wide_ret:
        raise RuntimeError("Could not read initial frame for ROI.")

    close_preview_cap = cv2.VideoCapture(CLOSE_SOURCE)
    close_ret, close_preview_frame = close_preview_cap.read()
    close_preview_cap.release()

    close_preview_frame = downscale_to_1080p(close_preview_frame)
    if not close_ret:
        raise RuntimeError("Could not read initial close frame for ROI.")

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

    finish_line = load_line(finish_line_path)
    finish_line_error = validate_finish_line(finish_line, frame_shape=preview_frame.shape, roi=yolo_roi)
    if finish_line_error is not None:
        print(f"Ignoring saved finish line: {finish_line_error}")
        finish_line = None

    cv2.namedWindow(multi_cam_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(multi_cam_window_name, display_panel_size[0] * 2, display_panel_size[1])
    cv2.moveWindow(multi_cam_window_name, *window_layout["multi_cam_pos"])
    cv2.namedWindow(lap_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lap_window_name, lap_panel_width, lap_panel_height)
    cv2.moveWindow(lap_window_name, *window_layout["lap_pos"])
    cv2.namedWindow(rink_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rink_window_name, *rink_canvas_size)
    cv2.moveWindow(rink_window_name, *window_layout["rink_pos"])

    close_yolo_roi = load_roi(CLOSE_YOLO_ROI_PATH)
    if close_yolo_roi is None:
        close_yolo_roi = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
        if close_yolo_roi is not None:
            save_roi(CLOSE_YOLO_ROI_PATH, close_yolo_roi)

    close_ocr_roi = load_roi(CLOSE_OCR_ROI_PATH)
    if close_ocr_roi is not None and close_yolo_roi is not None and not roi_inside_roi(close_ocr_roi, close_yolo_roi):
        print("Loaded close OCR ROI is outside close YOLO ROI. Please redraw.")
        close_ocr_roi = None

    if close_ocr_roi is None and close_yolo_roi is not None:
        selected_close_ocr = select_ocr_roi_inside_yolo(close_preview_frame, close_yolo_roi)
        if selected_close_ocr is not None:
            close_ocr_roi = selected_close_ocr
            save_roi(CLOSE_OCR_ROI_PATH, close_ocr_roi)

    # THE TRACKER
    tracker = Tracker(
        OCR_VOTE,
        CONF_THRESHOLD,
        ocr_roi,
        frame_rate=fps,
        finish_line=finish_line,
        total_laps=total_laps,
    )

    # Initialize threads and processes
    # WIDE Source
    pipeline = AsyncFramePipeline(
        source=WIDE_SOURCE,
        frame_skip=FRAME_SKIP,
        queue_size=3,
        inference_roi=yolo_roi,
    )
    # CLOSE Source
    pipeline_close = AsyncFramePipeline(
        source=CLOSE_SOURCE,
        frame_skip=FRAME_SKIP,
        queue_size=3,
        inference_roi=close_yolo_roi,
    )

    ocr_worker = OCRWorker()
    ocr_worker.start()
    pipeline.start()
    pipeline_close.start()

    close_buffer = deque(maxlen=32)
    latest_close_item = None

    # Load/Check Homography
    try:
        homography = load_homography(HOMOGRAPHY_PATH)
        if homography.source_role != "close" or homography.target_role != "wide":
            raise RuntimeError(
                f"Homography must map close -> wide, got {homography.source_role} -> {homography.target_role}."
            )
        print(
            f"Loaded homography direction: {homography.source_role} -> {homography.target_role}"
        )
    except FileNotFoundError:
        homography = None
        print("Homography missing, cross-camera crops disabled.")

    try:
        wide_rink_h = load_homography(RINK_WIDE_H_PATH)
        if wide_rink_h.source_role != "wide" or wide_rink_h.target_role != "rink":
            raise RuntimeError(
                f"Wide rink homography must map wide -> rink, got {wide_rink_h.source_role} -> {wide_rink_h.target_role}."
            )
        print("Loaded wide->rink homography.")
    except FileNotFoundError:
        wide_rink_h = None
        print("Wide rink homography missing, rink view will omit wide points.")

    try:
        close_rink_h = load_homography(RINK_CLOSE_H_PATH)
        if close_rink_h.source_role != "close" or close_rink_h.target_role != "rink":
            raise RuntimeError(
                f"Close rink homography must map close -> rink, got {close_rink_h.source_role} -> {close_rink_h.target_role}."
            )
        print("Loaded close->rink homography.")
    except FileNotFoundError:
        close_rink_h = None
        print("Close rink homography missing, rink view will omit close points.")

    # Section for initializing benchmark classes here:
    ocr_bench = OCRThroughputStats(log_every_sec=2.0)

    prev_frame_time = None
    fps_ema = 0.0
    frame_count = 0
    paused = False
    last_display_frame = None
    last_lap_panel = None
    last_lap_panel_key = None
    close_sync_misses = 0
    last_close_sync_log = 0.0

    print("Controls: Esc=quit, Space=pause/resume, r/o=wide ROIs, c/v=close ROIs")

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
                cv2.imshow(multi_cam_window_name, paused_frame)
                if last_lap_panel is not None:
                    cv2.imshow(lap_window_name, last_lap_panel)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    paused = False
                continue

            # Read frames from close camera (From Asyncpipeline queue)
            while True:
                close_item = pipeline_close.read(timeout=0.01)

                if close_item is None:
                    break
                latest_close_item = close_item
                close_buffer.append((close_item.ts, close_item))

            item = pipeline.read(timeout=0.5)
            if item is None:
                if pipeline.stop_event.is_set():
                    break
                continue

            frame = item.frame
            inference_frame = item.inference_frame if item.inference_frame is not None else frame
            inference_offset = item.inference_offset

            frame_count += 1

            # Maybe we only need detection for class 1 (people) here???
            result = model(
                inference_frame,
                conf=INFERENCE_CONFIG['conf'],
                iou=INFERENCE_CONFIG['iou'],
                max_det=INFERENCE_CONFIG['max_det'],
                imgsz=INFERENCE_CONFIG['imgsz'],
                half=INFERENCE_CONFIG['half'],
                device=INFERENCE_CONFIG['device'],
                verbose=INFERENCE_CONFIG['verbose'],
                classes=[PERSON_CLASS_ID]
            )[0]

            detections = sv.Detections.from_ultralytics(result)
            if yolo_roi is not None:
                detections = shift_detections_to_full_frame(detections, inference_offset)
                detections = keep_detections_inside_roi(detections, yolo_roi)

            # Run the Tracker
            tracker.track_detection(detections)
            tracker.update_lap_counts()

            # Extract better crop from helmet detections and give to worker queue
            close_frame = None
            close_inference_frame = None
            synced_close_item = None
            if homography is not None:
                synced_close_item = select_close_frame(close_buffer, item.ts, MAX_SYNC_DELTA)
                if synced_close_item is not None:
                    close_frame = synced_close_item.frame
                    close_inference_frame = (
                        synced_close_item.inference_frame
                        if synced_close_item.inference_frame is not None
                        else close_frame
                    )

            close_vis = None
            latest_close_frame = latest_close_item.frame if latest_close_item is not None else None

            wide_overlay_points = []
            close_people = None
            if ocr_roi is not None:
                helmet_tracks = tracker.get_non_confirmed_helmet_tracks()
                helmet_in_roi = keep_detections_inside_roi(helmet_tracks, ocr_roi)
                helmet_crops = []

                if homography is not None:
                    if synced_close_item is None:
                        if len(helmet_in_roi) > 0:
                            close_sync_misses += 1
                            now = time.perf_counter()
                            if now - last_close_sync_log >= SYNC_MISS_LOG_INTERVAL:
                                print(
                                    f"Skipping multi-cam OCR: no close frame within {MAX_SYNC_DELTA:.3f}s "
                                    f"(misses={close_sync_misses})"
                                )
                                last_close_sync_log = now

                # This whole if contains everything with homography association for helmet
                if synced_close_item is not None and homography is not None:
                    close_out = run_close_inference(
                        model=model,
                        frame_item=synced_close_item,
                        inference_config=INFERENCE_CONFIG,
                        helmet_class_id=HELMET_CLASS_ID,
                        person_class_id=PERSON_CLASS_ID,
                        yolo_roi=close_yolo_roi,
                        ocr_roi=close_ocr_roi,
                    )
                    close_helmets = close_out.helmets
                    close_people = close_out.people

                    close_vis = close_frame.copy()

                    draw_bboxes(close_vis, close_helmets, (0, 0, 255), thickness=2)
                    draw_bboxes(close_vis, close_people, (255, 0, 0), thickness=2)

                    helmet_person_matches = match_close_helmets_to_people(
                        close_helmets,
                        close_people,
                        max_dist=CLOSE_HELMET_PERSON_MAX_DIST,
                        max_person_top_below_ratio=CLOSE_HELMET_PERSON_MAX_BELOW_RATIO,
                    )
                    draw_match_lines(
                        close_vis,
                        close_helmets,
                        close_people,
                        helmet_person_matches,
                        bbox_top_center_xyxy,
                        bbox_top_center_xyxy,
                        (0, 255, 0),
                        thickness=2,
                    )

                    # Visualize homography on wide frame (project close -> wide).
                    if len(close_helmets) > 0:
                        for bbox in close_helmets.xyxy:
                            cx = float((bbox[0] + bbox[2]) / 2.0)
                            cy = float((bbox[1] + bbox[3]) / 2.0)

                            projected = map_close_point_to_wide_distorted(
                                homography,
                                cx,
                                cy,
                                wide_img_shape=frame.shape,
                            )
                            if projected is None:
                                continue
                            wide_overlay_points.append((int(projected[0]), int(projected[1])))

                        helmet_crops = associate_close_helmets_to_wide_helmet_tracks(
                            helmet_in_roi,
                            close_helmets,
                            close_frame,
                            homography,
                            CLOSE_MATCH_MAX_DIST,
                        )
                else:
                    # OLD: Without homography           [FIND A WAY TO GET RID OF THIS]
                    helmet_crops = extract_helmet_box(helmet_in_roi, frame)

                # Give associated helmet crops to the ocr_worker
                # This means if associate_close_helmet_crops() fails, nothing goes to the OCR
                for h in helmet_crops:
                    ocr_worker.submit(h)

            # Evaluate if the detection is good enough to be set for tracks
            helmets_res = ocr_worker.drain_results()
            tracker.check_for_ocr(helmets_res) # Includes all voting logic.

            # Annotate frames
            annotated = tracker.annotate(frame)

            # Draw projected points for visualization
            if wide_overlay_points:
                for px, py in wide_overlay_points:
                    # print(0 <= px < annotated.shape[1] and 0 <= py < annotated.shape[0])
                    if 0 <= px < annotated.shape[1] and 0 <= py < annotated.shape[0]:
                        cv2.circle(annotated, (px, py), 10, (0, 0, 255), -1)

            # Rink-space view (project wide + close points into rink coords)
            if wide_rink_h is not None or close_rink_h is not None:
                rink_canvas = build_rink_canvas(
                    RINK_BOUNDS,
                    rink_canvas_size,
                    draw_center_line=False,
                    draw_center_circle=True,
                    center_circle_radius=4.5,
                    horizontal=True,
                    red_lines=RINK_RED_LINES,
                )

                wide_rink_points = project_bboxes_to_rink_canvas(
                    tracker.people_tracks.xyxy if len(tracker.people_tracks) > 0 else None,
                    wide_rink_h,
                    RINK_BOUNDS,
                    rink_canvas_size,
                    horizontal=True,
                    undistort=True,
                    img_shape=frame.shape,
                )
                close_rink_points = project_bboxes_to_rink_canvas(
                    close_people.xyxy if close_people is not None and len(close_people) > 0 else None,
                    close_rink_h,
                    RINK_BOUNDS,
                    rink_canvas_size,
                    horizontal=True,
                )

                draw_rink_points(
                    rink_canvas,
                    [canvas_xy for _, canvas_xy in wide_rink_points],
                    color=(0, 0, 255),
                )
                draw_rink_points(
                    rink_canvas,
                    [canvas_xy for _, canvas_xy in close_rink_points],
                    color=(255, 0, 0),
                )

                # Draw connections between projected points from both cameras.
                if wide_rink_points and close_rink_points:
                    wide_xy = [p[0] for p in wide_rink_points]
                    close_xy = [p[0] for p in close_rink_points]
                    matches = hungarian_assign(wide_xy, close_xy, max_dist=RINK_MATCH_MAX_DIST)
                    draw_rink_match_lines(
                        rink_canvas,
                        matches,
                        [canvas_xy for _, canvas_xy in wide_rink_points],
                        [canvas_xy for _, canvas_xy in close_rink_points],
                        color=(0, 200, 0),
                    )

                cv2.imshow(rink_window_name, rink_canvas)

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
                frame_height, frame_width = annotated.shape[:2]
                arrow_y = max(30, label_y - 22)
                arrow_start = (max(15, label_x - 50), arrow_y)
                arrow_end = (min(frame_width - 15, label_x + 50), arrow_y)
                cv2.arrowedLine(annotated, arrow_start, arrow_end, (0, 165, 255), 3, tipLength=0.2)
                cv2.putText(
                    annotated,
                    "FINISH L->R",
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
            # Lap counting GUI logic
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

            if last_lap_panel is not None:
                cv2.imshow(lap_window_name, last_lap_panel)

            if close_vis is None and latest_close_frame is not None:
                close_vis = latest_close_frame.copy()
            if close_vis is not None and close_yolo_roi is not None:
                cv2.polylines(
                    close_vis,
                    [np.array(close_yolo_roi, dtype=np.int32)],
                    True,
                    (0, 255, 255),
                    2,
                )
            if close_vis is not None and close_ocr_roi is not None:
                cv2.polylines(
                    close_vis,
                    [np.array(close_ocr_roi, dtype=np.int32)],
                    True,
                    (0, 200, 0),
                    2,
                )

            close_subtitle = None
            if latest_close_frame is None:
                close_subtitle = "No close frame"
            elif homography is not None and synced_close_item is None:
                close_subtitle = "Unsynced preview"

            # Builds the two windows (rink and two cam feeds)
            # Makes them the correct size based on screen size.
            display_frame = compose_display_canvas(
                annotated,
                close_vis,
                display_panel_size,
                wide_subtitle=fps_text,
                close_subtitle=close_subtitle,
            )
            last_display_frame = display_frame
            cv2.imshow(multi_cam_window_name, display_frame)

            # [THis whole block should be placed somewhere else]
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
                    finish_line_error = validate_finish_line(finish_line, frame_shape=frame.shape, roi=yolo_roi)
                    if finish_line_error is not None:
                        print(f"Current finish line was cleared: {finish_line_error}")
                        finish_line = None
                        tracker.set_finish_line(None)
            if key == ord("o"):
                if yolo_roi is None:
                    print("Define YOLO ROI first (press 'r').")
                else:
                    new_ocr_roi = select_ocr_roi_inside_yolo(frame, yolo_roi)
                    if new_ocr_roi is not None:
                        ocr_roi = new_ocr_roi
                        tracker.set_roi(ocr_roi)
                        save_roi(OCR_ROI_PATH, ocr_roi)
            if key == ord("f"):
                new_finish_line = select_line(frame, window_name="Finish Line Selector")
                if new_finish_line is not None:
                    finish_line_error = validate_finish_line(
                        new_finish_line,
                        frame_shape=frame.shape,
                        roi=yolo_roi,
                    )
                    if finish_line_error is not None:
                        print(f"Finish line not saved: {finish_line_error}")
                    else:
                        line_changed = new_finish_line != finish_line
                        finish_line = new_finish_line
                        tracker.set_finish_line(finish_line)
                        if line_changed:
                            print("Finish line updated. Lap counts reset.")
                        save_line(finish_line_path, finish_line)
                        save_roi(OCR_ROI_PATH, ocr_roi)
            if key == ord("c"):
                if close_yolo_roi is None:
                    close_yolo_roi = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
                    if close_yolo_roi is not None:
                        save_roi(CLOSE_YOLO_ROI_PATH, close_yolo_roi)
                        pipeline_close.set_inference_roi(close_yolo_roi)
                else:
                    new_close_yolo = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
                    if new_close_yolo is not None:
                        close_yolo_roi = new_close_yolo
                        save_roi(CLOSE_YOLO_ROI_PATH, close_yolo_roi)
                        pipeline_close.set_inference_roi(close_yolo_roi)
            if key == ord("v"):
                if close_yolo_roi is None:
                    print("Define close YOLO ROI first (press 'c').")
                else:
                    new_close_ocr = select_ocr_roi_inside_yolo(close_preview_frame, close_yolo_roi)
                    if new_close_ocr is not None:
                        close_ocr_roi = new_close_ocr
                        save_roi(CLOSE_OCR_ROI_PATH, close_ocr_roi)
    finally:
        ocr_worker.stop()
        pipeline.stop()
        pipeline_close.stop()
        cv2.destroyAllWindows()

# Main guard needed for multiprocessing
if __name__ == '__main__':
    freeze_support()
    main()

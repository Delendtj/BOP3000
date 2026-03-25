from multiprocessing.spawn import freeze_support

import cv2
import os
import time

import numpy as np
import supervision as sv

from collections import deque

from functions.tracking.close_association import bbox_top_center_xyxy, match_close_helmets_to_people
from functions.tracking.cross_camera_transfer import (
    build_close_to_wide_mapping,
    build_helmet_crops_for_wide_ids,
)
from functions.detection.roi_inference import keep_detections_inside_roi, shift_detections_to_full_frame
from functions.detection.close_inference import run_close_inference
from functions.ocr.ocr_worker import OCRWorker
from functions.spatial.homography import select_close_frame, load_close_homography, load_wide_homography
from functions.visualization.lap_panel import render_lap_panel, prompt_total_laps
from functions.spatial.roi.io import save_line, save_roi
from functions.spatial.roi.selection import select_line, select_roi
from functions.spatial.roi.setup import (
    load_or_select_close_rois,
    load_or_select_wide_rois,
    select_ocr_roi_inside_yolo,
)
from functions.spatial.roi.validation import roi_inside_roi, validate_finish_line
from functions.tracking.tracker import Tracker
from functions.visualization.rink_view import build_rink_view
from functions.visualization.visualization import (
    compose_display_canvas,
    compute_window_layout,
    draw_bboxes,
    draw_match_lines,
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
CLOSE_HELMET_PERSON_MAX_DIST = 80
CLOSE_HELMET_PERSON_MAX_BELOW_RATIO = 0.08
HELMET_CLASS_ID = 0
PERSON_CLASS_ID = 1
SYNC_MISS_LOG_INTERVAL = 2.0

HOMOGRAPHY_PATH = os.path.join("data", "homography.json")
RINK_WIDE_H_PATH = os.path.join("data", "homography_wide.json")
RINK_CLOSE_H_PATH = os.path.join("data", "homography_close.json")
FINISH_LINE_PATH = os.path.join("data", "finish_line.json")
YOLO_ROI_PATH = os.path.join("data", "yolo_roi.json")
OCR_ROI_PATH = os.path.join("data", "ocr_roi.json")
CLOSE_YOLO_ROI_PATH = os.path.join("data", "close_yolo_roi.json")
CLOSE_OCR_ROI_PATH = os.path.join("data", "close_ocr_roi.json")

# IIHF 60m x 30m rink -> half-extents: x in [-15, 15], y in [-30, 30]
RINK_BOUNDS = (-15.0, 15.0, -30.0, 30.0)
RINK_GOAL_LINE_OFFSET = 26.0
RINK_RED_LINES = (0.0, -RINK_GOAL_LINE_OFFSET, RINK_GOAL_LINE_OFFSET)
RINK_MATCH_MAX_DIST = 1.5
multi_cam_window_name = "Multi-cam view"
rink_window_name = "Rink view"
lap_window_name = "Lap Count"

INFERENCE_CONFIG = {
    'conf': CONF_THRESHOLD,
    'iou': 0.5,
    'max_det': 100,
    'imgsz': 1280,
    'half': True, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

def main():
    detector = HardwareDetector(config)
    model = detector.initialize_model()

    # Show prompt for user input (laps)
    total_laps = prompt_total_laps()

    # Gather all screen specifications needed for layout
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
    # One for checking main model detections (YOLO)
    # One for running the helmet number (OCR)
    yolo_roi, ocr_roi, finish_line = load_or_select_wide_rois(
        preview_frame=preview_frame,
        yolo_roi_path=YOLO_ROI_PATH,
        ocr_roi_path=OCR_ROI_PATH,
        finish_line_path=FINISH_LINE_PATH,
    )

    # Create the windows and their layout
    # This is based on the screem specifications gathered earlier.
    cv2.namedWindow(multi_cam_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(multi_cam_window_name, display_panel_size[0] * 2, display_panel_size[1])
    cv2.moveWindow(multi_cam_window_name, *window_layout["multi_cam_pos"])
    cv2.namedWindow(lap_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lap_window_name, lap_panel_width, lap_panel_height)
    cv2.moveWindow(lap_window_name, *window_layout["lap_pos"])
    cv2.namedWindow(rink_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rink_window_name, *rink_canvas_size)
    cv2.moveWindow(rink_window_name, *window_layout["rink_pos"])

    # Loads the saved close ROIs if it find any from CLOSE_YOLO_ROI_PATH and CLOSE_OCR_ROI_PATH
    # Else it prompts the user to draw ROIs for the missing ROIs.
    close_yolo_roi, close_ocr_roi = load_or_select_close_rois(
        close_preview_frame=close_preview_frame,
        close_yolo_roi_path=CLOSE_YOLO_ROI_PATH,
        close_ocr_roi_path=CLOSE_OCR_ROI_PATH,
    )

    # Create the Tracker
    tracker = Tracker(
        OCR_VOTE,
        CONF_THRESHOLD,
        ocr_roi,
        frame_rate=fps,
        finish_line=finish_line,
        total_laps=total_laps,
    )

    # [SECTION] Initialize processes and threads.

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
    # Create worker process for OCR
    ocr_worker = OCRWorker()
    # Start both the async-pipelines (threads) and worker process
    ocr_worker.start()
    pipeline.start()
    pipeline_close.start()

    # Close buffer that contains elements of (timestamp, frame)
    # We do this because frames from both sources arrive at different time.
    # So when we process wide frames, we look at the latest_close_item
    # This is the frame that has the closest timestamp compared to the wide frame.
    close_buffer = deque(maxlen=32)
    latest_close_item = None


    # Loading rink homography for both wide and close feed.
    # These are the homography that contain the numbers that allow
    # the program to project points the different cameras into the virtual rink space.
    wide_rink_h = load_wide_homography(RINK_WIDE_H_PATH)
    close_rink_h = load_close_homography(RINK_CLOSE_H_PATH)

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
            synced_close_item = None
            if close_rink_h is not None:
                synced_close_item = select_close_frame(close_buffer, item.ts, MAX_SYNC_DELTA)
                if synced_close_item is not None:
                    close_frame = synced_close_item.frame

            close_vis = None
            latest_close_frame = latest_close_item.frame if latest_close_item is not None else None

            close_people = None
            helmet_crops = []
            if ocr_roi is not None:
                helmet_tracks = tracker.get_non_confirmed_helmet_tracks()
                helmet_in_roi = keep_detections_inside_roi(helmet_tracks, ocr_roi)

                if close_rink_h is not None:
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

                # Contains all about association between cameras
                if synced_close_item is not None and close_rink_h is not None:
                    close_out = run_close_inference(
                        model=model,
                        frame_item=synced_close_item,
                        inference_config=INFERENCE_CONFIG,
                        helmet_class_id=HELMET_CLASS_ID,
                        person_class_id=PERSON_CLASS_ID,
                        yolo_roi=close_yolo_roi,
                        ocr_roi=close_ocr_roi,
                    )
                    # Separate helmets and people from the result
                    close_helmets = close_out.helmets
                    close_people = close_out.people

                    close_vis = close_frame.copy()

                    # Draw boxes on each
                    draw_bboxes(close_vis, close_helmets, (0, 0, 255), thickness=2)
                    draw_bboxes(close_vis, close_people, (255, 0, 0), thickness=2)

                    # Runs the logic for pairing helmets and people tracks (close cam)
                    # This simply returns (helmet_idx, person_idx, dist) for each match it finds.
                    # Index for close_helmets, close_people.
                    helmet_person_matches = match_close_helmets_to_people(
                        close_helmets,
                        close_people,
                        max_dist=CLOSE_HELMET_PERSON_MAX_DIST,
                        max_person_top_below_ratio=CLOSE_HELMET_PERSON_MAX_BELOW_RATIO,
                    )

                    # Draws the line between top center of the matching helmet and person bboxes.
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

                    close_to_wide_tid = build_close_to_wide_mapping(
                        tracker.people_tracks,
                        close_people,
                        wide_rink_h,
                        close_rink_h,
                        img_shape=frame.shape,
                        max_dist=RINK_MATCH_MAX_DIST,
                    )
                    helmet_crops = build_helmet_crops_for_wide_ids(
                        close_helmets,
                        close_frame,
                        helmet_person_matches,
                        close_to_wide_tid,
                    )

                # Give associated helmet crops to the ocr_worker
                for h in helmet_crops:
                    ocr_worker.submit(h)

            # Evaluate if the detection is good enough to be set for tracks
            helmets_res = ocr_worker.drain_results()
            tracker.check_for_ocr(helmets_res) # Includes all voting logic.

            # Annotate frames
            annotated = tracker.annotate(frame)

            # Creates the Rink-Space window
            # This is used as a canvas for points representing people tracks on both cams
            if wide_rink_h is not None or close_rink_h is not None:
                rink_canvas = build_rink_view(
                    wide_tracks=tracker.people_tracks if len(tracker.people_tracks) > 0 else None,
                    close_people=close_people,
                    wide_rink_h=wide_rink_h,
                    close_rink_h=close_rink_h,
                    bounds=RINK_BOUNDS,
                    canvas_size=rink_canvas_size,
                    img_shape=frame.shape,
                    max_dist=RINK_MATCH_MAX_DIST,
                    horizontal=True,
                    draw_center_line=False,
                    draw_center_circle=True,
                    center_circle_radius=4.5,
                    red_lines=RINK_RED_LINES,
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

            # Checks if players cross finish line
            # (Should move this code)
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
            # (Should be moved)
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
            elif close_rink_h is not None and synced_close_item is None:
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
                        save_line(FINISH_LINE_PATH, finish_line)
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

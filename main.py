from multiprocessing.spawn import freeze_support

import cv2
import os
import time

import supervision as sv

from collections import deque

from functions.system.config import load_config
from functions.tracking.close_association import bbox_top_center_xyxy, match_close_helmets_to_people
from functions.tracking.cross_camera_transfer import (
    build_close_to_wide_mapping,
    build_helmet_crops_for_wide_ids,
)
from functions.detection.roi_inference import keep_detections_inside_roi, shift_detections_to_full_frame
from functions.detection.close_inference import run_close_inference
from functions.spatial.rink_projection import project_bboxes_to_rink_canvas
from functions.visualization.helmet_gui import prompt_race_setup
from functions.visualization.lap_panel import render_lap_panel
from functions.ocr.ocr_worker import OCRWorker
from functions.spatial.homography import select_close_frame, load_close_homography, load_wide_homography
from functions.system.controls import KeyboardControlState, handle_keypress
from functions.visualization.lap_panel import LapPanelState, prompt_total_laps, update_lap_panel_state
from functions.spatial.roi.setup import (
    load_or_select_close_rois,
    load_or_select_wide_rois,
)
from functions.tracking.tracker import Tracker
from functions.visualization.rink_view import build_rink_view
from functions.visualization.visualization import (
    compose_dashboard_canvas,
    compose_display_canvas,
    compute_window_layout,
    draw_bboxes,
    draw_finish_line_overlay,
    draw_match_lines,
    get_screen_size,
    draw_roi_lines,
    setup_window,
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

WIDE_SOURCE = "DJI_20260301122027_0001_D.MP4"
CLOSE_SOURCE = "Canon_2026-03-01_12-20-29.mp4"
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
dashboard_window_name = "Race Dashboard"

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
    startup_setup = prompt_race_setup()
    if startup_setup is None:
        print("Startup cancelled before race setup was completed.")
        return

    helmet_numbers = startup_setup.helmet_numbers
    total_laps = startup_setup.total_laps

    detector = HardwareDetector(config)
    model = detector.initialize_model()
    print(f"Loaded {len(helmet_numbers)} helmet numbers from startup GUI.")

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

    # Create the dashboard window based on the user's screen size.
    setup_window(window_layout, window_name=dashboard_window_name)

    # Loads the saved close ROIs if it finds any from CLOSE_YOLO_ROI_PATH and CLOSE_OCR_ROI_PATH
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

    # Handles all keyboard state changes
    # for the different modules (yolo_roi, ocr_roi etc...)
    keyboard_state = KeyboardControlState(
        yolo_roi=yolo_roi,
        ocr_roi=ocr_roi,
        finish_line=finish_line,
        close_yolo_roi=close_yolo_roi,
        close_ocr_roi=close_ocr_roi,
        paused=False,
    )

    prev_frame_time = None  # Timestamp of previous frame (Used for instant FPS)
    fps_ema = 0.0           # Exponential moving average (used for seeing a more stable FPS on display)
    frame_count = 0         # Current frame count (mostly for debug and logging)

    last_dashboard_frame = None
    close_sync_misses = 0
    last_close_sync_log = 0.0

    lap_panel_state = LapPanelState()

    print("Controls: Esc=quit, Space=pause/resume, r/o=wide ROIs, c/v=close ROIs")

    # Main Loop
    try:
        while True:
            # Pause logic (With Space Bar)
            if keyboard_state.paused and last_dashboard_frame is not None:
                paused_frame = last_dashboard_frame.copy()
                cv2.putText(
                    paused_frame,
                    "PAUSED (Space to resume)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(dashboard_window_name, paused_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    keyboard_state.paused = False
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

                    # Maps players from close to wide camera through the virtual rink
                    close_to_wide_tid = build_close_to_wide_mapping(
                        tracker.people_tracks,
                        close_people,
                        wide_rink_h,
                        close_rink_h,
                        img_shape=frame.shape,
                        max_dist=RINK_MATCH_MAX_DIST,
                    )
                    # Then we extract only the helmets that has been connected to people tracks across screens
                    helmet_crops = build_helmet_crops_for_wide_ids(
                        close_helmets,
                        close_frame,
                        helmet_person_matches,
                        close_to_wide_tid,
                    )

                # Give associated helmet crops to the ocr_worker
                # The ocr_worker will then perform ocr on the crops.
                for h in helmet_crops:
                    ocr_worker.submit(h)

            # Evaluate if the detection is good enough to be set for tracks
            helmets_res = ocr_worker.drain_results()
            tracker.check_for_ocr(helmets_res) # Includes all voting logic.

            # Annotate frames
            annotated_frame = tracker.annotate(frame)

            rink_canvas = None
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

            # Draws the wide camera ROIs onto the annotated frame.
            draw_roi_lines(annotated_frame, yolo_roi, ocr_roi)

            # Draw finish line
            if finish_line is not None:
                draw_finish_line_overlay(annotated_frame, finish_line)

            # Calculate FPS
            now = time.perf_counter()
            prev_frame_time = now
            if prev_frame_time is not None:
                elapsed = now - prev_frame_time
                if elapsed > 0:
                    # Instantaneous fps
                    fps_inst = 1.0 / elapsed
                    # Exponential Moving Average FPS
                    fps_ema = fps_inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * fps_inst)

            if ocr_bench.should_log(now):
                stats = ocr_worker.get_stats()
                print(ocr_bench.format_line(stats))

            fps_text = f"FPS: {fps_ema:.1f}" if fps_ema > 0 else "FPS: --"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # Update the lap_panel
            lap_panel_state = update_lap_panel_state(
                lap_panel_state,
                tracker=tracker,
                finish_line=finish_line,
                height=lap_panel_height,
                width=lap_panel_width,
            )

            # Fallback for when we don't have a synced close frame.
            if close_vis is None and latest_close_frame is not None:
                close_vis = latest_close_frame.copy()
            # Draws the ROI lines onto the close camera frame.
            draw_roi_lines(close_vis, close_yolo_roi, close_ocr_roi)

            close_subtitle = None
            if latest_close_frame is None:
                close_subtitle = "No close frame"
            elif close_rink_h is not None and synced_close_item is None:
                close_subtitle = "Unsynced preview"

            # Builds the two windows (rink and two cam feeds)
            # Makes them the correct size based on screen size.
            display_frame = compose_display_canvas(
                annotated_frame,
                close_vis,
                display_panel_size,
                wide_subtitle=fps_text,
                close_subtitle=close_subtitle,
            )
            dashboard_frame = compose_dashboard_canvas(
                lap_panel_state.panel,
                display_frame,
                rink_canvas,
                window_layout,
            )
            last_dashboard_frame = dashboard_frame
            cv2.imshow(dashboard_window_name, dashboard_frame)

            # Get the current keypress
            key = cv2.waitKey(1) & 0xFF
            # handle_keypress() then handles the keypress
            # and returns results based on what key is pressed.
            key_result = handle_keypress(
                key=key,
                state=keyboard_state,
                tracker=tracker,
                pipeline=pipeline,
                pipeline_close=pipeline_close,
                frame=frame,
                close_preview_frame=close_preview_frame,
                yolo_roi_path=YOLO_ROI_PATH,
                ocr_roi_path=OCR_ROI_PATH,
                close_yolo_roi_path=CLOSE_YOLO_ROI_PATH,
                close_ocr_roi_path=CLOSE_OCR_ROI_PATH,
                finish_line_path=FINISH_LINE_PATH,
            )

            # Give new state to the keyboard_state controller.
            keyboard_state = key_result.state
            yolo_roi = keyboard_state.yolo_roi
            ocr_roi = keyboard_state.ocr_roi
            finish_line = keyboard_state.finish_line
            close_yolo_roi = keyboard_state.close_yolo_roi
            close_ocr_roi = keyboard_state.close_ocr_roi
            if key_result.should_quit:
                break
            if keyboard_state.paused:
                continue
    finally:
        ocr_worker.stop()
        pipeline.stop()
        pipeline_close.stop()
        cv2.destroyAllWindows()

# Main guard needed for multiprocessing
if __name__ == '__main__':
    freeze_support()
    main()

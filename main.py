# Core
from multiprocessing.spawn import freeze_support
import logging
import os
import time

# Dependencies
import cv2
import supervision as sv

# Profiler (toggle ENABLE_PROFILING below)
from utilities.profiler import Profiler
ENABLE_PROFILING = False

# From functions folder
from functions.detection.close_inference import run_close_inference
from functions.detection.roi_inference import keep_detections_inside_roi, shift_detections_to_full_frame
from functions.spatial.homography import select_close_frame
from functions.system.config import load_config
from functions.system.init import initialize, shutdown
from functions.system.controls import handle_keypress, pause
from functions.tracking.close_association import bbox_top_center_xyxy, match_close_helmets_to_people
from functions.tracking.cross_camera_transfer import (
    build_close_to_wide_mapping,
    build_helmet_crops_for_wide_ids,
)
from functions.visualization.lap_panel import update_lap_panel_state
from functions.visualization.rink_view import build_rink_view
from functions.visualization.visualization import (
    compose_dashboard_canvas,
    compose_display_canvas,
    draw_bboxes,
    draw_finish_line_overlay,
    draw_match_lines,
    draw_roi_lines,
)

# Configure logging so logger.info() actually prints
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

dashboard_window_name = "Race Dashboard"
# Keep TensorRT TF32 behavior stable between engine build and execution contexts.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


def _section(profiler, name):
    """Helper: returns profiler.section(name) if profiler exists, else nullcontext."""
    from contextlib import nullcontext
    return profiler.section(name) if profiler else nullcontext()


def main(config):
    # Frequently used config variables
    runtime_config = config["Runtime"]
    inference_config = config["Inference"]
    rink_config = config["Rink"]
    conf_threshold = runtime_config["CONF_THRESHOLD"]
    frame_skip = runtime_config["FRAME_SKIP"]
    close_helmet_person_max_dist = runtime_config["CLOSE_HELMET_PERSON_MAX_DIST"]
    close_helmet_person_max_below_ratio = runtime_config["CLOSE_HELMET_PERSON_MAX_BELOW_RATIO"]

    # For syncing of frame across both cameras
    max_sync_delta = config["FrameSync"]["MAX_SYNC_DELTA"]
    sync_miss_log_interval = config["FrameSync"]["SYNC_MISS_LOG_INTERVAL"]

    # Rink
    rink_bounds = rink_config["RINK_BOUNDS"]
    rink_red_lines = rink_config["RINK_RED_LINES"]
    rink_match_max_dist = rink_config["RINK_MATCH_MAX_DIST"]

    context = initialize(config, dashboard_window_name=dashboard_window_name)
    if context is None:
        return

    prev_frame_time = None      # Timestamp of previous frame (used for instant FPS)
    fps_ema = 0.0               # Exponential moving average (used for seeing a more stable FPS on display)
    frame_count = 0             # Current frame count (mostly for debug and logging)

    last_dashboard_frame = None # Frame that is displayed/saved when paused.

    close_sync_misses = 0       # Amount of times close_buffer doesn't have frame/item for the OCR association in time.
    last_close_sync_log = 0.0   # Timestamp for above logging

    logger = logging.getLogger(__name__)
    logger.info("Controls: Esc=quit, Space=pause/resume, r/o=wide ROIs, c/v=close ROIs")

    # Profiler setup
    profiler = Profiler(log_every_sec=2.0) if ENABLE_PROFILING else None
    context.profiler = profiler

    # Main Loop
    try:
        while True:
            # Pause logic (With Space Bar)
            if context.keyboard_state.paused and last_dashboard_frame is not None:
                pause_result = pause(last_dashboard_frame, dashboard_window_name)
                if pause_result.should_quit:
                    break
                if pause_result.should_resume:
                    context.keyboard_state.paused = False
                continue

            # Read frames from close camera (From Asyncpipeline queue).
            # Use drain_available() (non-blocking) so we only process frames
            # already in the queue and do not stall on the fast 60 FPS feed.
            with _section(profiler, "read_close_queue"):
                for close_item in context.pipeline_close.drain_available():
                    context.latest_close_item = close_item
                    context.close_buffer.append((close_item.ts, close_item))

            # Read wide frame from pipeline queue
            with _section(profiler, "read_wide_queue"):
                item = context.pipeline.read(timeout=0.5)
            if item is None:
                if context.pipeline.stop_event.is_set():
                    break
                continue

            frame = item.frame
            inference_frame = item.inference_frame if item.inference_frame is not None else frame
            inference_offset = item.inference_offset

            frame_count += 1

            # Wide camera YOLO inference + ROI filtering
            with _section(profiler, "yolo_wide"):
                # Maybe we only need detection for class 1 (people) here???
                result = context.model(
                    inference_frame,
                    conf=inference_config['conf'],
                    iou=inference_config['iou'],
                    max_det=inference_config['max_det'],
                    imgsz=inference_config['imgsz'],
                    half=inference_config['half'],
                    device=inference_config['device'],
                    verbose=inference_config['verbose'],
                    classes=[config["ModelClass"]["PERSON_CLASS_ID"]]
                )[0]

                detections = sv.Detections.from_ultralytics(result)
                if context.yolo_roi is not None:
                    detections = shift_detections_to_full_frame(detections, inference_offset)
                    detections = keep_detections_inside_roi(detections, context.yolo_roi)

            # Run the Tracker
            with _section(profiler, "tracker_update"):
                context.tracker.track_detection(detections)
                context.tracker.update_lap_counts()

            # Extract better crop from helmet detections and give to worker queue
            close_frame = None
            synced_close_item = None
            if context.close_rink_h is not None:
                # Gives the best match out of close_buffer based on timestamp and max_sync_delta
                with _section(profiler, "close_sync"):
                    synced_close_item = select_close_frame(context.close_buffer, item.ts, max_sync_delta)
                if synced_close_item is not None:
                    close_frame = synced_close_item.frame

            close_vis = None
            latest_close_frame = context.latest_close_item.frame if context.latest_close_item is not None else None

            close_people = None
            helmet_crops = []
            if context.ocr_roi is None:
                logger.debug("OCR ROI not configured — skipping helmet OCR")
            if context.ocr_roi is not None:
                with _section(profiler, "close_pipeline"):
                    helmet_tracks = context.tracker.get_non_confirmed_people_tracks()
                    helmet_in_roi = keep_detections_inside_roi(helmet_tracks, context.ocr_roi)

                    # When the close_buffer doesn't have any valid items
                    # We log that we are skipping OCR. (to stats and terminal)
                    if context.close_rink_h is not None:
                        if synced_close_item is None:
                            if len(helmet_in_roi) > 0:
                                close_sync_misses += 1
                                now = time.perf_counter()
                                if now - last_close_sync_log >= sync_miss_log_interval:
                                    logger.warning(
                                        "Skipping multi-cam OCR: no close frame within %.3fs (misses=%d)",
                                        max_sync_delta, close_sync_misses,
                                    )
                                    last_close_sync_log = now

                    # Contains all about association between cameras
                    if synced_close_item is not None and context.close_rink_h is not None:
                        close_out = run_close_inference(
                            model=context.model,
                            frame_item=synced_close_item,
                            inference_config=inference_config,
                            helmet_class_id=config["ModelClass"]["HELMET_CLASS_ID"],
                            person_class_id=config["ModelClass"]["PERSON_CLASS_ID"],
                            yolo_roi=context.close_yolo_roi,
                            ocr_roi=context.close_ocr_roi,
                        )
                        # Separate helmets and people from the result
                        close_helmets = close_out.helmets
                        close_people = close_out.people

                        logger.debug(
                            "close_inference: helmets=%d, people=%d",
                            len(close_helmets) if close_helmets is not None else 0,
                            len(close_people) if close_people is not None else 0,
                        )

                        close_vis = close_frame.copy()

                        # Run helmet tracker to assign persistent track IDs
                        # (needed for OCR voting / cross-frame consistency)
                        close_helmets = context.tracker.track_helmet_detections(close_helmets)

                        # Draw boxes on each
                        draw_bboxes(close_vis, close_helmets, (0, 0, 255), thickness=2)
                        draw_bboxes(close_vis, close_people, (255, 0, 0), thickness=2)

                        # Runs the logic for pairing helmets and people tracks (close cam)
                        # This simply returns (helmet_idx, person_idx, dist) for each match it finds.
                        # Index for close_helmets, close_people.
                        helmet_person_matches = match_close_helmets_to_people(
                            close_helmets,
                            close_people,
                            max_dist=close_helmet_person_max_dist,
                            max_person_top_below_ratio=close_helmet_person_max_below_ratio,
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
                            context.tracker.people_tracks,
                            close_people,
                            context.wide_rink_h,
                            context.close_rink_h,
                            img_shape=frame.shape,
                            max_dist=rink_match_max_dist,
                        )

                        logger.debug("close_to_wide_tid=%d", len(close_to_wide_tid) if close_to_wide_tid is not None else 0)
                        # Then we extract only the helmets that has been connected to people tracks across screens
                        helmet_crops = build_helmet_crops_for_wide_ids(
                            close_helmets,
                            close_frame,
                            helmet_person_matches,
                            close_to_wide_tid,
                        )

                        logger.debug("helmet_crops=%d", len(helmet_crops) if helmet_crops is not None else 0)

                    # Give associated helmet crops to the ocr_worker
                    # The ocr_worker will then perform ocr on the crops.
                    for h in helmet_crops:
                        if context.tracker.is_person_confirmed(h["track_id"]):
                            logger.debug("Skipping confirmed track_id=%s", h['track_id'])
                            continue
                        context.ocr_worker.submit(h)

            # Evaluate if the detection is good enough to be set for tracks
            with _section(profiler, "ocr_drain_assign"):
                helmets_res = context.ocr_worker.drain_results()
                context.tracker.assign_helmet_numbers_to_people(helmets_res)  # Includes all voting logic.

            # Annotate frames
            with _section(profiler, "annotate_wide"):
                annotated_frame = context.tracker.annotate(frame)

            rink_canvas = None
            # Creates the Rink-Space window
            # This is used as a canvas for points representing people tracks on both cams
            if context.wide_rink_h is not None or context.close_rink_h is not None:
                with _section(profiler, "build_rink_view"):
                    rink_canvas = build_rink_view(
                        wide_tracks=context.tracker.people_tracks if len(context.tracker.people_tracks) > 0 else None,
                        close_people=close_people,
                        wide_rink_h=context.wide_rink_h,
                        close_rink_h=context.close_rink_h,
                        bounds=rink_bounds,
                        canvas_size=context.rink_canvas_size,
                        img_shape=frame.shape,
                        max_dist=rink_match_max_dist,
                        horizontal=True,
                        draw_center_line=False,
                        draw_center_circle=True,
                        center_circle_radius=4.5,
                        red_lines=rink_red_lines,
                    )

            # Draws the wide camera ROIs onto the annotated frame.
            with _section(profiler, "draw_overlays"):
                draw_roi_lines(annotated_frame, context.yolo_roi, context.ocr_roi)

                # Draw finish line
                if context.finish_line is not None:
                    draw_finish_line_overlay(annotated_frame, context.finish_line)

            # Calculate FPS
            now = time.perf_counter()
            if prev_frame_time is not None:
                elapsed = now - prev_frame_time
                if elapsed > 0:
                    # Instantaneous fps
                    fps_inst = 1.0 / elapsed
                    # Exponential Moving Average FPS
                    fps_ema = fps_inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * fps_inst)
            prev_frame_time = now

            if context.ocr_bench.should_log(now):
                stats = context.ocr_worker.get_stats()
                logger.info("%s", context.ocr_bench.format_line(stats))

            # Profiler periodic logging
            if profiler and profiler.should_log():
                logger.info("PROFILER:\n%s", profiler.format())

            fps_text = f"FPS: {fps_ema:.1f}" if fps_ema > 0 else "FPS: --"

            # Update the lap_panel
            with _section(profiler, "update_lap_panel"):
                context.lap_panel_state = update_lap_panel_state(
                    context.lap_panel_state,
                    tracker=context.tracker,
                    finish_line=context.finish_line,
                    height=context.lap_panel_height,
                    width=context.lap_panel_width,
                )

            # Compose close camera view
            with _section(profiler, "compose_close_vis"):
                # Fallback for when we don't have a synced close frame.
                if close_vis is None and latest_close_frame is not None:
                    close_vis = latest_close_frame.copy()
                # Draws the ROI lines onto the close camera frame.
                draw_roi_lines(close_vis, context.close_yolo_roi, context.close_ocr_roi)

                # Small FPS counter on the close camera feed (top-left corner)
                if close_vis is not None and fps_ema > 0:
                    cv2.putText(
                        close_vis,
                        fps_text,
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                close_subtitle = None
                if latest_close_frame is None:
                    close_subtitle = "No close frame"
                elif context.close_rink_h is not None and synced_close_item is None:
                    close_subtitle = "Unsynced preview"

            # Builds the two windows (rink and two cam feeds) + display
            with _section(profiler, "compose_dashboard"):
                display_frame = compose_display_canvas(
                    annotated_frame,
                    close_vis,
                    context.display_panel_size,
                    wide_subtitle=None,
                    close_subtitle=close_subtitle,
                )
                dashboard_frame = compose_dashboard_canvas(
                    context.lap_panel_state.panel,
                    display_frame,
                    rink_canvas,
                    context.window_layout,
                )
                last_dashboard_frame = dashboard_frame
                cv2.imshow(dashboard_window_name, dashboard_frame)

            # Non-blocking key press check (waitKey is outside the profiler section).
            key = cv2.waitKey(1) & 0xFF
            with _section(profiler, "handle_keypress"):
                # handle_keypress() then handles the keypress
                # and returns results based on what key is pressed.
                key_result = handle_keypress(
                    key=key,
                    state=context.keyboard_state,
                    tracker=context.tracker,
                    pipeline=context.pipeline,
                    pipeline_close=context.pipeline_close,
                    frame=frame,
                    close_preview_frame=context.close_preview_frame,
                    yolo_roi_path=config["Path"]["YOLO_ROI_PATH"],
                    ocr_roi_path=config["Path"]["OCR_ROI_PATH"],
                    close_yolo_roi_path=config["Path"]["CLOSE_YOLO_ROI_PATH"],
                    close_ocr_roi_path=config["Path"]["CLOSE_OCR_ROI_PATH"],
                    finish_line_path=config["Path"]["FINISH_LINE_PATH"],
                )

            # Give new state to the keyboard_state controller.
            context.keyboard_state = key_result.state
            context.yolo_roi = context.keyboard_state.yolo_roi
            context.ocr_roi = context.keyboard_state.ocr_roi
            context.finish_line = context.keyboard_state.finish_line
            context.close_yolo_roi = context.keyboard_state.close_yolo_roi
            context.close_ocr_roi = context.keyboard_state.close_ocr_roi
            if key_result.should_quit:
                break
            if context.keyboard_state.paused:
                continue
    finally:
        # Print profiler summary on exit
        if profiler:
            logger.info(profiler.summary())
        shutdown(context)


# Main guard needed for multiprocessing
if __name__ == '__main__':
    freeze_support()
    config = load_config("data/config.ini")
    main(config)

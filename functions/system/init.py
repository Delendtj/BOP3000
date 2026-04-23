from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2

from hardware_detector import HardwareDetector
from pipeline.async_pipeline import AsyncFramePipeline
from utilities.benchmark import OCRThroughputStats
from utilities.downscale_to_1080p import downscale_to_1080p

from functions.ocr.ocr_worker import OCRWorker
from functions.spatial.homography import load_close_homography, load_wide_homography
from functions.spatial.roi.setup import load_or_select_close_rois, load_or_select_wide_rois
from functions.system.controls import KeyboardControlState
from functions.tracking.tracker import Tracker
from functions.visualization.helmet_gui import prompt_race_setup
from functions.visualization.lap_panel import LapPanelState
from functions.visualization.visualization import compute_window_layout, get_screen_size, setup_window


@dataclass
class AppContext:
    model: Any
    tracker: Tracker
    pipeline: AsyncFramePipeline
    pipeline_close: AsyncFramePipeline
    ocr_worker: OCRWorker
    window_layout: dict
    display_panel_size: tuple[int, int]
    rink_canvas_size: tuple[int, int]
    lap_panel_width: int
    lap_panel_height: int
    yolo_roi: Any
    ocr_roi: Any
    finish_line: Any
    close_yolo_roi: Any
    close_ocr_roi: Any
    wide_rink_h: Any
    close_rink_h: Any
    keyboard_state: KeyboardControlState
    lap_panel_state: LapPanelState
    close_buffer: deque
    latest_close_item: Any
    total_laps: int
    helmet_numbers: list[str]
    fps: float
    close_preview_frame: Any
    ocr_bench: OCRThroughputStats


def initialize(config, *, dashboard_window_name: str) -> AppContext | None:
    """
    Initialize needed variables/object that the main inference loop needs.
    It stores them within an instance of a AppContext class.
    """

    startup_setup = prompt_race_setup()
    if startup_setup is None:
        print("Startup cancelled before race setup was completed.")
        return None

    helmet_numbers = startup_setup.helmet_numbers
    total_laps = startup_setup.total_laps

    detector = HardwareDetector(config["Model"])
    model = detector.initialize_model()
    print(f"Loaded {len(helmet_numbers)} helmet numbers from startup GUI.")

    screen_width, screen_height = get_screen_size()
    window_layout = compute_window_layout(screen_width, screen_height)
    display_panel_size = window_layout["display_panel_size"]
    rink_canvas_size = window_layout["rink_canvas_size"]
    lap_panel_width, lap_panel_height = window_layout["lap_panel_size"]

    preview_cap = cv2.VideoCapture(config["Path"]["WIDE_SOURCE"])
    fps = preview_cap.get(cv2.CAP_PROP_FPS)
    wide_ret, preview_frame = preview_cap.read()
    preview_cap.release()
    preview_frame = downscale_to_1080p(preview_frame)
    if not wide_ret:
        raise RuntimeError("Could not read initial frame for ROI.")

    close_preview_cap = cv2.VideoCapture(config["Path"]["CLOSE_SOURCE"])
    close_ret, close_preview_frame = close_preview_cap.read()
    close_preview_cap.release()
    close_preview_frame = downscale_to_1080p(close_preview_frame)
    if not close_ret:
        raise RuntimeError("Could not read initial close frame for ROI.")

    yolo_roi, ocr_roi, finish_line = load_or_select_wide_rois(
        preview_frame=preview_frame,
        yolo_roi_path=config["Path"]["YOLO_ROI_PATH"],
        ocr_roi_path=config["Path"]["OCR_ROI_PATH"],
        finish_line_path=config["Path"]["FINISH_LINE_PATH"],
    )

    setup_window(window_layout, window_name=dashboard_window_name)

    close_yolo_roi, close_ocr_roi = load_or_select_close_rois(
        close_preview_frame=close_preview_frame,
        close_yolo_roi_path=config["Path"]["CLOSE_YOLO_ROI_PATH"],
        close_ocr_roi_path=config["Path"]["CLOSE_OCR_ROI_PATH"],
    )

    tracker = Tracker(
        config["Runtime"]["OCR_VOTE"],
        config["Runtime"]["CONF_THRESHOLD"],
        ocr_roi,
        frame_rate=fps,
        finish_line=finish_line,
        total_laps=total_laps,
        accepted_numbers=helmet_numbers # The helmet numbers from the imported CSV
    )

    pipeline = AsyncFramePipeline(
        source=config["Path"]["WIDE_SOURCE"],
        frame_skip=config["Runtime"]["FRAME_SKIP"],
        queue_size=3,
        inference_roi=yolo_roi,
    )
    pipeline_close = AsyncFramePipeline(
        source=config["Path"]["CLOSE_SOURCE"],
        frame_skip=config["Runtime"]["FRAME_SKIP"],
        queue_size=3,
        inference_roi=close_yolo_roi,
    )

    ocr_config = config.get("OCR", {})
    ocr_worker = OCRWorker(
        ocr_base_url=ocr_config.get("BASE_URL"),
        ocr_model=ocr_config.get("MODEL"),
        ocr_prompt=ocr_config.get("PROMPT"),
        ocr_timeout=ocr_config.get("TIMEOUT"),
    )
    ocr_worker.start()
    pipeline.start()
    pipeline_close.start()

    close_buffer = deque(maxlen=32)
    latest_close_item = None

    wide_rink_h = load_wide_homography(config["Path"]["RINK_WIDE_H_PATH"])
    close_rink_h = load_close_homography(config["Path"]["RINK_CLOSE_H_PATH"])

    keyboard_state = KeyboardControlState(
        yolo_roi=yolo_roi,
        ocr_roi=ocr_roi,
        finish_line=finish_line,
        close_yolo_roi=close_yolo_roi,
        close_ocr_roi=close_ocr_roi,
        paused=False,
    )

    return AppContext(
        model=model,
        tracker=tracker,
        pipeline=pipeline,
        pipeline_close=pipeline_close,
        ocr_worker=ocr_worker,
        window_layout=window_layout,
        display_panel_size=display_panel_size,
        rink_canvas_size=rink_canvas_size,
        lap_panel_width=lap_panel_width,
        lap_panel_height=lap_panel_height,
        yolo_roi=yolo_roi,
        ocr_roi=ocr_roi,
        finish_line=finish_line,
        close_yolo_roi=close_yolo_roi,
        close_ocr_roi=close_ocr_roi,
        wide_rink_h=wide_rink_h,
        close_rink_h=close_rink_h,
        keyboard_state=keyboard_state,
        lap_panel_state=LapPanelState(),
        close_buffer=close_buffer,
        latest_close_item=latest_close_item,
        total_laps=total_laps,
        helmet_numbers=helmet_numbers,
        fps=fps,
        close_preview_frame=close_preview_frame,
        ocr_bench=OCRThroughputStats(log_every_sec=2.0),
    )


def shutdown(context: AppContext) -> None:
    context.ocr_worker.stop()
    context.pipeline.stop()
    context.pipeline_close.stop()
    cv2.destroyAllWindows()

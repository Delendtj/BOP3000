from __future__ import annotations

from dataclasses import dataclass
import cv2

from functions.spatial.roi.io import save_line, save_roi
from functions.spatial.roi.selection import select_line, select_roi
from functions.spatial.roi.setup import select_ocr_roi_inside_yolo
from functions.spatial.roi.validation import roi_inside_roi, validate_finish_line


@dataclass
class KeyboardControlState:
    yolo_roi: object = None
    ocr_roi: object = None
    finish_line: object = None
    close_yolo_roi: object = None
    close_ocr_roi: object = None
    paused: bool = False


@dataclass
class KeyboardControlResult:
    state: KeyboardControlState
    should_quit: bool = False


@dataclass
class PauseResult:
    should_quit: bool = False
    should_resume: bool = False


def pause(last_frame, window_name: str) -> PauseResult:
    paused_frame = last_frame.copy()
    cv2.putText(
        paused_frame,
        "PAUSED (Space to resume)",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, paused_frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        return PauseResult(should_quit=True)
    if key == ord(" "):
        return PauseResult(should_resume=True)
    return PauseResult()


def handle_keypress(
    *,
    key: int,
    state: KeyboardControlState,
    tracker,
    pipeline,
    pipeline_close,
    frame,
    close_preview_frame,
    yolo_roi_path: str,
    ocr_roi_path: str,
    close_yolo_roi_path: str,
    close_ocr_roi_path: str,
    finish_line_path: str,
) -> KeyboardControlResult:
    if key == 27:
        return KeyboardControlResult(state=state, should_quit=True)

    if key == ord(" "):
        state.paused = True
        return KeyboardControlResult(state=state)

    if key == ord("r"):
        new_yolo_roi = select_roi(frame, window_name="YOLO ROI Selector")
        if new_yolo_roi is not None:
            state.yolo_roi = new_yolo_roi
            pipeline.set_inference_roi(state.yolo_roi)
            save_roi(yolo_roi_path, state.yolo_roi)
            if state.ocr_roi is not None and not roi_inside_roi(state.ocr_roi, state.yolo_roi):
                print("Current OCR ROI is outside updated YOLO ROI. Press 'o' to redraw OCR ROI.")
                state.ocr_roi = None
                tracker.set_roi(None)
            finish_line_error = validate_finish_line(state.finish_line, frame_shape=frame.shape, roi=state.yolo_roi)
            if finish_line_error is not None:
                print(f"Current finish line was cleared: {finish_line_error}")
                state.finish_line = None
                tracker.set_finish_line(None)
        return KeyboardControlResult(state=state)

    if key == ord("o"):
        if state.yolo_roi is None:
            print("Define YOLO ROI first (press 'r').")
        else:
            new_ocr_roi = select_ocr_roi_inside_yolo(frame, state.yolo_roi)
            if new_ocr_roi is not None:
                state.ocr_roi = new_ocr_roi
                tracker.set_roi(state.ocr_roi)
                save_roi(ocr_roi_path, state.ocr_roi)
        return KeyboardControlResult(state=state)

    if key == ord("f"):
        new_finish_line = select_line(frame, window_name="Finish Line Selector")
        if new_finish_line is not None:
            finish_line_error = validate_finish_line(
                new_finish_line,
                frame_shape=frame.shape,
                roi=state.yolo_roi,
            )
            if finish_line_error is not None:
                print(f"Finish line not saved: {finish_line_error}")
            else:
                line_changed = new_finish_line != state.finish_line
                state.finish_line = new_finish_line
                tracker.set_finish_line(state.finish_line)
                if line_changed:
                    print("Finish line updated. Lap counts reset.")
                save_line(finish_line_path, state.finish_line)
                save_roi(ocr_roi_path, state.ocr_roi)
        return KeyboardControlResult(state=state)

    if key == ord("c"):
        if state.close_yolo_roi is None:
            state.close_yolo_roi = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
            if state.close_yolo_roi is not None:
                save_roi(close_yolo_roi_path, state.close_yolo_roi)
                pipeline_close.set_inference_roi(state.close_yolo_roi)
        else:
            new_close_yolo = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
            if new_close_yolo is not None:
                state.close_yolo_roi = new_close_yolo
                save_roi(close_yolo_roi_path, state.close_yolo_roi)
                pipeline_close.set_inference_roi(state.close_yolo_roi)
        return KeyboardControlResult(state=state)

    if key == ord("v"):
        if state.close_yolo_roi is None:
            print("Define close YOLO ROI first (press 'c').")
        else:
            new_close_ocr = select_ocr_roi_inside_yolo(close_preview_frame, state.close_yolo_roi)
            if new_close_ocr is not None:
                state.close_ocr_roi = new_close_ocr
                save_roi(close_ocr_roi_path, state.close_ocr_roi)
        return KeyboardControlResult(state=state)

    return KeyboardControlResult(state=state)

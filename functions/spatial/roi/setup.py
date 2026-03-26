from functions.spatial.roi.io import load_line, load_roi, save_roi
from functions.spatial.roi.selection import select_roi
from functions.spatial.roi.validation import roi_inside_roi, validate_finish_line


def load_or_select_wide_rois(
    *,
    preview_frame,
    yolo_roi_path: str,
    ocr_roi_path: str,
    finish_line_path: str,
):
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
        selected_ocr = select_roi(preview_frame, window_name="OCR ROI Selector")
        if selected_ocr is not None and roi_inside_roi(selected_ocr, yolo_roi):
            ocr_roi = selected_ocr
            save_roi(ocr_roi_path, ocr_roi)
        else:
            print("OCR ROI must be inside YOLO ROI. Draw again or press Esc to cancel.")

    finish_line = load_line(finish_line_path)
    finish_line_error = validate_finish_line(
        finish_line,
        frame_shape=preview_frame.shape,
        roi=yolo_roi,
    )
    if finish_line_error is not None:
        print(f"Ignoring saved finish line: {finish_line_error}")
        finish_line = None

    return yolo_roi, ocr_roi, finish_line


def load_or_select_close_rois(
    *,
    close_preview_frame,
    close_yolo_roi_path: str,
    close_ocr_roi_path: str,
):
    close_yolo_roi = load_roi(close_yolo_roi_path)
    if close_yolo_roi is None:
        close_yolo_roi = select_roi(close_preview_frame, window_name="Close YOLO ROI Selector")
        if close_yolo_roi is not None:
            save_roi(close_yolo_roi_path, close_yolo_roi)

    close_ocr_roi = load_roi(close_ocr_roi_path)
    if close_ocr_roi is not None and close_yolo_roi is not None and not roi_inside_roi(close_ocr_roi, close_yolo_roi):
        print("Loaded close OCR ROI is outside close YOLO ROI. Please redraw.")
        close_ocr_roi = None

    if close_ocr_roi is None and close_yolo_roi is not None:
        selected_close_ocr = select_roi(
            close_preview_frame,
            window_name="Close OCR ROI Selector",
        )
        if selected_close_ocr is not None and roi_inside_roi(selected_close_ocr, close_yolo_roi):
            close_ocr_roi = selected_close_ocr
            save_roi(close_ocr_roi_path, close_ocr_roi)
        else:
            print("Close OCR ROI must be inside close YOLO ROI. Draw again or press Esc to cancel.")

    return close_yolo_roi, close_ocr_roi


def select_ocr_roi_inside_yolo(frame, yolo_roi):
    if frame is None or yolo_roi is None:
        return None

    while True:
        candidate = select_roi(frame, window_name="OCR ROI Selector")
        if candidate is None:
            return None
        if roi_inside_roi(candidate, yolo_roi):
            return candidate

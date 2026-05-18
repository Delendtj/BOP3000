from __future__ import annotations

from dataclasses import dataclass

import supervision as sv

from functions.detection.roi_inference import keep_detections_inside_roi, shift_detections_to_full_frame


@dataclass
class CloseInferenceResult:
    detections: sv.Detections
    helmets: sv.Detections
    people: sv.Detections

def run_close_inference(
    model,
    frame_item,
    inference_config: dict,
    helmet_class_id: int,
    person_class_id: int,
    yolo_roi=None,
    ocr_roi=None,
) -> CloseInferenceResult:
    """
    Helper for running inference on the close camera and running ROI logic on the results.
    """

    result = model(
        frame_item.inference_frame if frame_item.inference_frame is not None else frame_item.frame,
        conf=inference_config["conf"],
        iou=inference_config["iou"],
        max_det=inference_config["max_det"],
        imgsz=inference_config["imgsz"],
        half=inference_config["half"],
        device=inference_config["device"],
        verbose=inference_config["verbose"],
        classes=[helmet_class_id, person_class_id],
    )[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = shift_detections_to_full_frame(detections, frame_item.inference_offset)

    if yolo_roi is not None and frame_item.inference_frame is None:
        detections = keep_detections_inside_roi(detections, yolo_roi)

    helmets = detections[detections.class_id == helmet_class_id]
    people = detections[detections.class_id == person_class_id]

    if ocr_roi is not None:
        helmets = keep_detections_inside_roi(helmets, ocr_roi)

    return CloseInferenceResult(
        detections=detections,
        helmets=helmets,
        people=people,
    )

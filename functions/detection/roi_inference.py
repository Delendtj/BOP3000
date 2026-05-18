import cv2
import numpy as np
import supervision as sv

from functions.spatial.roi.validation import bbox_center_in_roi


def crop_frame_to_roi(frame, roi, padding=0):
    if frame is None or roi is None or len(roi) < 3:
        return frame, (0, 0)

    frame_h, frame_w = frame.shape[:2]
    polygon = np.asarray(roi, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon)

    pad = int(max(0, padding))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)

    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0)

    return frame[y1:y2, x1:x2], (x1, y1)


def shift_detections_to_full_frame(detections: sv.Detections, offset):
    if detections is None or len(detections) == 0:
        return detections

    x_offset, y_offset = offset
    if x_offset == 0 and y_offset == 0:
        return detections

    detections.xyxy[:, [0, 2]] += float(x_offset)
    detections.xyxy[:, [1, 3]] += float(y_offset)
    return detections


def keep_detections_inside_roi(detections: sv.Detections, roi):
    if detections is None or roi is None or len(detections) == 0:
        return detections

    keep_mask = np.array(
        [
            bbox_center_in_roi((x1, y1, x2, y2), roi)
            for x1, y1, x2, y2 in detections.xyxy
        ],
        dtype=bool,
    )
    return detections[keep_mask]

from math import hypot

import cv2
import numpy as np


def bbox_center_in_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    polygon = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0


def point_in_roi(point, roi):
    if point is None or roi is None or len(roi) < 3:
        return False
    px, py = point
    polygon = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


def roi_inside_roi(inner_roi, outer_roi):
    if inner_roi is None or outer_roi is None:
        return False
    if len(inner_roi) < 3 or len(outer_roi) < 3:
        return False
    return all(point_in_roi((x, y), outer_roi) for x, y in inner_roi)

def validate_finish_line(line, frame_shape=None, roi=None, min_length_px=20.0, min_vertical_span_px=20.0):
    if line is None:
        return None
    if len(line) != 2:
        return "Finish line requires exactly two points."

    try:
        points = [tuple(map(int, pt)) for pt in line]
    except (TypeError, ValueError):
        return "Finish line contains invalid points."

    (x1, y1), (x2, y2) = points
    if hypot(float(x2) - float(x1), float(y2) - float(y1)) < float(min_length_px):
        return "Finish line is too short."
    if abs(int(y2) - int(y1)) < int(min_vertical_span_px):
        return "Finish line must have clear vertical span for left-to-right counting."

    if frame_shape is not None:
        height, width = frame_shape[:2]
        for px, py in points:
            if px < 0 or py < 0 or px >= width or py >= height:
                return "Finish line must stay inside the frame."

    if roi is not None and not all(point_in_roi(pt, roi) for pt in points):
        return "Finish line endpoints must be inside the YOLO ROI."

    return None

import cv2
import json
import os
import numpy as np


def load_roi(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "points" in data:
            points = [(int(x), int(y)) for x, y in data["points"]]
            if len(points) >= 3:
                return points
        # Backward-compatible rectangle format.
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (
                int(data["x1"]),
                int(data["y1"]),
                int(data["x2"]),
                int(data["y2"]),
            )
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return None


def save_roi(path, roi):
    if roi is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in roi]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def bbox_center_in_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    polygon = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0


def filter_dets_by_roi(det, roi):
    if det is None or roi is None:
        return det
    kept = []
    for row in det:
        x1, y1, x2, y2 = row[:4]
        if bbox_center_in_roi((x1, y1, x2, y2), roi):
            kept.append(row)
    if not kept:
        return None
    return np.asarray(kept)

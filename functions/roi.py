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
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in roi]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_line(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "points" not in data:
            return None
        points = [(int(x), int(y)) for x, y in data["points"]]
        if len(points) == 2:
            return points
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return None


def save_line(path, line):
    if line is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in line]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def select_roi(frame, window_name="ROI Selector"):
    if frame is None:
        return None

    points = []

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        canvas = frame.copy()

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
            for px, py in points:
                cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)

        cv2.putText(
            canvas,
            "Left click: add  Right click/U: undo  C: clear  Enter: save  Esc: cancel",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key in (13, 10):
            if len(points) >= 3:
                cv2.destroyWindow(window_name)
                return points
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None
        elif key in (ord("u"), ord("U"), 8):
            if points:
                points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()


def select_line(frame, window_name="Finish Line"):
    if frame is None:
        return None

    points = []

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        canvas = frame.copy()

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 165, 255), 2)
            for px, py in points:
                cv2.circle(canvas, (px, py), 5, (0, 165, 255), -1)

        cv2.putText(
            canvas,
            "Right click/U: undo  C: clear  Enter: save  Esc: cancel",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key in (13, 10):
            if len(points) == 2:
                cv2.destroyWindow(window_name)
                return points
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None
        elif key in (ord("u"), ord("U"), 8):
            if points:
                points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()


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

from __future__ import annotations

import cv2

"""
Helper functions for drawing/visualizing homography connections used in main.
"""

def draw_bboxes(frame, detections, color, thickness=2):
    if frame is None or detections is None or len(detections) == 0:
        return
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_match_lines(frame, detections_a, detections_b, matches, get_point_a, get_point_b, color, thickness=2):
    if frame is None or not matches:
        return
    for idx_a, idx_b, _ in matches:
        ax, ay = get_point_a(detections_a.xyxy[idx_a])
        bx, by = get_point_b(detections_b.xyxy[idx_b])
        cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)


def draw_rink_points(frame, canvas_points, color, radius=5, thickness=-1):
    if frame is None or not canvas_points:
        return
    h, w = frame.shape[:2]
    for x, y in canvas_points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (int(x), int(y)), radius, color, thickness)


def draw_rink_match_lines(frame, matches, points_a, points_b, color, thickness=2):
    if frame is None or not matches:
        return
    h, w = frame.shape[:2]
    for idx_a, idx_b, _ in matches:
        ax, ay = points_a[idx_a]
        bx, by = points_b[idx_b]
        if 0 <= ax < w and 0 <= ay < h and 0 <= bx < w and 0 <= by < h:
            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)

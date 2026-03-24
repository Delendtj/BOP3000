from __future__ import annotations

import cv2

"""
Helper functions for drawing/visualizing homography connections
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

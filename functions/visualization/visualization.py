from __future__ import annotations

import cv2
import numpy as np
import tkinter as tk
import screeninfo

from functions.spatial.rink_projection import rink_to_canvas

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


def build_rink_canvas(
    bounds: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    line_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    draw_center_line: bool = False,
    draw_center_circle: bool = True,
    center_circle_radius: float = 4.5,
    horizontal: bool = False,
    red_lines: tuple[float, ...] = (),
) -> np.ndarray:
    w, h = canvas_size
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    cvx1, cvy1 = rink_to_canvas(bounds[0], bounds[2], bounds, canvas_size, horizontal=horizontal)
    cvx2, cvy2 = rink_to_canvas(bounds[1], bounds[3], bounds, canvas_size, horizontal=horizontal)
    x1, y1 = min(cvx1, cvx2), min(cvy1, cvy2)
    x2, y2 = max(cvx1, cvx2), max(cvy1, cvy2)
    canvas[y1:y2, x1] = line_color
    canvas[y1:y2, x2 - 1] = line_color
    canvas[y1, x1:x2] = line_color
    canvas[y2 - 1, x1:x2] = line_color

    if draw_center_line:
        cx, cy_top = rink_to_canvas(0.0, bounds[3], bounds, canvas_size, horizontal=horizontal)
        _, cy_bot = rink_to_canvas(0.0, bounds[2], bounds, canvas_size, horizontal=horizontal)
        canvas[min(cy_top, cy_bot):max(cy_top, cy_bot), cx] = line_color

    if draw_center_circle and center_circle_radius > 0:
        cx, cy = rink_to_canvas(0.0, 0.0, bounds, canvas_size, horizontal=horizontal)
        rx, ry = rink_to_canvas(center_circle_radius, 0.0, bounds, canvas_size, horizontal=horizontal)
        radius_px = int(((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5)
        if radius_px > 0:
            cv2.circle(canvas, (cx, cy), radius_px, line_color, 2)

    for y in red_lines:
        x1, y1 = rink_to_canvas(bounds[0], y, bounds, canvas_size, horizontal=horizontal)
        x2, y2 = rink_to_canvas(bounds[1], y, bounds, canvas_size, horizontal=horizontal)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return canvas


def build_display_panel(frame, title, panel_size, subtitle=None):
    panel_w, panel_h = panel_size
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    if frame is not None:
        src_h, src_w = frame.shape[:2]
        scale = min(panel_w / src_w, panel_h / src_h)
        resized_w = max(1, int(src_w * scale))
        resized_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame, (resized_w, resized_h))
        x = (panel_w - resized_w) // 2
        y = (panel_h - resized_h) // 2
        panel[y:y + resized_h, x:x + resized_w] = resized

    cv2.putText(
        panel,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    if subtitle:
        cv2.putText(
            panel,
            subtitle,
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    return panel


def compose_display_canvas(wide_frame, close_frame, panel_size, wide_subtitle=None, close_subtitle=None):
    wide_panel = build_display_panel(wide_frame, "Wide", panel_size, wide_subtitle)
    close_panel = build_display_panel(close_frame, "Close", panel_size, close_subtitle)
    return np.hstack([wide_panel, close_panel])


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    current_screen = get_monitor_from_coord(root.winfo_x(), root.winfo_y())

    return current_screen.width, current_screen.height

def get_monitor_from_coord(x, y):
    monitors = screeninfo.get_monitors()

    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]

def _clamp(value, minimum, maximum):
    return max(minimum, min(int(value), maximum))


def compute_window_layout(screen_width, screen_height):
    padding = max(12, int(min(screen_width, screen_height) * 0.01))
    lap_width = _clamp(screen_width * 0.17, 280, 420)
    right_width = max(720, screen_width - lap_width - (padding * 3))
    window_gap = padding + 48
    available_right_height = max(620, screen_height - (padding * 2) - window_gap)
    top_height = _clamp(available_right_height * 0.52, 320, available_right_height - 240)
    bottom_height = max(220, available_right_height - top_height)

    return {
        "display_panel_size": (max(320, right_width // 2), top_height),
        "rink_canvas_size": (right_width, bottom_height),
        "lap_panel_size": (lap_width, screen_height - (padding * 2)),
        "multi_cam_pos": (lap_width + (padding * 2), padding),
        "lap_pos": (padding, padding),
        "rink_pos": (lap_width + (padding * 2), padding + top_height + window_gap),
    }

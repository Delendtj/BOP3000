from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from functions.homography import Homography, project_point
from functions.undistort import undistort_points

"""
Helpers for rink projections used inside main.
"""

def project_to_rink(homography: Homography, x: float, y: float) -> tuple[float, float] | None:
    """
    Project an image-space point into rink coordinates using image->rink homography.
    """
    return project_point(homography, x, y)


def bbox_bottom_center_xyxy(bbox) -> tuple[float, float]:
    return float((bbox[0] + bbox[2]) / 2.0), float(bbox[3])


def map_rink_xy(rink_pt: tuple[float, float]) -> tuple[float, float]:
    """
    Returns swapped axes only
    """
    rink_x, rink_y = rink_pt
    return rink_y, rink_x


def project_bbox_to_rink(
    bbox,
    homography: Homography,
    undistort: bool = False,
    img_shape=None,
) -> tuple[float, float] | None:
    x, y = bbox_bottom_center_xyxy(bbox)
    if undistort:
        undist = undistort_points([(x, y)], img_shape=img_shape)
        if undist is None or len(undist) == 0:
            return None
        x, y = map(float, undist[0])
    rink_pt = project_to_rink(homography, x, y)
    if rink_pt is None:
        return None
    return map_rink_xy(rink_pt)


def project_bboxes_to_rink_canvas(
    bboxes,
    homography: Homography | None,
    bounds: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    *,
    horizontal: bool = False,
    undistort: bool = False,
    img_shape=None,
) -> list[tuple[tuple[float, float], tuple[int, int]]]:
    if homography is None or bboxes is None:
        return []

    points = []
    for bbox in bboxes:
        rink_pt = project_bbox_to_rink(
            bbox,
            homography,
            undistort=undistort,
            img_shape=img_shape,
        )
        if rink_pt is None:
            continue
        canvas_pt = rink_to_canvas(
            rink_pt[0],
            rink_pt[1],
            bounds,
            canvas_size,
            horizontal=horizontal,
        )
        points.append((rink_pt, canvas_pt))
    return points


def rink_to_canvas(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    horizontal: bool = False,
) -> tuple[int, int]:
    """
    Convert rink coordinates to canvas pixel coordinates.
    bounds: (min_x, max_x, min_y, max_y)
    canvas_size: (width, height)
    """
    min_x, max_x, min_y, max_y = bounds
    w, h = canvas_size
    nx = 0.0 if max_x == min_x else (x - min_x) / (max_x - min_x)
    ny = 0.0 if max_y == min_y else (y - min_y) / (max_y - min_y)
    if horizontal:
        cx = int(ny * w)
        cy = int((1.0 - nx) * h)
    else:
        cx = int(nx * w)
        cy = int((1.0 - ny) * h)
    return cx, cy


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
    # Rink border
    cvx1, cvy1 = rink_to_canvas(bounds[0], bounds[2], bounds, canvas_size, horizontal=horizontal)
    cvx2, cvy2 = rink_to_canvas(bounds[1], bounds[3], bounds, canvas_size, horizontal=horizontal)
    x1, y1 = min(cvx1, cvx2), min(cvy1, cvy2)
    x2, y2 = max(cvx1, cvx2), max(cvy1, cvy2)
    canvas[y1:y2, x1] = line_color
    canvas[y1:y2, x2 - 1] = line_color
    canvas[y1, x1:x2] = line_color
    canvas[y2 - 1, x1:x2] = line_color
    # Center line (optional)
    if draw_center_line:
        cx, cy_top = rink_to_canvas(0.0, bounds[3], bounds, canvas_size, horizontal=horizontal)
        _, cy_bot = rink_to_canvas(0.0, bounds[2], bounds, canvas_size, horizontal=horizontal)
        canvas[min(cy_top, cy_bot):max(cy_top, cy_bot), cx] = line_color

    # Center circle (optional)
    if draw_center_circle and center_circle_radius > 0:
        cx, cy = rink_to_canvas(0.0, 0.0, bounds, canvas_size, horizontal=horizontal)
        rx, ry = rink_to_canvas(center_circle_radius, 0.0, bounds, canvas_size, horizontal=horizontal)
        radius_px = int(((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5)
        if radius_px > 0:
            cv2.circle(canvas, (cx, cy), radius_px, line_color, 2)

    # Red lines at fixed rink y positions (e.g., center and goal lines).
    for y in red_lines:
        x1, y1 = rink_to_canvas(bounds[0], y, bounds, canvas_size, horizontal=horizontal)
        x2, y2 = rink_to_canvas(bounds[1], y, bounds, canvas_size, horizontal=horizontal)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return canvas

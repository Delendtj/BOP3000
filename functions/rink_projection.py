from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from functions.homography import Homography, project_point


def project_to_rink(homography: Homography, x: float, y: float) -> tuple[float, float] | None:
    """
    Project an image-space point into rink coordinates using image->rink homography.
    """
    return project_point(homography, x, y)


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
    line_color=(200, 200, 200),
    draw_center_line: bool = False,
    horizontal: bool = False,
    red_lines: tuple[float, ...] = (),
) -> np.ndarray:
    w, h = canvas_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
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

    # Red lines at fixed rink y positions (e.g., center and goal lines).
    for y in red_lines:
        x1, y1 = rink_to_canvas(bounds[0], y, bounds, canvas_size, horizontal=horizontal)
        x2, y2 = rink_to_canvas(bounds[1], y, bounds, canvas_size, horizontal=horizontal)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return canvas

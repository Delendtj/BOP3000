from __future__ import annotations

from typing import Iterable

import numpy as np
import supervision as sv

from functions.spatial.rink_projection import project_bboxes_to_rink_canvas
from functions.tracking.assignment import hungarian_assign
from functions.visualization.visualization import (
    build_rink_canvas,
    draw_rink_match_lines,
    draw_rink_points,
)


def build_rink_view(
    *,
    wide_tracks: sv.Detections | None,
    close_people: sv.Detections | None,
    wide_rink_h,
    close_rink_h,
    bounds: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    img_shape,
    max_dist: float,
    horizontal: bool = True,
    draw_center_line: bool = False,
    draw_center_circle: bool = True,
    center_circle_radius: float = 4.5,
    red_lines: tuple[float, ...] = (),
) -> np.ndarray:
    """
    Simply a helper for building the rink window. (less bloat inside main)
    """
    rink_canvas = build_rink_canvas(
        bounds,
        canvas_size,
        draw_center_line=draw_center_line,
        draw_center_circle=draw_center_circle,
        center_circle_radius=center_circle_radius,
        horizontal=horizontal,
        red_lines=red_lines,
    )

    wide_rink_points = project_bboxes_to_rink_canvas(
        wide_tracks.xyxy if wide_tracks is not None and len(wide_tracks) > 0 else None,
        wide_rink_h,
        bounds,
        canvas_size,
        horizontal=horizontal,
        undistort=True,
        img_shape=img_shape,
    )
    close_rink_points = project_bboxes_to_rink_canvas(
        close_people.xyxy if close_people is not None and len(close_people) > 0 else None,
        close_rink_h,
        bounds,
        canvas_size,
        horizontal=horizontal,
    )

    draw_rink_points(
        rink_canvas,
        [canvas_xy for _, canvas_xy in wide_rink_points],
        color=(0, 0, 255),
    )
    draw_rink_points(
        rink_canvas,
        [canvas_xy for _, canvas_xy in close_rink_points],
        color=(255, 0, 0),
    )

    if wide_rink_points and close_rink_points:
        # p[0] is the rink_xy. Not the canvas_xy
        wide_xy = [p[0] for p in wide_rink_points]
        close_xy = [p[0] for p in close_rink_points]
        matches = hungarian_assign(wide_xy, close_xy, max_dist=max_dist)
        draw_rink_match_lines(
            rink_canvas,
            matches,
            [canvas_xy for _, canvas_xy in wide_rink_points],
            [canvas_xy for _, canvas_xy in close_rink_points],
            color=(0, 200, 0),
        )

    return rink_canvas

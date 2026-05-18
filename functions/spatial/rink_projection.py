from __future__ import annotations

from functions.spatial.homography import Homography, project_point
from functions.spatial.undistort import undistort_points

"""
Helpers for rink projections used inside main.
"""

def bbox_bottom_center_xyxy(bbox) -> tuple[float, float]:
    return float((bbox[0] + bbox[2]) / 2.0), float(bbox[3])


def swap_axes(rink_pt: tuple[float, float]) -> tuple[float, float]:
    """
    Returns swapped axes
    """
    rink_x, rink_y = rink_pt
    return rink_y, rink_x


def project_bbox_to_rink(
    bbox,
    homography: Homography,
    undistort: bool = False,
    img_shape=None,
) -> tuple[float, float] | None:
    """
    Takes the bottom center of the input bbox as this is what we use to represent where a skater is.
    We translate this into undistorted coordinates if its from wide camera.

    Then we project these coordinates into rink coordinates and swap the axes before returning.
    """
    x, y = bbox_bottom_center_xyxy(bbox)
    if undistort:
        undist = undistort_points([(x, y)], img_shape=img_shape)
        if undist is None or len(undist) == 0:
            return None
        x, y = map(float, undist[0])
    rink_pt = project_point(homography, x, y)
    if rink_pt is None:
        return None
    return swap_axes(rink_pt)


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
    """
    Projects bboxes to both rink coordinates and canvas coordinates.

    Returns array of tuple (rink_xy, canvas_xy)
    """
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
    # Normalize to 0-1 value.
    nx = 0.0 if max_x == min_x else (x - min_x) / (max_x - min_x)
    ny = 0.0 if max_y == min_y else (y - min_y) / (max_y - min_y)
    # Then scale it to pixels based on width and height.
    if horizontal:
        cx = int(ny * w)
        cy = int((1.0 - nx) * h)
    else:
        cx = int(nx * w)
        cy = int((1.0 - ny) * h)
    return cx, cy

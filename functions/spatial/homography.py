import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

from functions.spatial.undistort import distort_points


@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray
    source_role: str = "close"
    target_role: str = "wide"


def load_homography(path: str) -> Homography:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    matrix = np.asarray(payload["H"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix in {path}, got {matrix.shape}.")

    direction = payload.get("direction", {})
    return Homography(
        matrix=matrix,
        source_role=str(direction.get("source_role", "close")),
        target_role=str(direction.get("target_role", "wide")),
    )


def project_point(homography: Homography, x: float, y: float) -> tuple[float, float] | None:
    vec = np.array([x, y, 1.0], dtype=np.float32)
    projected = homography.matrix @ vec
    if projected[2] == 0:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


# This is no longer needed?
def invert_homography(homography: Homography) -> Homography:
    inv = np.linalg.inv(homography.matrix)
    return Homography(
        matrix=inv,
        source_role=homography.target_role,
        target_role=homography.source_role,
    )

def map_close_point_to_wide_distorted(
    homography: Homography,
    x: float,
    y: float,
    wide_img_shape,
) -> tuple[float, float] | None:
    """
    Map a close-camera point into distorted wide-camera pixel space.
    Assumes homography maps close -> wide (undistorted).
    """
    if homography.source_role != "close" or homography.target_role != "wide":
        raise ValueError(
            f"Expected homography close->wide, got {homography.source_role}->{homography.target_role}."
        )

    undistorted = project_point(homography, x, y)
    if undistorted is None:
        return None

    distorted = distort_points([undistorted], img_shape=wide_img_shape)

    return float(distorted[0, 0]), float(distorted[0, 1])


def select_close_frame(buffer, target_ts: float, max_delta: float) -> Any | None:
    best_item = None
    best_dt = float("inf")
    for ts, item in buffer:
        dt = abs(ts - target_ts)
        if dt < best_dt:
            best_dt = dt
            best_item = item
    if best_item is None or best_dt > max_delta:
        return None
    return best_item


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    return float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0)


def associate_close_helmets_to_wide_helmet_tracks(
    wide_helmet_tracks: sv.Detections,
    close_helmets: sv.Detections,
    close_frame: np.ndarray,
    homography: Homography,
    max_dist: float,
) -> list[dict]:
    if (
        wide_helmet_tracks is None
        or close_helmets is None
        or len(wide_helmet_tracks) == 0
        or len(close_helmets) == 0
    ):
        return []

    if homography.source_role != "close" or homography.target_role != "wide":
        raise ValueError(
            f"Expected homography close->wide, got {homography.source_role}->{homography.target_role}."
        )

    # might need to remove this later
    inv_h = invert_homography(homography)
    candidate_pairs = []
    for wide_idx, wide_bbox in enumerate(wide_helmet_tracks.xyxy):
        track_id = int(wide_helmet_tracks.tracker_id[wide_idx])
        if track_id == -1:
            continue

        projected = project_point(inv_h, *_bbox_center(wide_bbox))
        if projected is None:
            continue

        px, py = projected
        for close_idx, close_bbox in enumerate(close_helmets.xyxy):
            cx, cy = _bbox_center(close_bbox)
            dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
            if dist <= max_dist:
                candidate_pairs.append((dist, wide_idx, close_idx))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    assigned_wide = set()
    assigned_close = set()
    crops = []

    for _, wide_idx, close_idx in candidate_pairs:
        if wide_idx in assigned_wide or close_idx in assigned_close:
            continue

        assigned_wide.add(wide_idx)
        assigned_close.add(close_idx)

        x1, y1, x2, y2 = close_helmets.xyxy[close_idx]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(close_frame.shape[1], int(x2))
        y2 = min(close_frame.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crops.append(
            {
                "image": close_frame[y1:y2, x1:x2].copy(),
                "bbox": (x1, y1, x2, y2),
                "conf": float(close_helmets.confidence[close_idx]),
                "track_id": int(wide_helmet_tracks.tracker_id[wide_idx]),
            }
        )

    return crops


def associate_close_helmet_crops(
    close_helmets: sv.Detections,
    wide_tracks: sv.Detections,
    close_frame: np.ndarray,
    homography: Homography,
    max_dist: float,
) -> list[dict]:
    return associate_close_helmets_to_wide_helmet_tracks(
        wide_helmet_tracks=wide_tracks,
        close_helmets=close_helmets,
        close_frame=close_frame,
        homography=homography,
        max_dist=max_dist,
    )

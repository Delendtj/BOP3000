import json
from typing import NamedTuple

import numpy as np
import supervision as sv


class Homography(NamedTuple):
    matrix: np.ndarray


def load_homography(path: str) -> Homography:
    with open(path, "r") as fh:
        payload = json.load(fh)
    matrix = np.asarray(payload["H"], dtype=np.float32)
    return Homography(matrix=matrix)


def project_point(homography: Homography, x: float, y: float) -> tuple[float, float] | None:
    vec = np.array([x, y, 1.0], dtype=np.float32)
    projected = homography.matrix @ vec
    if projected[2] == 0:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def select_close_frame(buffer, target_ts: float, max_delta: float) -> np.ndarray | None:
    best = None
    best_dt = float("inf")
    for ts, frame in buffer:
        dt = abs(ts - target_ts)
        if dt < best_dt:
            best_dt = dt
            best = frame
    if best is None or best_dt > max_delta:
        return None
    return best


def build_close_bbox(
    center: tuple[float, float],
    frame_w: int,
    frame_h: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    half_size = int(crop_size // 2)
    x = int(center[0])
    y = int(center[1])
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(frame_w, x + half_size)
    y2 = min(frame_h, y + half_size)
    return x1, y1, x2, y2


def build_close_crops(
    detections: sv.Detections,
    close_frame: np.ndarray,
    homography: Homography,
    crop_size: int,
) -> list[dict]:
    crops = []
    for idx, bbox in enumerate(detections.xyxy):
        track_id = int(detections.tracker_id[idx])
        if track_id == -1:
            continue
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float(bbox[3])
        projected = project_point(homography, center_x, center_y)
        if projected is None:
            continue
        x1, y1, x2, y2 = build_close_bbox(
            projected,
            close_frame.shape[1],
            close_frame.shape[0],
            crop_size,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        crop = close_frame[y1:y2, x1:x2].copy()
        crops.append({
            "image": crop,
            "bbox": (x1, y1, x2, y2),
            "conf": float(detections.confidence[idx]),
            "track_id": track_id,
        })
    return crops


def associate_close_helmet_crops(
    close_helmets: sv.Detections,
    wide_tracks: sv.Detections,
    close_frame: np.ndarray,
    homography: Homography,
    max_dist: float,
) -> list[dict]:
    if close_helmets is None or len(close_helmets) == 0:
        return []

    crops = []
    used = set()

    helmet_centers = []
    for idx, bbox in enumerate(close_helmets.xyxy):
        cx = float((bbox[0] + bbox[2]) / 2.0)
        cy = float((bbox[1] + bbox[3]) / 2.0)
        helmet_centers.append((cx, cy))

    for idx, bbox in enumerate(wide_tracks.xyxy):
        track_id = int(wide_tracks.tracker_id[idx])
        if track_id == -1:
            continue

        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float(bbox[3])
        projected = project_point(homography, center_x, center_y)
        if projected is None:
            continue

        best_idx = None
        best_dist = max_dist
        px, py = projected
        for h_idx, (hx, hy) in enumerate(helmet_centers):
            if h_idx in used:
                continue
            # Pythagoras theorem.
            # Find distance between two points where a = (hx - px) and b = (hy - py)
            dist = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_idx = h_idx

        # Wasn't close enough
        if best_idx is None:
            continue

        used.add(best_idx) # Mark as used
        x1, y1, x2, y2 = close_helmets.xyxy[best_idx]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(close_frame.shape[1], int(x2)) # min of y1 and x2???!!!
        y2 = min(close_frame.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = close_frame[y1:y2, x1:x2].copy()
        crops.append({
            "image": crop,
            "bbox": (x1, y1, x2, y2),
            "conf": float(close_helmets.confidence[best_idx]),
            "track_id": track_id,
        })

    return crops

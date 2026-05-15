import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Homography:
    """
    Wrapper class, contains both homography matrix and its intended direction.
    """
    matrix: np.ndarray
    source_role: str = "close"
    target_role: str = "wide"


def load_homography(path: str) -> Homography:
    """
    Creates a Homography object based on json file path passed in.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    matrix = np.asarray(payload["H"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix in {path}, got {matrix.shape}.")

    direction = payload.get("direction")
    # Ensure the homography json contains direction information.
    if not isinstance(direction, dict):
        raise ValueError(
            f"Homography file {path} is missing a valid 'direction' object "
            "with 'source_role' and 'target_role'."
        )

    source_role = direction.get("source_role")
    target_role = direction.get("target_role")
    if not isinstance(source_role, str) or not source_role:
        raise ValueError(f"Homography file {path} is missing a valid direction.source_role.")
    if not isinstance(target_role, str) or not target_role:
        raise ValueError(f"Homography file {path} is missing a valid direction.target_role.")

    return Homography(
        matrix=matrix,
        source_role=source_role,
        target_role=target_role,
    )

def load_wide_homography(path: str) -> Homography:
    try:
        wide_h = load_homography(path)
        if wide_h.source_role != "wide" or wide_h.target_role != "rink":
            raise RuntimeError(
                f"Wide rink homography must map wide -> rink, got {wide_h.source_role} -> {wide_h.target_role}."
            )
        print("Loaded wide->rink homography.")
    except FileNotFoundError:
        wide_h = None
        print("Wide rink homography missing, rink view will omit wide points.")

    return wide_h


def load_close_homography(path: str) -> Homography:
    try:
        close_h = load_homography(path)
        if close_h.source_role != "close" or close_h.target_role != "rink":
            raise RuntimeError(
                f"Close rink homography must map close -> rink, got {close_h.source_role} -> {close_h.target_role}."
            )
        print("Loaded close->rink homography.")
    except FileNotFoundError:
        close_h = None
        print("Close rink homography missing, rink view will omit close points.")

    return close_h


def project_point(homography: Homography, x: float, y: float) -> tuple[float, float] | None:
    """
    Project point (x,y) into where homography matrix points to.
    """
    vec = np.array([x, y, 1.0], dtype=np.float32)
    # Does the actual homographic transformation
    projected = homography.matrix @ vec
    if projected[2] == 0:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def select_close_frame(buffer, target_ts: float, max_delta: float) -> Any | None:
    """
    Find the frame that is closest to the target timestamp (current frame) from the buffer.
    """
    best_item = None
    best_delta = float("inf")  # Positive infinity.
    for time_stamp, item in buffer:
        delta = abs(time_stamp - target_ts)
        if delta < best_delta:    # Guarantee true on first iteration
            best_delta = delta
            best_item = item
    if best_item is None or best_delta > max_delta:
        return None
    return best_item

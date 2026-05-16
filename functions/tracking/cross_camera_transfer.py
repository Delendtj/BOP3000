from __future__ import annotations

from typing import Iterable

import supervision as sv

from functions.spatial.rink_projection import project_bbox_to_rink
from functions.tracking.assignment import hungarian_assign


def build_close_to_wide_mapping(
    wide_tracks: sv.Detections,
    close_people: sv.Detections,
    # Homography
    wide_rink_h,
    close_rink_h,
    *,
    img_shape,
    max_dist: float,
) -> dict[int, int]:
    if (
            wide_rink_h is None
            or close_rink_h is None
            or wide_tracks is None
            or close_people is None
            or len(wide_tracks) == 0
            or len(close_people) == 0
    ):
        return {}

    wide_rink_pts = []
    wide_track_ids = []
    # Project/map WIDE detections points into rink
    for idx, bbox in enumerate(wide_tracks.xyxy):
        rink_pt = project_bbox_to_rink(bbox, wide_rink_h, undistort=True, img_shape=img_shape)
        if rink_pt is None:
            continue
        wide_rink_pts.append(rink_pt)
        wide_track_ids.append(int(wide_tracks.tracker_id[idx]))

    close_rink_pts = []
    close_person_indices = []
    # Project/map CLOSE detections points into rink
    for idx, bbox in enumerate(close_people.xyxy):
        rink_pt = project_bbox_to_rink(bbox, close_rink_h)
        if rink_pt is None:
            continue
        close_rink_pts.append(rink_pt)
        close_person_indices.append(idx)

    if not wide_rink_pts or not close_rink_pts:
        return {}

    # Matches them with Hungarian Algorithm
    matches = hungarian_assign(wide_rink_pts, close_rink_pts, max_dist=max_dist)
    # Creates the actual map between the close and wide tracks, and returns it.
    mapping = {}
    for wide_idx, close_idx, _ in matches:
        if wide_idx < len(wide_track_ids) and close_idx < len(close_person_indices):
            wide_tid = wide_track_ids[wide_idx]
            if wide_tid != -1:
                mapping[close_person_indices[close_idx]] = wide_tid
    return mapping


def build_helmet_crops_for_wide_ids(
    close_helmets: sv.Detections,
    close_frame,
    helmet_person_matches_close: Iterable[tuple[int, int, float]],
    close_to_wide_tid: dict[int, int],
) -> list[dict]:
    """
    Extract helmets input helmet detections, and returns crops
    if they are connected within helmet_person_matches
    
    Params:
    helmet_person_matches_close = close helmet -> close person
    close_to_wide_tid = close person -> wide person id
    """
    crops = []
    if close_helmets is None or close_frame is None:
        return crops

    for helmet_idx, person_idx, _ in helmet_person_matches_close:
        wide_tid = close_to_wide_tid.get(person_idx)
        if wide_tid is None:
            continue
        x1, y1, x2, y2 = close_helmets.xyxy[helmet_idx]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(close_frame.shape[1], int(x2))
        y2 = min(close_frame.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(
            {
                "image": close_frame[y1:y2, x1:x2].copy(), # Cropped image
                "bbox": (x1, y1, x2, y2),
                "conf": float(close_helmets.confidence[helmet_idx]),
                "track_id": wide_tid,
            }
        )

    return crops

from __future__ import annotations

import supervision as sv

def bbox_top_center_xyxy(bbox):
    x1, y1, x2, _ = bbox
    return (x1 + x2) / 2.0, y1


def point_in_bbox(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def match_close_helmets_to_people(
    close_helmets: sv.Detections,
    close_people: sv.Detections,
    max_dist: float,
    max_person_top_below_ratio: float | None = None,
):
    if (
        close_helmets is None
        or close_people is None
        or len(close_helmets) == 0
        or len(close_people) == 0
    ):
        return []

    candidate_pairs = []
    for helmet_idx, helmet_bbox in enumerate(close_helmets.xyxy):
        helmet_top = bbox_top_center_xyxy(helmet_bbox)
        for person_idx, person_bbox in enumerate(close_people.xyxy):
            person_top = bbox_top_center_xyxy(person_bbox)
            if max_person_top_below_ratio is not None:
                person_h = person_bbox[3] - person_bbox[1]
                if person_h > 0 and person_top[1] > helmet_top[1] + max_person_top_below_ratio * person_h:
                    continue
            dx = helmet_top[0] - person_top[0]
            dy = helmet_top[1] - person_top[1]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist > max_dist and not point_in_bbox(helmet_top, person_bbox):
                continue
            candidate_pairs.append((dist, helmet_idx, person_idx))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    assigned_helmets = set()
    assigned_people = set()
    matches = []

    for dist, helmet_idx, person_idx in candidate_pairs:
        if helmet_idx in assigned_helmets or person_idx in assigned_people:
            continue
        assigned_helmets.add(helmet_idx)
        assigned_people.add(person_idx)
        matches.append((helmet_idx, person_idx, dist))

    return matches

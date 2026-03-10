from collections import defaultdict
from math import hypot


# Lap counting is kept separate from the tracker wrapper so it stays testable.
epsilon = 1e-6


def bbox_center(bbox):
    # Return the center point of an xyxy bounding box.
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def line_side_value(point, line):
    # Return which side of the directed line the point lies on.
    if line is None or len(line) != 2 or point is None:
        return None
    (ax, ay), (bx, by) = line
    px, py = point
    return ((bx - ax) * (py - ay)) - ((by - ay) * (px - ax))


def movement_distance(point_a, point_b):
    # Return the distance moved between consecutive track centers.
    if point_a is None or point_b is None:
        return 0.0
    return hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def _orientation(a, b, c):
    value = ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a, b, c):
    return (
        min(a[0], c[0]) - epsilon <= b[0] <= max(a[0], c[0]) + epsilon
        and min(a[1], c[1]) - epsilon <= b[1] <= max(a[1], c[1]) + epsilon
    )


def segments_intersect(p1, p2, q1, q2):
    # Return True when two line segments intersect.
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True

    return False


def should_count_lap_crossing(prev_center, curr_center, line, last_side, current_side, min_movement_px):
    # Count only real forward crossings and reject small jitter around the line.
    if line is None or len(line) != 2:
        return False
    if prev_center is None or curr_center is None:
        return False
    if last_side is None or current_side is None:
        return False
    if movement_distance(prev_center, curr_center) < float(min_movement_px):
        return False
    if last_side >= -epsilon or current_side < -epsilon:
        return False
    return segments_intersect(prev_center, curr_center, line[0], line[1])


class LapCounter:
    def __init__(self, frame_rate=30.0, finish_line=None, total_laps=None):
        # Counts are keyed by person tracker_id and kept only in memory.
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0
        self.finish_line = None
        self.total_laps = int(total_laps) if total_laps is not None else None
        self.person_lap_counts = defaultdict(int)
        self.person_cross_state = {}
        self.frame_index = 0
        self.lap_cooldown_frames = max(8, int(round(self.frame_rate * 0.3)))
        self.min_lap_movement_px = 8.0
        self.cross_state_ttl_frames = max(30, int(round(self.frame_rate * 5.0)))
        self.set_finish_line(finish_line)

    def set_finish_line(self, line):
        # Reset transient crossing state when the finish line changes.
        self.finish_line = [tuple(map(int, pt)) for pt in line] if line is not None and len(line) == 2 else None
        self.person_cross_state.clear()

    def set_total_laps(self, total_laps):
        # Store the configured race distance used by the lap panel.
        self.total_laps = int(total_laps) if total_laps is not None else None

    def reset(self):
        # Clear all in-memory lap counts and crossing history.
        self.person_lap_counts.clear()
        self.person_cross_state.clear()

    def get_lap_count(self, track_id):
        # Return the current lap count for one tracker ID.
        return int(self.person_lap_counts.get(int(track_id), 0))

    def get_active_lap_counts(self, people_tracks):
        # Build the row model consumed by the lap panel.
        rows = []
        for i, tid in enumerate(people_tracks.tracker_id):
            tid = int(tid)
            if tid == -1:
                continue
            rows.append(
                {
                    "track_id": tid,
                    "lap_count": self.get_lap_count(tid),
                    "predicted": bool(people_tracks.confidence[i] == 0),
                }
            )
        rows.sort(key=lambda row: row["track_id"])
        return rows

    def update(self, people_tracks):
        # Process one frame of tracked people and increment laps when needed.
        self.frame_index += 1
        if self.finish_line is None or len(people_tracks) == 0:
            self._cleanup_cross_state(set())
            return

        active_ids = set()

        for i, tid in enumerate(people_tracks.tracker_id):
            tid = int(tid)
            if tid == -1:
                continue

            active_ids.add(tid)
            center = bbox_center(people_tracks.xyxy[i])
            current_side = line_side_value(center, self.finish_line)

            state = self.person_cross_state.get(
                tid,
                {
                    "last_center": None,
                    "last_side": None,
                    "cooldown_until": -1,
                    "last_seen_frame": self.frame_index,
                },
            )

            state["last_seen_frame"] = self.frame_index

            # A crossing is accepted only when motion and direction both match.
            if self.frame_index >= state["cooldown_until"] and should_count_lap_crossing(
                state["last_center"],
                center,
                self.finish_line,
                state["last_side"],
                current_side,
                self.min_lap_movement_px,
            ):
                self.person_lap_counts[tid] += 1
                state["cooldown_until"] = self.frame_index + self.lap_cooldown_frames

            state["last_center"] = center
            state["last_side"] = current_side
            self.person_cross_state[tid] = state

        self._cleanup_cross_state(active_ids)

    def _cleanup_cross_state(self, active_ids):
        # Drop stale per-track motion state after tracks disappear.
        for tid in list(self.person_cross_state.keys()):
            state = self.person_cross_state[tid]
            if tid in active_ids:
                continue
            if self.frame_index - state["last_seen_frame"] > self.cross_state_ttl_frames:
                del self.person_cross_state[tid]

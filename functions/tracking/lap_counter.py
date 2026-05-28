from collections import defaultdict
from dataclasses import dataclass
from math import hypot


# Lap counting is kept separate from the tracker wrapper so it stays testable.
epsilon = 1e-6

def bbox_bottom_right(bbox):
    # Use the box bottom-right corner as the finish-line reference point.
    _x1, _y1, x2, y2 = bbox
    return (float(x2), float(y2))


def display_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))

def movement_distance(point_a, point_b):
    # Return the distance moved between consecutive track points.
    if point_a is None or point_b is None:
        return 0.0
    return hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def finish_line_offset(point, line):
    # Return how far left/right the point is from the finish line at the same y.
    if line is None or len(line) != 2 or point is None:
        return None
    (ax, ay), (bx, by) = line
    px, py = point
    dy = float(by) - float(ay)
    if abs(dy) <= epsilon:
        return None
    t = (float(py) - float(ay)) / dy
    line_x = float(ax) + (t * (float(bx) - float(ax)))
    return float(px) - line_x


def point_near_finish_line(point, line, band_px):
    # Return True when the point is close enough to the finish line to arm counting.
    offset = finish_line_offset(point, line)
    if offset is None:
        return False
    min_y = min(float(line[0][1]), float(line[1][1])) - float(band_px)
    max_y = max(float(line[0][1]), float(line[1][1])) + float(band_px)
    return min_y <= float(point[1]) <= max_y and abs(offset) <= float(band_px)

def should_count_lap_crossing(prev_point, curr_point, line, last_offset, current_offset, min_movement_px):
    # Count only real left-to-right crossings and reject small jitter around the line.
    if line is None or len(line) != 2:
        return False
    if prev_point is None or curr_point is None:
        return False
    if last_offset is None or current_offset is None:
        return False
    if movement_distance(prev_point, curr_point) < float(min_movement_px):
        return False
    if curr_point[0] <= (prev_point[0] + epsilon):
        return False
    return last_offset < -epsilon and current_offset >= 0.0


@dataclass
class CrossState:
    last_point: tuple[float, float] | None = None
    last_offset: float | None = None
    armed: bool = False
    cooldown_until: int = -1
    last_seen_frame: int = 0


class LapCounter:
    def __init__(self, frame_rate=30.0, finish_line=None, total_laps=None):
        # Counts are keyed by persistent helmet number when confirmed, otherwise
        # by a temporary ByteTrack track key.
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0
        self.finish_line = None
        self.total_laps = int(total_laps) if total_laps is not None else None
        self.person_lap_counts = defaultdict(int)
        self.person_cross_state = {}
        self.frame_index = 0
        self.lap_cooldown_frames = max(8, int(round(self.frame_rate * 0.3)))
        self.min_lap_movement_px = 8.0
        self.finish_line_band_px = 24.0
        self.cross_state_ttl_frames = max(30, int(round(self.frame_rate * 5.0)))
        self._ghost_confirm_timestamps: dict[str, int] = {}  # helmet_number -> last_seen_frame_index
        self._ghost_confirm_cooldown = max(3, int(round(self.frame_rate * 0.5)))  # seconds before re-registering
        # Persistent ghost entries: confirmed helmet numbers that have been lost.
        # These survive track drops and are never auto-removed.
        self._ghost_entries = {}  # helmet_number -> lap_count
        self.set_finish_line(finish_line)

    def set_finish_line(self, line):
        # Reset crossing state when the finish line geometry changes.
        # Lap counts and ghost entries survive — identified helmets are permanent.
        next_line = [tuple(map(int, pt)) for pt in line] if line is not None and len(line) == 2 else None
        line_changed = next_line != self.finish_line
        self.finish_line = next_line
        self.person_cross_state.clear()
        if line_changed:
            self.person_lap_counts.clear()

    def set_total_laps(self, total_laps):
        # Store the configured race distance used by the lap panel.
        self.total_laps = int(total_laps) if total_laps is not None else None

    def reset(self):
        # Clear all in-memory lap counts, crossing history, and ghost entries.
        self.person_lap_counts.clear()
        self.person_cross_state.clear()
        self._ghost_entries.clear()

    def _identity_key(self, track_id, helmet_number=-1):
        if helmet_number not in (-1, None, ""):
            return ("helmet", str(helmet_number))
        return ("track", int(track_id))

    def get_lap_count(self, track_id, helmet_number=-1):
        return int(self.person_lap_counts.get(self._identity_key(track_id, helmet_number), 0))

    def confirm_identity(self, track_id, helmet_number):
        temp_key = self._identity_key(track_id)
        helmet_key = self._identity_key(track_id, helmet_number)

        # Merge any existing lap counts so they are never lost.
        existing = self.person_lap_counts.get(helmet_key, 0)
        if temp_key in self.person_lap_counts:
            existing += int(self.person_lap_counts[temp_key])
        if existing:
            self.person_lap_counts[helmet_key] = existing
            # Remove stale temp key if it didn't contribute.
            if temp_key in self.person_lap_counts and self.person_lap_counts[temp_key] == 0:
                del self.person_lap_counts[temp_key]

        if temp_key in self.person_cross_state:
            self.person_cross_state[helmet_key] = self.person_cross_state.pop(temp_key)

    def register_lost_helmet(self, helmet_number, lap_count):
        """Register a confirmed helmet as a persistent ghost entry.
        Only updates if count is newer — avoids redundant writes each frame."""
        existing = self._ghost_entries.get(helmet_number, -1)
        if existing < lap_count:
            self._ghost_entries[helmet_number] = lap_count

    def get_active_lap_counts(self, people_tracks):
        rows = []

        # --- active tracks ---
        active_helmet_numbers = set()
        for i, tid in enumerate(people_tracks.tracker_id):
            tid = int(tid)
            if tid == -1:
                continue

            helmet_number = -1
            if "helmet_number" in people_tracks.data:
                helmet_number = people_tracks.data["helmet_number"][i]

            if helmet_number not in (-1, None, ""):
                display_id = helmet_number
                active_helmet_numbers.add(helmet_number)
                lap_count = self.get_lap_count(tid, helmet_number)
                # Register as ghost only if enough time has passed since last registration.
                self._maybe_register_ghost(helmet_number, lap_count)
            else:
                display_id = f"T{tid}"
                lap_count = self.get_lap_count(tid, helmet_number)

            rows.append(
                {
                    "track_id": tid,
                    "display_id": display_id,
                    "lap_count": lap_count,
                    "predicted": bool(people_tracks.confidence[i] == 0),
                    "confirmed": helmet_number not in (-1, None, ""),
                }
            )

        # --- ghost (lost) confirmed helmets ---
        for helmet_number, count in self._ghost_entries.items():
            if helmet_number in active_helmet_numbers:
                continue
            rows.append(
                {
                    "track_id": -1,
                    "display_id": helmet_number,
                    "lap_count": count,
                    "predicted": False,
                    "confirmed": True,
                }
            )

        rows.sort(key=lambda row: display_sort_key(row["display_id"]))
        return rows

    def update(self, people_tracks):
        self.frame_index += 1
        # Periodic cleanup of confirmed helmet tracking (once per ~5 seconds)
        if self.frame_index > 0 and self.frame_index % max(150, int(self.frame_rate * 5)) == 0:
            self._cleanup_aging_confirmed()
        if self.finish_line is None or len(people_tracks) == 0:
            self._cleanup_cross_state(set())
            return

        active_ids = set()

        for i, tid in enumerate(people_tracks.tracker_id):
            tid = int(tid)
            if tid == -1:
                continue

            helmet_number = -1
            if "helmet_number" in people_tracks.data:
                helmet_number = people_tracks.data["helmet_number"][i]

            identity_key = self._identity_key(tid, helmet_number)
            active_ids.add(identity_key)

            point = bbox_bottom_right(people_tracks.xyxy[i])
            current_offset = finish_line_offset(point, self.finish_line)
            near_line = point_near_finish_line(point, self.finish_line, self.finish_line_band_px)
            state = self.person_cross_state.get(identity_key, CrossState())
            state.last_seen_frame = self.frame_index

            if self.frame_index >= state.cooldown_until:
                if near_line and current_offset is not None and current_offset <= epsilon:
                    state.armed = True
                elif state.armed and not near_line:
                    state.armed = False

                if state.armed and should_count_lap_crossing(
                    state.last_point,
                    point,
                    self.finish_line,
                    state.last_offset,
                    current_offset,
                    self.min_lap_movement_px,
                ):
                    self.person_lap_counts[identity_key] += 1
                    state.armed = False
                    state.cooldown_until = self.frame_index + self.lap_cooldown_frames

            state.last_point = point
            state.last_offset = current_offset
            self.person_cross_state[identity_key] = state

        self._cleanup_cross_state(active_ids)

    def _cleanup_cross_state(self, active_ids):
        # Drop stale per-track motion state after tracks disappear.
        for identity_key in list(self.person_cross_state.keys()):
            state = self.person_cross_state[identity_key]
            if identity_key in active_ids:
                continue
            if self.frame_index - state.last_seen_frame > self.cross_state_ttl_frames:
                del self.person_cross_state[identity_key]

    def _maybe_register_ghost(self, helmet_number, lap_count):
        """Register as ghost only after cooldown has elapsed since last registration."""
        last_frame = self._ghost_confirm_timestamps.get(helmet_number, -self._ghost_confirm_cooldown * 2)
        if self.frame_index - last_frame >= self._ghost_confirm_cooldown * int(self.frame_rate):
            self.register_lost_helmet(helmet_number, lap_count)
            self._ghost_confirm_timestamps[helmet_number] = self.frame_index

    def _cleanup_aging_confirmed(self):
        """Prune cross_state entries for confirmed helmets that are no longer active but still tracked."""
        pass  # Currently cross_state is cleaned by _cleanup_cross_state; kept as hook for future use

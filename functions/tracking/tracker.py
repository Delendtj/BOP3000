from collections import defaultdict
from dataclasses import dataclass
from math import hypot

import numpy as np
import supervision as sv
from functions.tracking.lap_counter import LapCounter
from trackers import ByteTrackTracker


@dataclass
class CanonicalTrackMemory:
    last_bbox: tuple[float, float, float, float]
    last_center: tuple[float, float]
    velocity: tuple[float, float]
    last_seen_frame: int
    active: bool = True


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


class Tracker:
    def __init__(self, ocr_conf_threshold, conf_threshold, roi, frame_rate=30.0, finish_line=None, total_laps=None, accepted_numbers=None):
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0

        # Tuned ByteTrack settings for steadier helmet/person IDs.
        self.tracker_people = ByteTrackTracker(
            lost_track_buffer=120,
            frame_rate=self.frame_rate,
            track_activation_threshold=0.35,
            high_conf_det_threshold=0.45,
            minimum_iou_threshold=0.15,
            minimum_consecutive_frames=2,
        )
        ###########################
        # [SUBJECT TOP BE REMOVED]#
        ###########################
        # I don't think we need trackers on helmets anymore?
        # Because we use numbers on the helmets as ids, and there is no benefit for tracks across frames.
        self.tracker_helmet = ByteTrackTracker(
            lost_track_buffer=90,
            frame_rate=self.frame_rate,
            #track_activation_threshold=0.35,
            high_conf_det_threshold=0.45,
            minimum_iou_threshold=0.08,
            minimum_consecutive_frames=2,
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.people_tracks = sv.Detections.empty()

        self.processed_tracker_ids = set()
        self.people_hnum_final = {}

        self.person_class_id = 1
        self.helmet_class_id = 0

        self.accepted_numbers = accepted_numbers
        self.ocr_conf_threshold = ocr_conf_threshold

        self.person_frame_index = 0
        self.next_person_canonical_id = 1
        self.active_raw_to_canonical = {}
        self.canonical_person_memories = {}
        self.reuse_max_missing_frames = max(45, int(round(self.frame_rate * 3.0)))
        self.reuse_max_distance_px = 180.0
        self.reuse_strict_distance_px = 100.0
        self.reuse_min_iou = 0.05

        self.conf_threshold = conf_threshold
        self.roi = roi

        self.lap_counter = LapCounter(frame_rate=self.frame_rate, finish_line=finish_line, total_laps=total_laps)

    @property
    def total_laps(self):
        return self.lap_counter.total_laps

    def set_roi(self, roi):
        self.roi = roi

    def set_finish_line(self, line):
        self.lap_counter.set_finish_line(line)

    def set_total_laps(self, total_laps):
        self.lap_counter.set_total_laps(total_laps)

    def reset_lap_counts(self):
        self.lap_counter.reset()

    def is_person_confirmed(self, track_id: int) -> bool:
        return int(track_id) in self.people_hnum_final

    def get_lap_count(self, track_id):
        return self.lap_counter.get_lap_count(track_id)

    def get_active_lap_counts(self):
        return self.lap_counter.get_active_lap_counts(self.people_tracks)

    def track_detection(self, detections: sv.Detections):
        self.person_frame_index += 1
        people_detections = detections[detections.class_id == self.person_class_id]

        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]

        raw_people_tracks = self.tracker_people.update(people_for_tracker)
        self.people_tracks = self._apply_person_reuse(raw_people_tracks)

        # Fill data attribute into the people_tracks.data (Detections obj)
        # with a "helmet_number" dict key to store helmet numbers
        if "helmet_number" not in self.people_tracks.data:
            self.people_tracks.data["helmet_number"] = np.full(len(self.people_tracks), -1, dtype=object)

        # Actually add them into the tracks
        # Init as -1
        self.people_tracks.data["helmet_number"] = np.array(
            [self.people_hnum_final.get(tid, -1) for tid in self.people_tracks.tracker_id],
            dtype=object,
        )

    def _apply_person_reuse(self, raw_people_tracks: sv.Detections):
        raw_tracker_ids = np.asarray(raw_people_tracks.tracker_id, dtype=int) if len(raw_people_tracks) > 0 else np.empty(0, dtype=int)
        self._mark_missing_people_inactive(raw_tracker_ids)
        assignments = {
            int(raw_id): int(self.active_raw_to_canonical[raw_id])
            for raw_id in raw_tracker_ids
            if int(raw_id) in self.active_raw_to_canonical and int(raw_id) != -1
        }

        new_people = []
        for idx, raw_id in enumerate(raw_tracker_ids):
            raw_id = int(raw_id)
            if raw_id == -1 or raw_id in assignments:
                continue
            bbox = tuple(float(v) for v in raw_people_tracks.xyxy[idx])
            new_people.append(
                {
                    "raw_id": raw_id,
                    "bbox": bbox,
                    "center": bbox_center(bbox),
                }
            )

        assignments.update(self._match_new_people_to_lost_ids(new_people))

        for person in new_people:
            raw_id = person["raw_id"]
            if raw_id in assignments:
                continue
            canonical_id = self._allocate_person_canonical_id()
            assignments[raw_id] = canonical_id
            self.active_raw_to_canonical[raw_id] = canonical_id

        canonical_ids = np.full(len(raw_people_tracks), -1, dtype=int)
        for idx, raw_id in enumerate(raw_tracker_ids):
            raw_id = int(raw_id)
            if raw_id == -1:
                continue
            canonical_id = int(assignments[raw_id])
            canonical_ids[idx] = canonical_id
            self.active_raw_to_canonical[raw_id] = canonical_id
            bbox = tuple(float(v) for v in raw_people_tracks.xyxy[idx])
            self._update_person_memory(canonical_id, bbox)

        raw_people_tracks.tracker_id = canonical_ids
        raw_people_tracks.data["raw_tracker_id"] = raw_tracker_ids.copy()
        self._prune_stale_person_memories()
        return raw_people_tracks

    def _mark_missing_people_inactive(self, current_raw_tracker_ids):
        current_ids = {int(raw_id) for raw_id in current_raw_tracker_ids if int(raw_id) != -1}
        for raw_id, canonical_id in list(self.active_raw_to_canonical.items()):
            if raw_id in current_ids:
                continue
            memory = self.canonical_person_memories.get(int(canonical_id))
            if memory is not None:
                memory.active = False
            del self.active_raw_to_canonical[raw_id]

    def _match_new_people_to_lost_ids(self, new_people):
        lost_ids = [
            canonical_id
            for canonical_id, memory in self.canonical_person_memories.items()
            if not memory.active and (self.person_frame_index - memory.last_seen_frame) <= self.reuse_max_missing_frames
        ]
        candidates = []
        for person in new_people:
            for canonical_id in lost_ids:
                distance_px, iou = self._reuse_candidate_metrics(canonical_id, person["bbox"], person["center"])
                if distance_px is None:
                    continue
                candidates.append((distance_px, person["raw_id"], canonical_id))

        assignments = {}
        matched_raw_ids = set()
        matched_canonical_ids = set()
        for distance_px, raw_id, canonical_id in sorted(candidates, key=lambda item: item[0]):
            if raw_id in matched_raw_ids or canonical_id in matched_canonical_ids:
                continue
            assignments[raw_id] = canonical_id
            matched_raw_ids.add(raw_id)
            matched_canonical_ids.add(canonical_id)
            self.active_raw_to_canonical[raw_id] = canonical_id
            self.canonical_person_memories[canonical_id].active = True
            print(f"Reused person ID {canonical_id} for new raw track {raw_id} (distance={distance_px:.1f}px)")
        return assignments

    def _reuse_candidate_metrics(self, canonical_id, bbox, center):
        memory = self.canonical_person_memories.get(int(canonical_id))
        if memory is None:
            return None, None
        gap_frames = max(1, self.person_frame_index - memory.last_seen_frame)
        predicted_center = (
            memory.last_center[0] + (memory.velocity[0] * gap_frames),
            memory.last_center[1] + (memory.velocity[1] * gap_frames),
        )
        distance_px = hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
        if distance_px > self.reuse_max_distance_px:
            return None, None
        iou = bbox_iou(memory.last_bbox, bbox)
        if iou < self.reuse_min_iou and distance_px > self.reuse_strict_distance_px:
            return None, None
        return distance_px, iou

    def _allocate_person_canonical_id(self):
        canonical_id = int(self.next_person_canonical_id)
        self.next_person_canonical_id += 1
        return canonical_id

    def _update_person_memory(self, canonical_id, bbox):
        center = bbox_center(bbox)
        existing = self.canonical_person_memories.get(int(canonical_id))
        if existing is None:
            self.canonical_person_memories[int(canonical_id)] = CanonicalTrackMemory(
                last_bbox=bbox,
                last_center=center,
                velocity=(0.0, 0.0),
                last_seen_frame=self.person_frame_index,
                active=True,
            )
            return

        frame_delta = max(1, self.person_frame_index - existing.last_seen_frame)
        existing.velocity = (
            (center[0] - existing.last_center[0]) / float(frame_delta),
            (center[1] - existing.last_center[1]) / float(frame_delta),
        )
        existing.last_bbox = bbox
        existing.last_center = center
        existing.last_seen_frame = self.person_frame_index
        existing.active = True

    # Im not sure what this even does...
    def _prune_stale_person_memories(self):
        for canonical_id, memory in list(self.canonical_person_memories.items()):
            if memory.active:
                continue
            if (self.person_frame_index - memory.last_seen_frame) > self.reuse_max_missing_frames:
                del self.canonical_person_memories[canonical_id]

    def update_lap_counts(self):
        self.lap_counter.update(self.people_tracks)

    def get_non_confirmed_people_tracks(self) -> sv.Detections:
        """
        Returns part of self.people_tracks that is non confirmed.
        """
        # Filter to get only confirmed IDs
        confirmed_ids = np.array(list(self.people_hnum_final.keys()), dtype=int)

        if confirmed_ids.size == 0:
            return self.people_tracks

        # Mask of nonconfirmed
        keep = np.isin(self.people_tracks.tracker_id, confirmed_ids, invert=True)
        return self.people_tracks[keep]

    def assign_helmet_numbers_to_people(self, helmets):
        """
        Assign helmet numbers to people_tracks.data["helmet_number"] using one-shot OCR.
        Accepts a result immediately if:
            - The track is not yet confirmed
            - The number matches an accepted number (via _match_partial)
            - The OCR confidence meets the threshold
        """
        non_confirmed_people = self.get_non_confirmed_people_tracks()

        if (helmets is None
                or len(helmets) == 0
                or self.roi is None
                or len(non_confirmed_people) == 0):
            return

        # Get a set of nonconfirmed TIDs
        allowed_ids = set(non_confirmed_people.tracker_id.tolist())

        # Loop through each OCR result and assign immediately if confidence is sufficient
        for h in helmets:
            tid = h["track_id"]
            number = h["helmet_number"]
            ocr_conf = h.get("ocr_conf", 0.0)

            # Skip ids that are already confirmed or not in active tracks
            if tid not in allowed_ids:
                continue

            # Skip tracks without a valid tracker ID
            if tid == -1:
                continue

            # If track is already confirmed, skip entirely
            if tid in self.people_hnum_final:
                continue

            # Check confidence threshold
            if ocr_conf < self.ocr_conf_threshold:
                print(f"[OCR] track_id={tid}: conf={ocr_conf:.1f}% below threshold ({self.ocr_conf_threshold}%) — rejected")
                continue

            # Match against accepted helmet numbers (from CSV)
            matched = self._match_partial(number)

            if matched == "" or matched is None:
                print(f"[OCR] track_id={tid}: number='{number}' didn't match accepted numbers — rejected")
                continue

            # Check for duplicate assignment (uniksjeck)
            if matched in self.people_hnum_final.values():
                print(f"[OCR] track_id={tid}: #{matched} already assigned to another skater — rejected")
                continue

            # Assign the number — one-shot confirmation!
            mask = self.people_tracks.tracker_id == tid
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                self.people_tracks.data["helmet_number"][idxs[0]] = matched

            self.people_hnum_final[tid] = matched
            print(f"[OCR CONFIRMED] track_id={tid} -> #{matched} (conf={ocr_conf:.1f}%)")


    def annotate(self, frame):
        annotated = self.box_annotator.annotate(frame, self.people_tracks)
        # annotated = self.box_annotator.annotate(annotated, self.helmet_tracks)

        if len(self.people_tracks) > 0:
            labels = []
            for i, tid in enumerate(self.people_tracks.tracker_id):
                helmet_number = -1
                if "helmet_number" in self.people_tracks.data:
                    helmet_number = self.people_tracks.data["helmet_number"][i]
                display_id = helmet_number if helmet_number not in (-1, None, "") else tid
                if self.people_tracks.confidence[i] == 0:
                    labels.append(f"ID {display_id} (pred)")
                else:
                    labels.append(f"ID {display_id}")
            annotated = self.label_annotator.annotate(annotated, self.people_tracks, labels=labels)

        # Old code for annotating helmets.
        # I leave it here in case we need to do this again.
        # if len(self.helmet_tracks) > 0:
        #     labels = []
        #     for i, tid in enumerate(self.helmet_tracks.tracker_id):
        #         labels.append(f"ID {tid}, Number: {self.helmet_tracks.data['helmet_number'][i]}")
        #     annotated = self.label_annotator.annotate(annotated, self.helmet_tracks, labels=labels)

        return annotated

    def _match_partial(self, partial):
        print("[TRACKER._match_partial] Input from OCR = ", repr(partial))
        # Needs to be longer than 2
        if len(partial) < 2:
            print(f"[TRACKER._match_partial] Rejected: length={len(partial)} < 2")
            return None
        if self.accepted_numbers is None:
            print(f"[TRACKER._match_partial] No accepted_numbers, returning partial as-is: {repr(partial)}")
            return partial

        # Loops over accepted numbers and keep those where partial is a substring of n.
        candidates = [n for n in self.accepted_numbers if partial in n]
        print(f"[TRACKER._match_partial] partial={repr(partial)}, candidates={candidates}")
        # If the candidate is only one then we return it. Else it's too ambiguous.
        result = candidates[0] if len(candidates) == 1 else None
        print(f"[TRACKER._match_partial] Result: {repr(result)}")
        return result

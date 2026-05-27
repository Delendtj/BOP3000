import copy
import numpy as np
import supervision as sv

from functions.tracking.lap_counter import LapCounter
from trackers import ByteTrackTracker


class Tracker:
    def __init__(self, ocr_conf_threshold, conf_threshold, roi, frame_rate=30.0, finish_line=None, total_laps=None, accepted_numbers=None):
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0

        self.tracker_people = ByteTrackTracker(
            lost_track_buffer=120,
            frame_rate=self.frame_rate,
            track_activation_threshold=0.35,
            high_conf_det_threshold=0.45,
            minimum_iou_threshold=0.15,
            minimum_consecutive_frames=2,
        )
        self.tracker_helmet = ByteTrackTracker(
            lost_track_buffer=90,
            frame_rate=self.frame_rate,
            high_conf_det_threshold=0.45,
            minimum_iou_threshold=0.08,
            minimum_consecutive_frames=2,
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.people_tracks = sv.Detections.empty()

        self.processed_tracker_ids = set()

        self.person_class_id = 1
        self.helmet_class_id = 0

        self.accepted_numbers = accepted_numbers
        self.ocr_conf_threshold = ocr_conf_threshold
        self.conf_threshold = conf_threshold
        self.roi = roi

        # Active short-term association: ByteTrack raw ID -> confirmed helmet number.
        self.active_track_helmet_numbers = {}
        self.helmet_to_active_track = {}

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
        return int(track_id) in self.active_track_helmet_numbers

    def get_lap_count(self, track_id):
        helmet_number = self.active_track_helmet_numbers.get(int(track_id), -1)
        return self.lap_counter.get_lap_count(track_id, helmet_number)

    def get_active_lap_counts(self):
        return self.lap_counter.get_active_lap_counts(self.people_tracks)

    def track_helmet_detections(self, close_helmets: sv.Detections) -> sv.Detections:
        """
        Run ByteTrack on helmet detections to assign persistent track IDs.

        Returns the tracked detections (same structure as input, but with
        ``tracker_id`` column filled by ByteTrack).  This is necessary so
        that helmet crops carry a stable identity across frames, which in
        turn enables OCR voting / consensus logic later.

        Parameters
        ----------
        close_helmets : sv.Detections
            Raw helmet detections from the close-camera YOLO inference.

        Returns
        -------
        sv.Detections
            Helmet detections with persistent ``tracker_id`` values.
        """
        if close_helmets is None or len(close_helmets) == 0:
            return sv.Detections.empty()

        helmet_detections = close_helmets  # already filtered to class 0 by YOLO
        helmet_for_tracker = helmet_detections[helmet_detections.confidence > self.conf_threshold]

        tracked = self.tracker_helmet.update(helmet_for_tracker)

        # Merge tracker_id back into original detections
        # (untracked helmets keep tracker_id == -1)
        if len(tracked) == 0:
            out = copy.deepcopy(close_helmets)
            out.data["tracker_id"] = np.full(len(out), -1, dtype=int)
            return out

        # Create a full copy with tracker_id initialized to -1
        out = copy.deepcopy(close_helmets)
        out.data["tracker_id"] = np.full(len(close_helmets), -1, dtype=int)

        # Map tracked detections back to original detections by bounding-box IoU.
        # ByteTrack.update() returns a Detections object that may reorder
        # detections and no longer carries the original indices, so we match
        # by IoU (threshold 0.1 is generous for helmet-sized boxes).
        iou_threshold = 0.1
        # sv.box_iou_batch expects (N, 4) arrays, not Detections objects
        iou_matrix = sv.box_iou_batch(tracked.xyxy, close_helmets.xyxy)  # shape (n_tracked, n_original)
        for i in range(len(tracked)):
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] >= iou_threshold:
                out.data["tracker_id"][best_j] = int(tracked.tracker_id[i])

        return out

    def track_detection(self, detections: sv.Detections):
        people_detections = detections[detections.class_id == self.person_class_id]
        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]

        self.people_tracks = self.tracker_people.update(people_for_tracker)
        raw_tracker_ids = (
            np.asarray(self.people_tracks.tracker_id, dtype=int)
            if len(self.people_tracks) > 0 else np.empty(0, dtype=int)
        )

        self.people_tracks.data["raw_tracker_id"] = raw_tracker_ids.copy()
        self._drop_inactive_confirmed_tracks(raw_tracker_ids)

        self.people_tracks.data["helmet_number"] = np.array(
            [self.active_track_helmet_numbers.get(int(tid), -1) for tid in raw_tracker_ids],
            dtype=object,
        )

    def _drop_inactive_confirmed_tracks(self, current_raw_tracker_ids):
        active_ids = {int(raw_id) for raw_id in current_raw_tracker_ids if int(raw_id) != -1}
        for raw_id, helmet_number in list(self.active_track_helmet_numbers.items()):
            if int(raw_id) in active_ids:
                continue
            del self.active_track_helmet_numbers[raw_id]
            if self.helmet_to_active_track.get(helmet_number) == raw_id:
                del self.helmet_to_active_track[helmet_number]

    def update_lap_counts(self):
        self.lap_counter.update(self.people_tracks)

    def get_non_confirmed_people_tracks(self) -> sv.Detections:
        """
        Returns part of self.people_tracks that is non confirmed.
        """
        confirmed_ids = np.array(list(self.active_track_helmet_numbers.keys()), dtype=int)

        if confirmed_ids.size == 0:
            return self.people_tracks

        keep = np.isin(self.people_tracks.tracker_id, confirmed_ids, invert=True)
        return self.people_tracks[keep]

    def assign_helmet_numbers_to_people(self, helmets):
        """
        Confirm racers by OCR. Raw ByteTrack IDs are temporary only; once OCR matches
        an accepted helmet number, persistent racer state is keyed by that number.
        """
        non_confirmed_people = self.get_non_confirmed_people_tracks()

        if (helmets is None
                or len(helmets) == 0
                or self.roi is None
                or len(non_confirmed_people) == 0):
            return

        allowed_ids = set(int(tid) for tid in non_confirmed_people.tracker_id.tolist())

        for h in helmets:
            tid = int(h["track_id"])
            number = h["helmet_number"]
            ocr_conf = h.get("ocr_conf", 0.0)

            if tid not in allowed_ids or tid == -1:
                continue

            if tid in self.active_track_helmet_numbers:
                continue

            if ocr_conf < self.ocr_conf_threshold:
                print(f"[OCR] track_id={tid}: conf={ocr_conf:.1f}% below threshold ({self.ocr_conf_threshold}%) - rejected")
                continue

            matched = self._match_partial(number)
            if matched in ("", None):
                print(f"[OCR] track_id={tid}: number='{number}' didn't match accepted numbers - rejected")
                continue

            bound_track_id = self.helmet_to_active_track.get(matched)
            if bound_track_id is not None and int(bound_track_id) != tid:
                print(f"[OCR] track_id={tid}: #{matched} already bound to active track {bound_track_id} - rejected")
                continue

            mask = self.people_tracks.tracker_id == tid
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                self.people_tracks.data["helmet_number"][idxs[0]] = matched

            self.active_track_helmet_numbers[tid] = matched
            self.helmet_to_active_track[matched] = tid
            self.lap_counter.confirm_identity(tid, matched)
            print(f"[OCR CONFIRMED] track_id={tid} -> #{matched} (conf={ocr_conf:.1f}%)")

    def annotate(self, frame):
        annotated = self.box_annotator.annotate(frame, self.people_tracks)

        if len(self.people_tracks) > 0:
            labels = []
            for i, tid in enumerate(self.people_tracks.tracker_id):
                tid = int(tid)
                helmet_number = -1
                if "helmet_number" in self.people_tracks.data:
                    helmet_number = self.people_tracks.data["helmet_number"][i]

                if helmet_number not in (-1, None, ""):
                    display_id = helmet_number
                elif tid != -1:
                    display_id = f"T{tid}"
                else:
                    display_id = "?"

                if self.people_tracks.confidence[i] == 0:
                    labels.append(f"ID {display_id} (pred)")
                else:
                    labels.append(f"ID {display_id}")

            annotated = self.label_annotator.annotate(annotated, self.people_tracks, labels=labels)

        return annotated

    def _match_partial(self, partial):
        print("[TRACKER._match_partial] Input from OCR = ", repr(partial))
        if len(partial) < 2:
            print(f"[TRACKER._match_partial] Rejected: length={len(partial)} < 2")
            return None
        if self.accepted_numbers is None:
            print(f"[TRACKER._match_partial] No accepted_numbers, returning partial as-is: {repr(partial)}")
            return partial

        candidates = [n for n in self.accepted_numbers if partial in n]
        print(f"[TRACKER._match_partial] partial={repr(partial)}, candidates={candidates}")
        result = candidates[0] if len(candidates) == 1 else None
        print(f"[TRACKER._match_partial] Result: {repr(result)}")
        return result

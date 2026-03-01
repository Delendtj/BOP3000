from collections import Counter, defaultdict

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from functions.BBExtractor import extract_helmet_box
from functions.register_helmet import register_helmet
from functions.roi import bbox_center_in_roi


class Tracker:
    def __init__(self, ocr_frames, conf_threshold, roi, frame_rate=30.0):
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0

        # Tuned ByteTrack settings for steadier helmet/person IDs.
        self.tracker_people = ByteTrackTracker(
            lost_track_buffer=90,
            frame_rate=self.frame_rate,
        )
        self.tracker_helmet = ByteTrackTracker(
            lost_track_buffer=90,
            frame_rate=self.frame_rate,
            track_activation_threshold=0.35,
            high_conf_det_threshold=0.45,
            minimum_iou_threshold=0.08,
            minimum_consecutive_frames=2,
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.people_tracks = sv.Detections.empty()
        self.helmet_tracks = sv.Detections.empty()

        self.processed_tracker_ids = set()
        self.ocr_runs = defaultdict(lambda: {"runs": 0, "cooldown": 0, "votes": []})
        self.helmet_numbers_final = {}

        self.PERSON_CLASS_ID = 1
        self.HELMET_CLASS_ID = 0
        self.RUNS_BEFORE_RETRY = 2
        self.RUNS_COOLDOWN = 10

        self.ocr_frames = ocr_frames
        self.conf_threshold = conf_threshold
        self.roi = roi

    def set_roi(self, roi):
        self.roi = roi

    def track_detection(self, detections: sv.Detections, frame):
        people_detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == self.HELMET_CLASS_ID]

        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]
        helmets_for_tracker = helmet_detections[helmet_detections.confidence > self.conf_threshold]

        self.people_tracks = self.tracker_people.update(people_for_tracker)
        self.helmet_tracks = self.tracker_helmet.update(helmets_for_tracker)

        if "helmet_number" not in self.helmet_tracks.data:
            self.helmet_tracks.data["helmet_number"] = np.full(len(self.helmet_tracks), -1, dtype=object)

        self.helmet_tracks.data["helmet_number"] = np.array(
            [self.helmet_numbers_final.get(tid, -1) for tid in self.helmet_tracks.tracker_id],
            dtype=object,
        )

        non_confirmed_helmets = self.helmet_tracks[
            np.isin(self.helmet_tracks.tracker_id, list(self.helmet_numbers_final.keys()), invert=True)
        ]

        self.check_for_ocr(non_confirmed_helmets, frame)

    def check_for_ocr(self, non_confirmed_helmets: sv.Detections, frame):
        if frame is None or self.roi is None or len(non_confirmed_helmets) == 0:
            return

        det_full = np.column_stack(
            [
                non_confirmed_helmets.xyxy,
                non_confirmed_helmets.confidence,
                non_confirmed_helmets.class_id,
                non_confirmed_helmets.tracker_id,
            ]
        )

        helmets = extract_helmet_box(det_full, frame)
        helmets = [h for h in helmets if bbox_center_in_roi(h["bbox"], self.roi)]
        if len(helmets) == 0:
            return

        helmet_results = register_helmet(helmets, debug=True)

        for h in helmet_results:
            tid = h["track_id"]
            number = h["helmet_number"]

            # Skip pre-confirmation tracks from ByteTrack.
            if tid == -1:
                continue

            state = self.ocr_runs[tid]

            if state["runs"] >= self.RUNS_BEFORE_RETRY:
                state["cooldown"] += 1
                if state["cooldown"] >= self.RUNS_COOLDOWN:
                    state["runs"] = 0
                    state["cooldown"] = 0
                    state["votes"].clear()
                continue

            state["runs"] += 1

            if number != "":
                state["votes"].append(number)

            print("tid:", tid, "votes:", state["votes"])
            if len(state["votes"]) >= self.ocr_frames and tid not in self.helmet_numbers_final:
                final_number = Counter(state["votes"]).most_common(1)[0][0]
                self.helmet_numbers_final[tid] = final_number
                print(f"Tracker {tid} final helmet number: {final_number}")

                mask = self.helmet_tracks.tracker_id == tid
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    self.helmet_tracks.data["helmet_number"][idxs[0]] = final_number

                self.processed_tracker_ids.add(tid)

    def annotate(self, frame):
        annotated = self.box_annotator.annotate(frame, self.people_tracks)
        annotated = self.box_annotator.annotate(annotated, self.helmet_tracks)

        if len(self.people_tracks) > 0:
            labels = []
            for i, tid in enumerate(self.people_tracks.tracker_id):
                if self.people_tracks.confidence[i] == 0:
                    labels.append(f"ID {tid} (pred)")
                else:
                    labels.append(f"ID {tid}")
            annotated = self.label_annotator.annotate(annotated, self.people_tracks, labels=labels)

        if len(self.helmet_tracks) > 0:
            labels = []
            for i, tid in enumerate(self.helmet_tracks.tracker_id):
                labels.append(f"ID {tid}, Number: {self.helmet_tracks.data['helmet_number'][i]}")
            annotated = self.label_annotator.annotate(annotated, self.helmet_tracks, labels=labels)

        return annotated
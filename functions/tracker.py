from collections import Counter, defaultdict

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class Tracker:
    def __init__(self, ocr_votes_count, conf_threshold, roi, frame_rate=30.0):
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0

        # Tuned ByteTrack settings for steadier helmet/person IDs.
        self.tracker_people = ByteTrackTracker(
            lost_track_buffer=90,
            frame_rate=self.frame_rate,
        )
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
        self.helmet_tracks = sv.Detections.empty()

        self.ocr_runs = defaultdict(lambda: {"runs": 0, "cooldown": 0, "votes": []})
        self.helmet_numbers_final = {}

        self.PERSON_CLASS_ID = 1
        self.HELMET_CLASS_ID = 0

        # Runs we wait before we wait RUNS_COOLDOWN frames before we try again on a given TID
        self.RUNS_BEFORE_RETRY = ocr_votes_count + 1
        self.RUNS_COOLDOWN = 5

        self.ocr_votes_count = ocr_votes_count
        self.conf_threshold = conf_threshold
        self.roi = roi

    def set_roi(self, roi):
        self.roi = roi

    def track_detection(self, detections: sv.Detections):


        people_detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == self.HELMET_CLASS_ID]

        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]
        helmets_for_tracker = helmet_detections[helmet_detections.confidence > self.conf_threshold]

        self.people_tracks = self.tracker_people.update(people_for_tracker)
        self.helmet_tracks = self.tracker_helmet.update(helmets_for_tracker)

        # Fill data attribute with a "helmet_number" dict key to store helmet numbers
        if "helmet_number" not in self.helmet_tracks.data:
            self.helmet_tracks.data["helmet_number"] = np.full(len(self.helmet_tracks), -1, dtype=object)

        # Actually add them into the tracks
        self.helmet_tracks.data["helmet_number"] = np.array(
            [self.helmet_numbers_final.get(tid, -1) for tid in self.helmet_tracks.tracker_id],
            dtype=object,
        )

    def get_non_confirmed_helmet_tracks(self):
        confirmed_ids = np.array(list(self.helmet_numbers_final.keys()), dtype=int)

        if confirmed_ids.size == 0:
            return self.helmet_tracks

        keep = np.isin(self.helmet_tracks.tracker_id, confirmed_ids, invert=True)
        return self.helmet_tracks[keep]

    def check_for_ocr(self, helmets):

        non_confirmed_helmets = self.helmet_tracks[
            np.isin(self.helmet_tracks.tracker_id, list(self.helmet_numbers_final.keys()), invert=True)
        ]

        if helmets is None or self.roi is None or len(non_confirmed_helmets) == 0:
            return

        if len(helmets) == 0:
            return

        #helmet_results = register_helmet(helmets, debug=True)

        allowed_ids = set(non_confirmed_helmets.tracker_id.tolist())

        for h in helmets:
            tid = h["track_id"]
            number = h["helmet_number"]

            # Skip ids that are already confirmed
            if tid not in allowed_ids:
                continue

            # Skip tracks that isn't confirmed by ByteTrack yet.
            if tid == -1:
                continue


            state = self.ocr_runs[tid]

            if state["runs"] >= self.RUNS_BEFORE_RETRY:
                state["cooldown"] += 1
                if state["cooldown"] >= self.RUNS_COOLDOWN:
                    state["runs"] = 0
                    state["cooldown"] = 0
                continue

            state["runs"] += 1

            if number != "":
                state["votes"].append(number)

            print("tid:", tid, "votes:", state["votes"])
            # Given enough votes and tid not being a confirmed helmet
            if len(state["votes"]) >= self.ocr_votes_count and tid not in self.helmet_numbers_final:
                final_number = Counter(state["votes"]).most_common(1)[0][0]
                self.helmet_numbers_final[tid] = final_number
                print(f"Tracker {tid} final helmet number: {final_number}")

                mask = self.helmet_tracks.tracker_id == tid
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    self.helmet_tracks.data["helmet_number"][idxs[0]] = final_number


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

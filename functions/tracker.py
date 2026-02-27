import supervision as sv
import numpy as np
from trackers import ByteTrackTracker
from collections import defaultdict, Counter

from functions.BBExtractor import extract_helmet_box
from functions.register_helmet import register_helmet
from functions.roi import bbox_center_in_roi

class Tracker:
    def __init__(self, ocr_frames, conf_threshold, roi):

        # BoxAnnotator draws the bounding boxes, LabelAnnotator draws the track ID.
        self.tracker_people = ByteTrackTracker(lost_track_buffer=150)
        self.tracker_helmet = ByteTrackTracker(lost_track_buffer=150)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.people_tracks = sv.Detections.empty()
        self.helmet_tracks = sv.Detections.empty()

        # Keeps track of tracks that have confirmed helmet numbers
        self.processed_tracker_ids = set()
        # Keeps track on OCR cooldown and retries logic
        self.ocr_runs = defaultdict(lambda: {
            "runs": 0,
            "cooldown": 0,
            "votes": []
        })

        # self.ocr_votes = defaultdict(list)  # {tracker_id: [list of ocr strings]}
        self.helmet_numbers_final = {}  # {tracker_id: final_number}

        # Consts
        self.PERSON_CLASS_ID = 1
        self.HELMET_CLASS_ID = 0
        self.RUNS_BEFORE_RETRY = 3

        # Param Options
        self.ocr_frames = ocr_frames  # collect votes for N frames before deciding
        self.conf_threshold = conf_threshold

        self.roi = roi

    def track_detection(self,detections: sv.Detections, frame):
        # Separate them
        people_detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == self.HELMET_CLASS_ID]

        # Filter out low confidence detections (0.3)
        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]
        helmets_for_tracker = helmet_detections[helmet_detections.confidence > self.conf_threshold]

        # Tracking
        self.people_tracks = self.tracker_people.update(people_for_tracker)
        self.helmet_tracks = self.tracker_helmet.update(helmets_for_tracker)

        # Add "helmet_number" to helmet tracks if they do not yet exist
        # data is a dict
        if "helmet_number" not in self.helmet_tracks.data:
            self.helmet_tracks.data["helmet_number"] = np.full(len(self.helmet_tracks), -1, dtype=object)

        # Add the confirmed/accepted helmet numbers into the tracks
        self.helmet_tracks.data["helmet_number"] = np.array(
            [self.helmet_numbers_final.get(tid, -1) for tid in self.helmet_tracks.tracker_id], dtype=object)

        # final numbers for a given track_id is stored in helmet_numbers_final
        # We filter these out of the tracks we are working on.
        non_confirmed_helmets = self.helmet_tracks[
            np.isin(self.helmet_tracks.tracker_id, list(self.helmet_numbers_final.keys()), invert=True)
        ]

        self.check_for_ocr(non_confirmed_helmets, frame)

    def check_for_ocr(self, non_confirmed_helmets: sv.Detections, frame):
        if len(self.helmet_tracks) > 0 and self.roi is not None:
            #Format for BBExtractor
            det_full = np.column_stack([
                non_confirmed_helmets.xyxy,
                non_confirmed_helmets.confidence,
                non_confirmed_helmets.class_id,
                non_confirmed_helmets.tracker_id,
            ])

            # Extracts the bbox for the helmet
            helmets = extract_helmet_box(det_full, frame)

            if self.roi is not None:
                helmets = [h for h in helmets if bbox_center_in_roi(h["bbox"], self.roi)]

            if len(helmets) > 0:

                # [RUNS THE OCR]
                # Gets the OCR result for helmet number based on extracted bbox
                helmet_results = register_helmet(helmets, debug=True)

                for h in helmet_results:
                    tid = h['track_id']
                    number = h['helmet_number']

                    # Skip if we have done three consecutive runs on a track
                    if self.ocr_runs[tid]['runs'] >= self.RUNS_BEFORE_RETRY:
                        self.ocr_runs[tid]['cooldown'] += 1

                        # Reset if cooldown reached
                        if self.ocr_runs[tid]['cooldown'] >= self.RUNS_COOLDOWN:
                            self.ocr_runs[tid]['runs'] = 0
                        continue

                    # ocr_votes is a list of OCR results (helmet_number) for a given tracker_id
                    if number != "":  # only count non-empty results
                        self.ocr_runs[tid]['votes'].append(number)

                    # Once we have enough votes, pick the winner
                    # Currently this makes it so that when a number is set
                    # it is set forever for that tracker id
                    print("tid: ", tid, " votes: ", self.ocr_runs[tid]['votes'])
                    if len(self.ocr_runs[tid]['votes']) >= self.ocr_frames and tid not in self.helmet_numbers_final:
                        final_number = Counter(self.ocr_runs[tid]['votes']).most_common(1)[0][0]
                        # Idk if this is actually redundant, because we already add this into helmet_tracks at the end
                        self.helmet_numbers_final[tid] = final_number
                        print(f"Tracker {tid} final helmet number: {final_number}")

                        # Get the index of the tracker_id with the current tid
                        mask = self.helmet_tracks.tracker_id == tid
                        idxs = np.where(mask)[0]
                        # Then add the final number to that specific Track
                        if len(idxs) > 0:
                            self.helmet_tracks.data['helmet_number'][idxs[0]] = final_number
                            # This we can later use to skip runs on tracks that have set Helmet nums
                            # We are currently doing runs on them regardless.
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

        # Draw labels for the helmet numbers
        if len(self.helmet_tracks) > 0:
            labels = []
            for i, tid in enumerate(self.helmet_tracks.tracker_id):
                labels.append(f"ID {tid}, Number: {self.helmet_tracks.data['helmet_number'][i]}")

            annotated = self.label_annotator.annotate(annotated, self.helmet_tracks, labels=labels)

        return annotated
import supervision as sv
from trackers import ByteTrackTracker


class Tracker:
    def __init__(self, conf_threshold, roi=None, frame_rate=30.0):
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else 30.0

        # Keep tuned ByteTrack settings for stable IDs.
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

        self.PERSON_CLASS_ID = 1
        self.HELMET_CLASS_ID = 0
        self.conf_threshold = conf_threshold
        self.roi = roi

    def set_roi(self, roi):
        self.roi = roi

    def track_detection(self, detections: sv.Detections, frame=None, frame_idx=None):
        # Separate detections
        people_detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == self.HELMET_CLASS_ID]

        # Filter low-confidence detections before association.
        people_for_tracker = people_detections[people_detections.confidence > self.conf_threshold]
        helmets_for_tracker = helmet_detections[helmet_detections.confidence > self.conf_threshold]

        # Tracking
        self.people_tracks = self.tracker_people.update(people_for_tracker)
        self.helmet_tracks = self.tracker_helmet.update(helmets_for_tracker)

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
                if self.helmet_tracks.confidence[i] == 0:
                    labels.append(f"ID {tid} (pred)")
                else:
                    labels.append(f"ID {tid}")
            annotated = self.label_annotator.annotate(annotated, self.helmet_tracks, labels=labels)

        return annotated

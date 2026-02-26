import cv2
import os
import time
import tkinter as tk

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from collections import defaultdict, Counter

# Main program functions
from functions.register_helmet import register_helmet
from functions.BBExtractor import extract_helmet_box
from functions.roi import load_roi, save_roi, select_roi
from hardware_detector import HardwareDetector

config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_PT_path': "models/1280.pt",
    'Tensor_engine_path': "models/1280.engine",
    'USE_FP16': True,
    'IMGSZ': 1280,
}

data_path = "../videos/DJI_CUT.MP4"
conf_threshold = 0.3
frame_skip = 1

INFERENCE_CONFIG = {
    'conf': conf_threshold,
    'iou': 0.5,
    'max_det': 300,
    'imgsz': 1280,
    'half': False, # Switch til True hvis du bruker GPU
    'device': None, # Same here
    'verbose': False,
}

ROI_PATH = os.path.join("img", "detection_roi.json")

detector = HardwareDetector(config)
model = detector.initialize_model()

# Screen resolution for window sizing
root = tk.Tk()
system_width = root.winfo_screenwidth()
system_height = root.winfo_screenheight()
root.destroy()

# Open video
cap = cv2.VideoCapture(data_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, preview_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read initial frame for ROI.")

roi = load_roi(ROI_PATH)
if roi is None:
    roi = select_roi(preview_frame)
    if roi is not None:
        save_roi(ROI_PATH, roi)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.namedWindow('Yolo vision', cv2.WINDOW_NORMAL)

# Initialize tracker
# BoxAnnotator draws the bounding boxes, LabelAnnotator draws the track ID.
tracker_people = ByteTrackTracker(lost_track_buffer=150)
tracker_helmet = ByteTrackTracker(lost_track_buffer=150)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_count = 0
helmet_saved = False

last_detections = sv.Detections.empty()
prev_frame_time = None
fps_ema = 0.0

# Init trackers
people_tracks = sv.Detections.empty()
helmet_tracks = sv.Detections.empty()
processed_tracker_ids = set()

ocr_votes = defaultdict(list)        # {tracker_id: [list of ocr strings]}
OCR_FRAMES = 3                     # collect votes for N frames before deciding
helmet_numbers_final = {}            # {tracker_id: final_number}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        result = model(
            frame,
            conf=INFERENCE_CONFIG['conf'],
            iou=INFERENCE_CONFIG['iou'],
            max_det=INFERENCE_CONFIG['max_det'],
            imgsz=INFERENCE_CONFIG['imgsz'],
            half=INFERENCE_CONFIG['half'],
            device=INFERENCE_CONFIG['device'],
            verbose=INFERENCE_CONFIG['verbose']
        )[0]


        detections = sv.Detections.from_ultralytics(result)


        PERSON_CLASS_ID = 1
        HELMET_CLASS_ID = 0

        # Separate them
        people_detections = detections[detections.class_id == PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == HELMET_CLASS_ID]

        # Filter out low confidence detections (0.3)
        people_for_tracker = people_detections[people_detections.confidence > conf_threshold]
        helmets_for_tracker = helmet_detections[helmet_detections.confidence > conf_threshold]

        # Tracking
        people_tracks = tracker_people.update(people_for_tracker)
        helmet_tracks = tracker_helmet.update(helmets_for_tracker)

        # Add "helmet_number" to helmet tracks if they do not yet exist
        # data is a dict
        if "helmet_number" not in helmet_tracks.data:
            helmet_tracks.data["helmet_number"] = np.full(len(helmet_tracks), -1, dtype=object)

        # Add the confirmed/accepted helemet numbers into the tracks
        helmet_tracks.data["helmet_number"] = np.array(
            [helmet_numbers_final.get(tid, -1) for tid in helmet_tracks.tracker_id], dtype=object)

        # final numbers for a given track_id is stored in helmet_numbers_final
        # We filter these out of the tracks we are working on.
        non_confirmed_helmets = helmet_tracks[
            np.isin(helmet_tracks.tracker_id, list(helmet_numbers_final.keys()), invert=True)
        ]

        if len(helmet_tracks) > 0:
            #Format for BBExtractor
            det_full = np.column_stack([
                non_confirmed_helmets.xyxy,
                non_confirmed_helmets.confidence,
                non_confirmed_helmets.class_id,
                non_confirmed_helmets.tracker_id,
            ])

            # Extracts the bbox for the helmet
            helmets = extract_helmet_box(det_full, frame)

            if len(helmets) > 0:
                # Gets the OCR result for helmet number based on extracted bbox
                helmet_results = register_helmet(helmets, debug=True)
                for h in helmet_results:
                    tid = h['track_id']
                    number = h['helmet_number']

                    # ocr_votes is a list of OCR results (helmet_number) for a given tracker_id
                    if number != "":  # only count non-empty results
                        ocr_votes[tid].append(number)

                    # Once we have enough votes, pick the winner
                    # Currently this makes it so that when a number is set
                    # it is set forever for that tracker id
                    print("tid: ", tid, " votes: ", ocr_votes[tid])
                    if len(ocr_votes[tid]) >= OCR_FRAMES and tid not in helmet_numbers_final:
                        final_number = Counter(ocr_votes[tid]).most_common(1)[0][0]
                        # Idk if this is actually redundant, because we already add this into helmet_tracks at the end
                        helmet_numbers_final[tid] = final_number
                        print(f"Tracker {tid} final helmet number: {final_number}")

                        # Get the index of the tracker_id with the current tid
                        mask = helmet_tracks.tracker_id == tid
                        idxs = np.where(mask)[0]
                        # Then add the final number to that specific Track
                        if len(idxs) > 0:
                            helmet_tracks.data['helmet_number'][idxs[0]] = final_number
    else:
        pass


    annotated = box_annotator.annotate(frame, people_tracks)
    annotated = box_annotator.annotate(annotated, helmet_tracks)
    if len(people_tracks) > 0:
        labels = []
        for i, tid in enumerate(people_tracks.tracker_id):
            if people_tracks.confidence[i] == 0:
                labels.append(f"ID {tid} (pred)")
            else:
                labels.append(f"ID {tid}")

        annotated = label_annotator.annotate(annotated, people_tracks, labels=labels)

    # Draw labels for the helmet numbers
    if len(helmet_tracks) > 0:
        labels = []
        for i, tid in enumerate(helmet_tracks.tracker_id):
            labels.append(f"ID {tid}, Number: {helmet_tracks.data['helmet_number'][i]}")

        annotated = label_annotator.annotate(annotated, helmet_tracks, labels=labels)

    if roi is not None:
        cv2.polylines(
            annotated,
            [np.array(roi, dtype=np.int32)],
            True,
            (0, 255, 255),
            2,
        )

    now = time.perf_counter()
    if prev_frame_time is not None:
        elapsed = now - prev_frame_time
        if elapsed > 0:
            fps_inst = 1.0 / elapsed
            fps_ema = fps_inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * fps_inst)
    prev_frame_time = now

    fps_text = f"FPS: {fps_ema:.1f}" if fps_ema > 0 else "FPS: --"
    cv2.putText(
        annotated,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    display_frame = cv2.resize(annotated, (1920, 1080))
    cv2.imshow('Yolo vision', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("r"):
        new_roi = select_roi(frame)
        if new_roi is not None:
            roi = new_roi
            save_roi(ROI_PATH, roi)

cap.release()
cv2.destroyAllWindows()
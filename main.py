import cv2
import tkinter as tk
import supervision as sv
from trackers import ByteTrackTracker

# Main program functions
from functions.register_helmet import register_helmet
from functions.BBExtractor import extract_helmet_box
from hardware_detector import HardwareDetector

config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_PT_path': "models/1280.pt",
    'Tensor_engine_path': "models/1280.engine",
    'USE_FP16': True,
    'IMGSZ': 1280,
}

data_path = "DJI_20260211182952_0004_D.MP4"
conf_threshold = 0.3
frame_skip = 1

INFERENCE_CONFIG = {
    'conf': conf_threshold,
    'iou': 0.5,
    'max_det': 300,
    'imgsz': 1280,
    'half': True,
    'device': 0,
    'verbose': False,
}

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

cv2.namedWindow('Yolo vision', cv2.WINDOW_NORMAL)

# Initialize tracker
# BoxAnnotator draws the bounding boxes, LabelAnnotator draws the track ID.
tracker = ByteTrackTracker(lost_track_buffer=150)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_count = 0
helmet_saved = False

last_detections = sv.Detections.empty()

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
        print(f"Frame {frame_count}: YOLO found {len(detections)} boxes")

        PERSON_CLASS_ID = 1
        HELMET_CLASS_ID = 0

        people_detections = detections[detections.class_id == PERSON_CLASS_ID]
        helmet_detections = detections[detections.class_id == HELMET_CLASS_ID]

        people_for_tracker = people_detections[people_detections.confidence > 0.3]
        last_detections = tracker.update(people_for_tracker)
    else:
        pass


    annotated = box_annotator.annotate(frame, last_detections)
    if len(last_detections) > 0:
        labels = []
        for i, tid in enumerate(last_detections.tracker_id):
            if last_detections.confidence[i] == 0:
                labels.append(f"ID {tid} (pred)")
            else:
                labels.append(f"ID {tid}")

        annotated = label_annotator.annotate(annotated, last_detections, labels=labels)

    display_frame = cv2.resize(annotated, (1920, 1080))
    cv2.imshow('Yolo vision', display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


"""

        if not helmet_saved and len(detections) > 0:
            det_numpy = detections.xyxy
            import numpy as np
            det_full = np.column_stack([
                det_numpy,
                detections.confidence,
                detections.class_id
            ])
            helmets = extract_helmet_box(det_full, frame)
            if len(helmets) > 0:
                helmet_results = register_helmets(helmets, debug=False)
                for h in helmet_results:
                    print(f"Helmet {h['bbox']}: Number={h['helmet_number']}, "
                          f"OCR confidence={h['ocr_conf']:.1f}%")
                helmet_saved = True
"""
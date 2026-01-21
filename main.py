from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = "woman_skating.mp4"
MODEL_PATH = "modeln/train/weights/best_openvino_model"
OUTPUT_PATH = "tracked_output.mp4"

IMGSZ = 640                  # YOLO input size
INITIAL_CONF = 0.99          # high confidence for first detection
REFRESH_CONF = 0.5           # lower confidence for periodic re-detection
DETECT_EVERY_N_FRAMES = 60   # YOLO runs every 60 frames (~1 min at 1 FPS)
TRACK_SCALE = 0.5            # scale down frame for faster tracking
MAX_TRAIL_POINTS = 30        # trail length

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO(MODEL_PATH, task="detect")

# -----------------------------
# VIDEO SETUP
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

USE_GUI = hasattr(cv2, 'imshow')
frame_count = 0

# -----------------------------
# TRACKING VARIABLES
# -----------------------------
trackers = []
tracker_ids = []
tracker_classes = []
tracker_confs = []
track_history = {}
next_id = 0

# -----------------------------
# TRACKER CREATION FUNCTION
# -----------------------------
def create_tracker():
    """MOSSE tracker with KCF fallback"""
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        try:
            return cv2.TrackerMOSSE_create()
        except AttributeError:
            return cv2.legacy.TrackerKCF_create()

# -----------------------------
# INITIALIZE TRACKERS FROM YOLO
# -----------------------------
def init_trackers_from_yolo(frame, boxes, classes, confs):
    global next_id
    trackers_local, ids_local, classes_local, confs_local = [], [], [], []

    small = cv2.resize(frame, None, fx=TRACK_SCALE, fy=TRACK_SCALE)

    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = box.astype(int)
        bbox = (
            int(x1 * TRACK_SCALE),
            int(y1 * TRACK_SCALE),
            int((x2 - x1) * TRACK_SCALE),
            int((y2 - y1) * TRACK_SCALE)
        )
        tracker = create_tracker()
        success = tracker.init(small, bbox)
        if success:
            trackers_local.append(tracker)
            ids_local.append(next_id)
            classes_local.append(cls)
            confs_local.append(conf)
            next_id += 1

    return trackers_local, ids_local, classes_local, confs_local

# -----------------------------
# MAIN LOOP
# -----------------------------
print("Starting hybrid YOLO + MOSSE tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_boxes = []

    # -------------------------
    # YOLO DETECTION PHASE
    # -------------------------
    if frame_count % DETECT_EVERY_N_FRAMES == 1:
        # Use high confidence for first frame or lower for periodic refresh
        conf_threshold = INITIAL_CONF if frame_count == 1 else REFRESH_CONF
        print(f"Frame {frame_count}: Running YOLO detection with conf={conf_threshold}...")

        results = model(frame, device="cpu", imgsz=IMGSZ, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            # Initialize or refresh trackers
            trackers, tracker_ids, tracker_classes, tracker_confs = init_trackers_from_yolo(frame, boxes, classes, confs)
            current_boxes = [(box, tid, cls, conf) for box, tid, cls, conf in zip(boxes, tracker_ids, tracker_classes, tracker_confs)]
            print(f"  Initialized/Refreshed {len(trackers)} trackers")
        else:
            trackers, tracker_ids, tracker_classes, tracker_confs = [], [], [], []
            print("  No objects detected")

    # -------------------------
    # TRACKING PHASE
    # -------------------------
    elif trackers:
        small = cv2.resize(frame, None, fx=TRACK_SCALE, fy=TRACK_SCALE)
        new_trackers, new_ids, new_classes, new_confs = [], [], [], []

        for tracker, tid, cls, conf in zip(trackers, tracker_ids, tracker_classes, tracker_confs):
            ok, bbox = tracker.update(small)
            if ok:
                x, y, w, h = bbox
                x1 = int(x / TRACK_SCALE)
                y1 = int(y / TRACK_SCALE)
                x2 = int((x + w) / TRACK_SCALE)
                y2 = int((y + h) / TRACK_SCALE)
                box = np.array([x1, y1, x2, y2])
                current_boxes.append((box, tid, cls, conf))
                new_trackers.append(tracker)
                new_ids.append(tid)
                new_classes.append(cls)
                new_confs.append(conf)

        trackers, tracker_ids, tracker_classes, tracker_confs = new_trackers, new_ids, new_classes, new_confs

    # -------------------------
    # DRAWING
    # -------------------------
    for box, tid, cls, conf in current_boxes:
        x1, y1, x2, y2 = box.astype(int)

        # Stable color per ID
        rng = np.random.RandomState(tid)
        color = tuple(rng.randint(0, 255, 3).tolist())

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label + confidence
        label = f"ID:{tid} {model.names[cls]} {int(conf*100)}%"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Update trail
        center = ((x1 + x2)//2, (y1 + y2)//2)
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append(center)
        if len(track_history[tid]) > MAX_TRAIL_POINTS:
            track_history[tid].pop(0)

        # Draw trail
        if len(track_history[tid]) > 1:
            pts = np.array(track_history[tid], np.int32)
            cv2.polylines(frame, [pts], False, color, 2, lineType=cv2.LINE_AA)

    # Display mode info
    mode = "YOLO" if frame_count % DETECT_EVERY_N_FRAMES == 1 else "Tracking"
    cv2.putText(frame, f"Frame: {frame_count} | Mode: {mode} | Trackers: {len(trackers)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Write frame to video
    video_out.write(frame)

    # Show GUI if available
    if USE_GUI:
        cv2.imshow("Hybrid YOLO + MOSSE Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}... {len(trackers)} trackers active")

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
video_out.release()
if USE_GUI:
    cv2.destroyAllWindows()

print("\nProcessing complete!")
print(f"Total frames: {frame_count}")
print(f"Output saved to: {OUTPUT_PATH}")

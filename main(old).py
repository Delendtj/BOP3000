from ultralytics import YOLO
import cv2
import numpy as np
import threading
from queue import Queue
import time
import argparse

# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', type=str,
                    required=True)
# We should implement usb stream so it can take camera input, in the  future.
parser.add_argument('--source', help='Source to the input video', type=str,
                    required=True)
parser.add_argument('--init-conf', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', type=float,
                    default=0.5, dest='init_conf')
parser.add_argument('--refresh-conf', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', type=float,
                    default=0.3, dest='refresh_conf')
parser.add_argument('--imagesize', type=int,
                    default=640)
parser.add_argument('--detect-every', help="Amount of frames before the model detects. 300 means that it detects every 120th frame.", type=int,
                    default=60, dest='detect_every')

args = parser.parse_args()

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = args.source
MODEL_PATH = args.model

IMGSZ = args.imagesize
INITIAL_CONF = args.init_conf  # lowered for better detection
REFRESH_CONF = args.refresh_conf
DETECT_EVERY_N_FRAMES = args.detect_every
TRACK_SCALE = 1
MAX_TRAIL_POINTS = 30

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
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_interval = 1.0 / fps

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
# THREADING SETUP
# -----------------------------
frame_queue = Queue(maxsize=1)   # main thread puts frames here
yolo_queue = Queue(maxsize=3)    # YOLO results (buffer up to 3)

# -----------------------------
# TRACKER CREATION FUNCTION
# -----------------------------
def create_tracker():
    return cv2.legacy.TrackerMOSSE_create()

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
# YOLO WORKER THREAD
# -----------------------------
def yolo_worker():
    """
    Runs YOLO detection on frames received from the main thread.
    """
    local_frame_count = 0
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:  # sentinel to exit
            break
        local_frame_count, frame_copy = frame_data

        if local_frame_count % DETECT_EVERY_N_FRAMES == 1:
            results = model(frame_copy, device="cpu", imgsz=IMGSZ, conf=INITIAL_CONF, verbose=False)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                if yolo_queue.full():
                    yolo_queue.get_nowait()
                yolo_queue.put((local_frame_count, boxes, classes, confs))

# Start YOLO thread
thread = threading.Thread(target=yolo_worker, daemon=True)
thread.start()

# -----------------------------
# MAIN LOOP
# -----------------------------
print("Starting hybrid YOLO + MOSSE tracking...")

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_boxes = []

    # Send frame to YOLO thread
    if not frame_queue.full():
        frame_queue.put((frame_count, frame.copy()))

    # -------------------------
    # APPLY YOLO RESULTS (LATENCY-TOLERANT)
    # -------------------------
    while not yolo_queue.empty():
        yolo_frame_idx, boxes, classes, confs = yolo_queue.get()
        # Accept results for current frame or any past frames not yet processed
        if yolo_frame_idx <= frame_count:
            trackers, tracker_ids, tracker_classes, tracker_confs = init_trackers_from_yolo(
                frame, boxes, classes, confs
            )
            current_boxes = [(box, tid, cls, conf) for box, tid, cls, conf in zip(
                boxes, tracker_ids, tracker_classes, tracker_confs)]
            print(f"Frame {frame_count}: YOLO updated {len(trackers)} trackers")

    # -------------------------
    # TRACKING PHASE
    # -------------------------
    if trackers:
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
        rng = np.random.RandomState(tid)
        color = tuple(rng.randint(0, 255, 3).tolist())

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {model.names[cls]} {int(conf*100)}%"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Object info box
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Trails
        center = ((x1 + x2)//2, (y1 + y2)//2)
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append(center)
        if len(track_history[tid]) > MAX_TRAIL_POINTS:
            track_history[tid].pop(0)

        if len(track_history[tid]) > 1:
            pts = np.array(track_history[tid], np.int32)
            cv2.polylines(frame, [pts], False, color, 2, lineType=cv2.LINE_AA)

    mode = "Tracking"
    if current_boxes:
        mode = "YOLO"

    current_fps = frame_count / (time.time() - start_time)

    cv2.putText(frame, f"FrameCount: {frame_count} | Mode: {mode} | Trackers: {len(trackers)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"FPS: {format(current_fps, '.2f')}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # -------------------------
    # DISPLAY WITH REAL-TIME SYNC
    # -------------------------
    if USE_GUI:
        cv2.imshow("Hybrid YOLO + MOSSE Tracking", frame)
        elapsed = time.time() - start_time
        expected_time = frame_count / fps
        wait_ms = max(int((expected_time - elapsed) * 1000), 1)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
    else:
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}... {len(trackers)} trackers active")

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)  # stop YOLO thread

print("\nProcessing complete!")
print(f"Total frames: {frame_count}")

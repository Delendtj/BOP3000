import cv2
import numpy as np
import time
from openvino.runtime import Core
from queue import Queue
from threading import Thread

MODEL_XML = "models/tuned_openvino/tuned_model.xml"
VIDEO_PATH = "GX010008.MP4"
INPUT_SIZE = 640
CONF_THRESH = 0.6
ESC_KEY = 27
NUM_ASYNC_REQUESTS = 2
MAX_QUEUE_SIZE = 20
CLASS_NAMES = None

core = Core()
core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})

model = core.read_model(MODEL_XML)
compiled_model = core.compile_model(model, "GPU")
input_layer = compiled_model.input(0)

cv2.namedWindow("YOLOv26-s OpenVINO GPU", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv26-s OpenVINO GPU", 1280, 720)

def resize_to_screen(frame, max_w=0.95, max_h=0.9):
    screen_w = cv2.getWindowImageRect("YOLOv26-s OpenVINO GPU")[2]
    screen_h = cv2.getWindowImageRect("YOLOv26-s OpenVINO GPU")[3]

    h, w = frame.shape[:2]
    scale = min((screen_w * max_w) / w, (screen_h * max_h) / h, 1.0)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return frame

# -------------------- PREPROCESS (LETTERBOX) --------------------
def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    img = padded[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))[None].astype(np.float32) / 255.0

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_shape": (h, w)
    }
    return img, meta

# -------------------- POSTPROCESS --------------------
def postprocess(output, meta):
    h, w = meta["orig_shape"]
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]

    detections = []

    if len(output.shape) == 3:
        output = output[0]

    for det in output:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, score, class_id = det[:6]
        if score < CONF_THRESH:
            continue

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = int(max(0, min(x1, w)))
        y1 = int(max(0, min(y1, h)))
        x2 = int(max(0, min(x2, w)))
        y2 = int(max(0, min(y2, h)))

        detections.append((x1, y1, x2, y2, float(score), int(class_id)))

    return detections

# -------------------- PIPELINES --------------------
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
output_queue = Queue(maxsize=MAX_QUEUE_SIZE)

def frame_reader(video_path, queue):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor, meta = preprocess(frame)
        queue.put((frame, input_tensor, meta))

    cap.release()
    for _ in range(NUM_ASYNC_REQUESTS):
        queue.put(None)

Thread(target=frame_reader, args=(VIDEO_PATH, frame_queue), daemon=True).start()

def inference_worker():
    request = compiled_model.create_infer_request()

    while True:
        item = frame_queue.get()
        if item is None:
            output_queue.put(None)
            break

        frame, input_tensor, meta = item
        request.infer({input_layer: input_tensor})
        output = request.get_output_tensor(0).data[:].astype(np.float32)
        output_queue.put((frame, output, meta))

for _ in range(NUM_ASYNC_REQUESTS):
    Thread(target=inference_worker, daemon=True).start()

# -------------------- DISPLAY LOOP --------------------
prev_time = time.time()
while True:
    item = output_queue.get()
    if item is None:
        break

    frame, output, meta = item
    detections = postprocess(output, meta)

    for x1, y1, x2, y2, score, class_id in detections:
        label = f"{class_id} {score:.2f}"
        if CLASS_NAMES and class_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 7, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    display_frame = resize_to_screen(frame)
    cv2.imshow("YOLOv26-s OpenVINO GPU", display_frame)
    if cv2.waitKey(1) & 0xFF == ESC_KEY:
        break

cv2.destroyAllWindows()

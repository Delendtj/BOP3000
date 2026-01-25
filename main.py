import cv2
import numpy as np
import time
from openvino.runtime import Core
from queue import Queue
from threading import Thread

# ---------------- CONFIG ----------------
MODEL_XML = "models/openvino_updated_model/model.xml"
VIDEO_PATH = "victor.mp4"
INPUT_SIZE = 640
CONF_THRESH = 0.4
ESC_KEY = 27
NUM_ASYNC_REQUESTS = 2
MAX_QUEUE_SIZE = 20
CLASS_NAMES = None

# ---------------- LOAD MODEL ----------------
core = Core()

# CRITICAL FIX: Force FP32 precision on GPU
core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})

model = core.read_model(MODEL_XML)
compiled_model = core.compile_model(model, "GPU")
input_layer = compiled_model.input(0)


# ---------------- PREPROCESS ----------------
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]  # BGR -> RGB
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, :] / 255.0, dtype=np.float32)
    return img


# ---------------- POSTPROCESS ----------------
def postprocess(output, frame_shape):
    h, w = frame_shape[:2]
    detections = []

    # Handle batch dimension
    if len(output.shape) == 3:
        output = output[0]

    # Calculate scaling factors from input size to original frame
    scale_x = w / INPUT_SIZE
    scale_y = h / INPUT_SIZE

    for det in output:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, score, class_id = det[:6]
        score = float(score)

        if score < CONF_THRESH:
            continue

        # Scale coordinates from 640x640 to original frame size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Clamp to frame boundaries
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        detections.append((x1, y1, x2, y2, score, int(class_id)))

    return detections


# ---------------- QUEUES ----------------
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
output_queue = Queue(maxsize=MAX_QUEUE_SIZE)


# ---------------- FRAME READER ----------------
def frame_reader(video_path, queue):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor = preprocess(frame)
        queue.put((frame, input_tensor))
    cap.release()
    # Signal inference loop to stop
    for _ in range(NUM_ASYNC_REQUESTS):
        queue.put(None)


Thread(target=frame_reader, args=(VIDEO_PATH, frame_queue), daemon=True).start()


# ---------------- INFERENCE WORKER ----------------
def inference_worker():
    request = compiled_model.create_infer_request()

    while True:
        item = frame_queue.get()
        if item is None:
            output_queue.put(None)
            break

        frame, input_tensor = item

        # Run inference
        request.infer({input_layer: input_tensor})

        # Get output
        output_tensor = request.get_output_tensor(0)
        output = output_tensor.data[:].astype(np.float32)

        # Put result in output queue
        output_queue.put((frame, output))


# Start inference workers
for _ in range(NUM_ASYNC_REQUESTS):
    Thread(target=inference_worker, daemon=True).start()

# ---------------- DISPLAY LOOP ----------------
prev_time = time.time()
while True:
    item = output_queue.get()
    if item is None:
        break

    frame, output = item
    detections = postprocess(output, frame.shape)

    for x1, y1, x2, y2, score, class_id in detections:
        label = f"{class_id} {score:.2f}"
        if CLASS_NAMES and class_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 7, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv26-s OpenVINO GPU", frame)
    if cv2.waitKey(1) & 0xFF == ESC_KEY:
        break

cv2.destroyAllWindows()
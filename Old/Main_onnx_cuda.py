import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
from queue import Queue
from threading import Thread, Lock

cuda.init()
device = cuda.Device(0)
cuda_context = device.make_context()
context_lock = Lock()

# ---------------- CONFIG ----------------
ONNX_MODEL_PATH = "../models/model_fp16.onnx"  # Change to model_fp16.onnx for FP16
ENGINE_PATH = "models/yolo_engine.trt"  # Cached TensorRT engine
VIDEO_PATH = "../testdata.mp4"
INPUT_SIZE = 640
CONF_THRESH = 0.7
ESC_KEY = 27
NUM_ASYNC_REQUESTS = 2
MAX_QUEUE_SIZE = 20
CLASS_NAMES = None
USE_FP16 = True

# ---------------- BUILD OR LOAD ENGINE ----------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, use_fp16=False):
    """Build TensorRT engine from ONNX model"""
    print(f"\n{'=' * 60}")
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"This may take a few minutes on first run...")
    print(f"{'=' * 60}\n")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")

    # Build engine
    print("Building engine... (this takes time)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print('ERROR: Failed to build engine')
        return None

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ Engine saved to {engine_path}\n")
    return serialized_engine


def load_engine(engine_path):
    """Load TensorRT engine from file"""
    with open(engine_path, 'rb') as f:
        return f.read()


# Build or load engine
import os

if os.path.exists(ENGINE_PATH):
    print(f"✓ Loading cached engine from {ENGINE_PATH}")
    serialized_engine = load_engine(ENGINE_PATH)
else:
    serialized_engine = build_engine(ONNX_MODEL_PATH, ENGINE_PATH, USE_FP16)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

print(f"Engine info:")
print(f"  Input: {engine.get_tensor_name(0)}, shape: {engine.get_tensor_shape(engine.get_tensor_name(0))}")
print(f"  Output: {engine.get_tensor_name(1)}, shape: {engine.get_tensor_shape(engine.get_tensor_name(1))}")
print(f"  Device: {cuda.Device(0).name()}\n")

# Allocate buffers
input_binding = engine.get_tensor_name(0)
output_binding = engine.get_tensor_name(1)

input_shape = engine.get_tensor_shape(input_binding)
output_shape = engine.get_tensor_shape(output_binding)

input_size_bytes = trt.volume(input_shape) * np.dtype(np.float32).itemsize
output_size_bytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize

# Allocate device memory
d_input = cuda.mem_alloc(input_size_bytes)
d_output = cuda.mem_alloc(output_size_bytes)

# Create CUDA stream
stream = cuda.Stream()


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

    # Calculate scaling factors
    scale_x = w / INPUT_SIZE
    scale_y = h / INPUT_SIZE

    for det in output:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, score, class_id = det[:6]
        score = float(score)

        if score < CONF_THRESH:
            continue

        # Scale coordinates
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

    # Get original video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / original_fps if original_fps > 0 else 0

    print(f"Video FPS: {original_fps:.2f}")

    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        queue.put((frame, input_tensor, last_frame_time + frame_delay))

        # Maintain original FPS timing
        if frame_delay > 0:
            elapsed = time.time() - last_frame_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        last_frame_time = time.time()

    cap.release()
    # Signal inference workers to stop
    for _ in range(NUM_ASYNC_REQUESTS):
        queue.put(None)


Thread(target=frame_reader, args=(VIDEO_PATH, frame_queue), daemon=True).start()


# ---------------- INFERENCE WORKER ----------------
def inference_worker():
    # Push context to this thread
    with context_lock:
        cuda_context.push()

    try:
        # Each worker needs its own context and stream
        worker_context = engine.create_execution_context()
        worker_stream = cuda.Stream()

        # Allocate device memory for this worker
        worker_d_input = cuda.mem_alloc(input_size_bytes)
        worker_d_output = cuda.mem_alloc(output_size_bytes)

        while True:
            item = frame_queue.get()
            if item is None:
                output_queue.put(None)
                break

            frame, input_tensor, target_time = item

            # Copy input to device
            cuda.memcpy_htod_async(worker_d_input, input_tensor, worker_stream)

            # Set tensor addresses
            worker_context.set_tensor_address(input_binding, int(worker_d_input))
            worker_context.set_tensor_address(output_binding, int(worker_d_output))

            # Execute
            worker_context.execute_async_v3(stream_handle=worker_stream.handle)

            # Copy output from device
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, worker_d_output, worker_stream)

            # Synchronize
            worker_stream.synchronize()

            # Put result in output queue with timing
            output_queue.put((frame, output, target_time))
    finally:
        # Pop context when thread exits
        with context_lock:
            cuda_context.pop()


# Start inference workers
for _ in range(NUM_ASYNC_REQUESTS):
    Thread(target=inference_worker, daemon=True).start()

# ---------------- DISPLAY LOOP ----------------
prev_time = time.time()
frame_count = 0
start_time = time.time()

while True:
    item = output_queue.get()
    if item is None:
        break

    frame, output, target_time = item
    detections = postprocess(output, frame.shape)

    for x1, y1, x2, y2, score, class_id in detections:
        label = f"{class_id} {score:.2f}"
        if CLASS_NAMES and class_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 7, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate instantaneous FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    frame_count += 1

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv26-s TensorRT", frame)

    # Wait to match original video timing
    wait_time = max(1, int((target_time - time.time()) * 1000))
    if cv2.waitKey(wait_time) & 0xFF == ESC_KEY:
        break

cv2.destroyAllWindows()

# Cleanup CUDA context
with context_lock:
    cuda_context.pop()

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"Processed {frame_count} frames in {elapsed:.2f}s")
print(f"Average FPS: {frame_count / elapsed:.2f}")
print(f"{'=' * 60}")
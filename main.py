import cv2
import numpy as np
import tkinter as tk
from boxmot import ByteTrack

from hardware_detector import HardwareDetector
from functions.BBExtractor import extract_helmet_box


config = {
    'Model_OV_path': "models/best_openvino_model",
    'Model_Cuda_path': "models/model_fp16.onnx",
    'Model_PT_path': "models/best.pt",
}

INFERENCE_CONFIG = {
    'conf': 0.6,
    'iou': 0.45,  # NMS IOU threshold
    'max_det': 300,
    'imgsz': 640,
    'half': False,  # Use FP16 (set True for compatible GPUs)
    'device': None,  # Auto-detect (or set '0' for GPU, 'cpu' for CPU)
    'verbose': False,  # Suppress YOLO logging
}

# Data
data_path = "DJI_20260211183300_0036_D.MP4"


detector = HardwareDetector(config)
model = detector.initialize_model()

# Dette er for å finne current system resolution size.
# Kan få problemer for hvis man kjører med flere skjermer/scaled res.
root = tk.Tk()
system_width = root.winfo_screenwidth()
system_height = root.winfo_screenheight()

# Open video
cap = cv2.VideoCapture(data_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Hvis systemets resolution er mindre enn video res så bruker vi system res.
output_wind_height = system_height if system_height < int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_wind_width = system_width if system_width < int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cv2.namedWindow('Yolo vision', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yolo vision', output_wind_width, output_wind_height)

# NOTE (til når vi skal adde OCR):
# HVis vi går med å bruke dedikert kamera for REID så burde vi
# bare gjøre OCR(hjelm)/REID på en viss del av input bildene, så ikke hele.
# + ha store mellomrom mellom hver gang
# (for hver N detection innenfor en spesifik del av bilde)

# Initialize tracker
tracker = ByteTrack(device='cpu', half=False, track_high_thresh=0.65)


frame_count = 0
processed_count = 0
helmet_saved = False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

     #INFERENCE
    results = model(
        frame,
        conf=INFERENCE_CONFIG['conf'],
        iou=INFERENCE_CONFIG['iou'],
        max_det=INFERENCE_CONFIG['max_det'],
        imgsz=INFERENCE_CONFIG['imgsz'],
        half=INFERENCE_CONFIG['half'],
        device=INFERENCE_CONFIG['device'],
        verbose=INFERENCE_CONFIG['verbose']
    )

    # Extract detections
    boxes = results[0].boxes

    if len(boxes) > 0:
        # Convert to numpy array format: [x1, y1, x2, y2, conf, class]
        detections = np.column_stack([
            boxes.xyxy.cpu().numpy(),  # Bounding box coordinates
            boxes.conf.cpu().numpy().reshape(-1, 1),  # Confidence scores
            boxes.cls.cpu().numpy().reshape(-1, 1)  # Class IDs
        ])

        # Update tracker with detections
        tracks = tracker.update(detections, frame)

        # Plot tracking results on frame
        tracker.plot_results(frame, fontscale=1, show_lost=True, show_trajectories=False)

        # Extract helmet boxes for OCR (class 0 only)
        if not helmet_saved:
            helmets = extract_helmet_box(detections, frame)
            if len(helmets) > 0:
                helmet_saved = True


        # Display detection count
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Yolo vision', frame)
    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
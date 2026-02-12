import cv2
import numpy as np

# For å hente system resolution
import tkinter as tk

# Egen utils
from hardware_detector import HardwareDetector
from inference import InferenceEngine
from utilities.tracker import tracking

# Til tracking
from boxmot import BotSort
from boxmot import ByteTrack
from pathlib import Path

config = {
    'Model_OV_path': "models/tuned_openvino/tuned_model.xml",
    'Model_Cuda_path': "models/model_fp16.onnx",
    'Tensor_engine_path': "models/tensor_engine",
    'USE_FP16': True
}

data_path="../videos/MVI_5224.MP4"
conf_threshold = 0.6
input_size = 640


#henter info om systemet og starter riktig modell backend
detector = HardwareDetector(config)
model = detector.initialize_model()
engine = InferenceEngine(model, detector.hardware_type)

# Dette er for å finne current system resolution size.
# Kan få problemer for hvis man kjører med flere skjermer/scaled res.
root = tk.Tk()

system_width = root.winfo_screenwidth()
system_height = root.winfo_screenheight()

# Open video
cap = cv2.VideoCapture(data_path)

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

# Vi kan bruke ByteTrack her også, merket at det gikk litt raskere.
tracker = BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device='cpu', half=False, track_high_thresh=0.65)

# Tror vi ender opp med å må kjøre REID på en egen thread på en cropped detection/hjelmnummer.
# Virker som mye mer setup og kompleks da, men tror det blir mer effetkivt.
# SÅ pipeline blir å  detect_hjelm > OCR(detect_hjelm) > output_tall > REID(output_tall)
# run_reid(tracker)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Lagre original shape for å sette riktig bbox senere.
    org_shape = frame.shape

    # Preprocess: resize to 640x640
    resized = cv2.resize(frame, (640, 640))

    # Convert to model input format
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    input_data = np.transpose(normalized, (2, 0, 1))

    input_data_batched = np.expand_dims(input_data, axis=0)

    # DETTE KJØRER INFERENCE
    output = engine.run(input_data_batched)

    # Dette er formatet output kommer i
    print(f"Shape: {output.shape}")
    print(f"Value: {output[0, 0, :]}")

    # Legg til tracking id på output.
    tracking(tracker, output, frame, org_shape)

    cv2.imshow('Yolo vision', frame)

    # Hvis de siste 8 bitsa utgjør tallet 27
    # De siste 8 bitsa representerer ASCII tegnet
    # Der 27 = ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

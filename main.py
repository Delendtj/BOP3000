import cv2
import numpy as np
from hardware_detector import HardwareDetector
from inference import InferenceEngine



config = {
    'Model_OV_path': "models/tuned_openvino/tuned_model.xml",
    'Model_Cuda_path': "models/model_fp16.onnx",
    'Tensor_engine_path': "models/tensor_engine",
    'USE_FP16': True
}

data_path="testdata.mp4"
conf_threshold = 0.6
input_size = 640


#henter info om systemet og starter riktig modell backend
detector = HardwareDetector(config)
model = detector.initialize_model()
engine = InferenceEngine(model, detector.hardware_type)

# Open video
cap = cv2.VideoCapture(data_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess: resize to 640x640
    resized = cv2.resize(frame, (640, 640))

    # Convert to model input format
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    input_data = np.transpose(normalized, (2, 0, 1))

    input_data_batched = np.expand_dims(input_data, axis=0)

    # DETTE KJØRER INFERENCE
    output = engine.run(input_data_batched)

    cv2.imshow('Yolo vision', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

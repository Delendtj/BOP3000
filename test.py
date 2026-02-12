import os

model_path = "models/tuned_openvino/tuned_model.xml"
weights_path = "models/tuned_openvino/tuned_model.bin"

print(f"XML exists: {os.path.exists(model_path)}")
print(f"BIN exists: {os.path.exists(weights_path)}")
print(f"XML size: {os.path.getsize(model_path)} bytes")
print(f"BIN size: {os.path.getsize(weights_path)} bytes")

from openvino import Core

try:
    core = Core()
    model = core.read_model(model="models/tuned_openvino/tuned_model.xml")
    compiled_model = core.compile_model(model, "CPU")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

import openvino as ov
print(f"OpenVINO version: {ov.__version__}")
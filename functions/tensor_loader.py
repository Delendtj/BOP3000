import os
from ultralytics import YOLO

def init_tensorrt(config):
    pt_path    = config.get('Model_PT_path', 'models/best.pt')
    use_fp16   = config.get('USE_FP16', True)
    imgsz      = config.get('IMGSZ', 640)  # read from config, default to 640

    if 'Tensor_engine_path' in config:
        engine_path = config['Tensor_engine_path']
    else:
        engine_path = os.path.splitext(pt_path)[0] + '.engine'

    if os.path.exists(engine_path):
        try:
            print(f"  Loading TensorRT engine: {engine_path}")
            model = YOLO(engine_path, task='detect')
            print("  ✓ TensorRT engine loaded successfully")
            return model
        except Exception as e:
            print(f"  ✗ Engine load failed ({e}), rebuilding...")

    if os.path.exists(pt_path):
        try:
            print(f"  Building TensorRT engine at imgsz={imgsz} from: {pt_path}")
            print("  This will take several minutes — TensorRT is profiling for your GPU...")
            model = YOLO(pt_path, task='detect')
            # Pass imgsz here so the engine is built for the right input size
            model.export(format='engine', half=use_fp16, simplify=True, imgsz=imgsz)

            model = YOLO(engine_path, task='detect')
            print("  ✓ TensorRT engine built and loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"TensorRT build failed: {e}") from e

    raise FileNotFoundError(
        f"No model files found. Checked:\n"
        f"  engine : {engine_path}\n"
        f"  pt     : {pt_path}"
    )
import subprocess
from ultralytics import YOLO
from functions.detection.tensor_loader import init_tensorrt


class HardwareDetector:
    def __init__(self, config):
        self.config = config
        self.hardware_type = None

    def detect_hardware(self):
        """Detect available hardware and choose the best backend."""
        if self._has_nvidia_gpu():
            self.hardware_type = 'cuda'
            print("NVIDIA GPU detected, using CUDA/TensorRT")
            return 'cuda'

        self.hardware_type = 'openvino'
        print("No NVIDIA GPU found, using OpenVINO")
        return 'openvino'

    def _has_nvidia_gpu(self):
        """Return True if nvidia-smi reports at least one GPU."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  Found NVIDIA GPU: {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def initialize_model(self):
        """Detect hardware (if not done yet) and return a ready-to-use model."""
        if self.hardware_type is None:
            self.detect_hardware()

        print(f"\nInitializing {self.hardware_type.upper()} model...")

        if self.hardware_type == 'cuda':
            return init_tensorrt(self.config)
        else:
            return self._init_openvino_model()

    def _init_openvino_model(self):
        """Load a YOLO model for OpenVINO (Intel GPU/CPU) or fall back to PyTorch."""
        ov_path = self.config.get('Model_OV_path', 'models/best_openvino_model')
        pt_path = self.config.get('Model_PT_path', 'models/best.pt')


        if __import__('os').path.exists(ov_path):
            try:
                print(f"  Loading OpenVINO model: {ov_path}")
                model = YOLO(ov_path, task='detect')
                print("  ✓ OpenVINO model loaded successfully")
                return model
            except Exception as e:
                print(f"  ✗ OpenVINO loading failed: {e}")


        if __import__('os').path.exists(pt_path):
            print(f"  Loading PyTorch model (CPU fallback): {pt_path}")
            model = YOLO(pt_path, task='detect')
            print("  ✓ PyTorch model loaded")
            return model

        raise FileNotFoundError(
            f"No model files found. Checked:\n"
            f"  openvino : {ov_path}\n"
            f"  pytorch  : {pt_path}"
        )

    def get_hardware_type(self):
        """Return the detected hardware type, running detection if needed."""
        if self.hardware_type is None:
            self.detect_hardware()
        return self.hardware_type
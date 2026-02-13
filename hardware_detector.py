import os
import subprocess
from ultralytics import YOLO


class HardwareDetector:
    def __init__(self, config):
        self.config = config
        self.hardware_type = None

    def detect_hardware(self):
        """Detect available hardware and choose best backend"""
        if self._has_nvidia_gpu():
            self.hardware_type = 'cuda'
            print("NVIDIA GPU detected, using CUDA/TensorRT")
            return 'cuda'

        self.hardware_type = 'openvino'
        print("No NVIDIA GPU, using OpenVINO")
        return 'openvino'

    def _has_nvidia_gpu(self):
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                print(f"  Found NVIDIA GPU: {gpu_name}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return False

    def initialize_model(self):
        """Initialize YOLO model with appropriate backend"""
        if self.hardware_type is None:
            self.detect_hardware()

        print(f"\nInitializing {self.hardware_type.upper()} model with YOLO backend...")

        if self.hardware_type == 'cuda':
            return self._init_cuda_model()
        else:  # openvino
            return self._init_openvino_model()

    def _init_cuda_model(self):
        """Initialize YOLO model for NVIDIA GPU"""
        # Try different model formats in order of preference
        onnx_path = self.config.get('Model_Cuda_path', 'models/model_fp16.onnx')
        pt_path = self.config.get('Model_PT_path', 'models/best.pt')

        # Option 1: Try ONNX (good for TensorRT)
        if os.path.exists(onnx_path):
            try:
                print(f"  Loading ONNX model: {onnx_path}")
                model = YOLO(onnx_path, task='detect')
                print("  ✓ CUDA model loaded successfully (ONNX)")
                return model
            except Exception as e:
                print(f"  ✗ ONNX loading failed: {e}")

        # Option 2: Fallback to PyTorch (YOLO will optimize for CUDA)
        if os.path.exists(pt_path):
            print(f"  Loading PyTorch model: {pt_path}")
            model = YOLO(pt_path, task='detect')
            print("  ✓ CUDA model loaded successfully (PyTorch)")
            return model

        raise FileNotFoundError(f"No model found. Checked: {onnx_path}, {pt_path}")

    def _init_openvino_model(self):
        """Initialize YOLO model for OpenVINO (Intel GPU/CPU)"""
        ov_path = self.config.get('Model_OV_path', 'models/best_openvino_model')
        pt_path = self.config.get('Model_PT_path', 'models/best.pt')

        # Option 1: Try OpenVINO exported model
        if os.path.exists(ov_path):
            try:
                print(f"  Loading OpenVINO model: {ov_path}")
                model = YOLO(ov_path, task='detect')
                print("  ✓ OpenVINO model loaded successfully")
                return model
            except Exception as e:
                print(f"  ✗ OpenVINO loading failed: {e}")

        # Option 2: Fallback to PyTorch (will run on CPU)
        if os.path.exists(pt_path):
            print(f"  Loading PyTorch model: {pt_path}")
            model = YOLO(pt_path, task='detect')
            print("  ✓ PyTorch model loaded (running on CPU)")
            return model

        raise FileNotFoundError(f"No model found. Checked: {ov_path}, {pt_path}")

    def get_hardware_type(self):
        """Get the detected hardware type"""
        if self.hardware_type is None:
            self.detect_hardware()
        return self.hardware_type
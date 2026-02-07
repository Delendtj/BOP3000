import os
import sys
import subprocess

import numpy as np


class HardwareDetector:
    def __init__(self, config):

        self.config = config
        self.hardware_type = None

    def detect_hardware(self):
        if self._has_nvidia_gpu():
            self.hardware_type = 'cuda'
            print("NVIDIA GPU, using TensorRT")
            return 'cuda'

        self.hardware_type = 'openvino'
        print("No NVIDIA GPU, using OpenVINO")
        return 'openvino'

    def _has_nvidia_gpu(self):
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
        if self.hardware_type is None:
            self.detect_hardware()

        print(f"\nInitializing {self.hardware_type.upper()} model...")

        if self.hardware_type == 'cuda':
            return self._init_tensorrt_model()
        else:  # openvino
            return self._init_openvino_model()

    def _init_tensorrt_model(self):
        model_path = self.config['Model_Cuda_path']
        engine_path = self.config['Tensor_engine_path']

        if not engine_path.endswith('.engine'):
            engine_path = f"{engine_path}.engine"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        try:
            import tensor_loader
        except ImportError:
            raise ImportError(
                "Tensor_loader fil ikke funnet."
            )

        use_fp16 = self.config.get('USE_FP16', True)
        # init tensorrt fra filen
        trt_objects = Tensor_loader.init_tensorrt(
            onnx_path=model_path,
            engine_path=engine_path,
            use_fp16=use_fp16
        )
        print("  TensorRT initialized successfully")
        return trt_objects

    def _init_openvino_model(self):
        model_path = self.config['Model_OV_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OpenVINO model not found: {model_path}")
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError(
                "OpenVINO not installed. Install with:\n"
                "pip install openvino"
            )
        print(f"  Loading model: {model_path}")
        core = Core()
        model = core.read_model(model_path)
        devices = core.available_devices
        device = 'GPU' if 'GPU' in devices else 'CPU'
        compiled_model = core.compile_model(model, device)
        print(f"  OpenVINO model loaded successfully on {device}")
        return compiled_model



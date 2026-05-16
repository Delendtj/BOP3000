import logging
import os
import importlib.util

import torch
from ultralytics import YOLO


class HardwareDetector:
    def __init__(self, config):
        self.config = config
        self.hardware_type = None

    def detect_hardware(self):
        """Detect the best usable inference backend for this machine."""
        logger = logging.getLogger(__name__)
        can_use_tensorrt, reasons = self._can_use_tensorrt()
        for reason in reasons:
            logger.info(reason)

        if can_use_tensorrt:
            self.hardware_type = 'cuda'
            logger.info("Backend selection: TensorRT/CUDA")
            return 'cuda'

        self.hardware_type = 'openvino'
        logger.info("Backend selection: OpenVINO/PyTorch fallback")
        return 'openvino'

    def _can_use_tensorrt(self):
        """
        Return whether the TensorRT path is usable.

        This is stricter than checking for an NVIDIA GPU name because the
        project needs a working CUDA + TensorRT Python runtime, not just
        detectable hardware.
        """
        reasons = []

        if not torch.cuda.is_available():
            reasons.append("TensorRT check failed: torch.cuda.is_available() is False.")
            return False, reasons
        reasons.append("TensorRT check passed: CUDA device is available to PyTorch.")

        if importlib.util.find_spec("tensorrt") is None:
            reasons.append("TensorRT check failed: Python module 'tensorrt' is not installed.")
            return False, reasons
        reasons.append("TensorRT check passed: Python module 'tensorrt' is installed.")

        try:
            import tensorrt as trt

            _ = trt.Logger(trt.Logger.WARNING)
        except Exception as exc:
            reasons.append(f"TensorRT check failed: import/runtime initialization error: {exc}")
            return False, reasons

        reasons.append("TensorRT check passed: TensorRT runtime imported successfully.")
        return True, reasons

    def initialize_model(self):
        """Detect hardware (if not done yet) and return a ready-to-use model."""
        if self.hardware_type is None:
            self.detect_hardware()

        logging.getLogger(__name__).info("Initializing backend '%s'...", self.hardware_type)

        if self.hardware_type == 'cuda':
            from functions.detection.tensor_loader import init_tensorrt
            return init_tensorrt(self.config)
        else:
            return self._init_openvino_model()

    def _init_openvino_model(self):
        """Load a YOLO model for OpenVINO (Intel GPU/CPU) or fall back to PyTorch."""
        logger = logging.getLogger(__name__)
        ov_path = self.config.get('Model_OV_path', 'models/best_openvino_model')
        pt_path = self.config.get('Model_PT_path', 'models/best.pt')

        if os.path.exists(ov_path):
            try:
                logger.info("Loading OpenVINO model: %s", ov_path)
                model = YOLO(ov_path, task='detect')
                logger.info("OpenVINO model loaded successfully")
                self.hardware_type = 'openvino'
                return model
            except Exception as e:
                logger.error("OpenVINO loading failed: %s", e)

        if os.path.exists(pt_path):
            logger.info("Loading PyTorch model (CPU fallback): %s", pt_path)
            model = YOLO(pt_path, task='detect')
            logger.info("PyTorch model loaded")
            self.hardware_type = 'pytorch'
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

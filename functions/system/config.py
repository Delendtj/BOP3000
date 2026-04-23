import configparser
import os

def create_config(path):
    """
    Creates config file based on config below.
    Only creates when config doesn't exist.
    """
    if os.path.exists(path):
        return

    config = configparser.ConfigParser()

    config["Path"] = {
        "WIDE_SOURCE": "../videos/wide_cam.mp4",
        "CLOSE_SOURCE": "../videos/close_cam.mp4",
        "YOLO_ROI_PATH": os.path.join("data", "yolo_roi.json"),
        "OCR_ROI_PATH": os.path.join("data", "ocr_roi.json"),
        "CLOSE_YOLO_ROI_PATH": os.path.join("data", "close_yolo_roi.json"),
        "CLOSE_OCR_ROI_PATH": os.path.join("data", "close_ocr_roi.json"),
        "HOMOGRAPHY_PATH": os.path.join("data", "homography.json"),
        "RINK_WIDE_H_PATH": os.path.join("data", "homography_wide.json"),
        "RINK_CLOSE_H_PATH": os.path.join("data", "homography_close.json"),
        "FINISH_LINE_PATH": os.path.join("data", "finish_line.json"),
    }

    config["Model"] = {
        'Model_OV_path': "models/best_openvino_model",
        'Model_PT_path': "models/1280.pt",
        'Tensor_engine_path': "models/1280.engine",
        'USE_FP16': "True",
        'IMGSZ': "1280",
    }

    config["FrameSync"] = {
        "SYNC_MISS_LOG_INTERVAL": "2.0",
        "MAX_SYNC_DELTA": "0.05"
    }

    config["ModelClass"] = {
        "HELMET_CLASS_ID": "0",
        "PERSON_CLASS_ID": "1"
    }

    CONF_THRESHOLD = 0.2

    config["Inference"] = {
        'conf': str(CONF_THRESHOLD),
        'iou': "0.5",
        'max_det': "100",
        'imgsz': "1280",
        'half': "True", # Switch til True hvis du bruker GPU
        'device': "", # Same here
        'verbose': "False",
    }

    config["Runtime"] = {
        "CONF_THRESHOLD": str(CONF_THRESHOLD),
        "FRAME_SKIP": "1",
        "OCR_VOTE": "3",  # collect votes for N frames before deciding
        "CLOSE_HELMET_PERSON_MAX_DIST": "80",
        "CLOSE_HELMET_PERSON_MAX_BELOW_RATIO": "0.08",
    }

    RINK_GOAL_LINE_OFFSET = 26

    config["Rink"] = {
        "RINK_BOUNDS": "-15.0,15.0,-30.0,30.0",
        "RINK_GOAL_LINE_OFFSET": str(RINK_GOAL_LINE_OFFSET),
        "RINK_RED_LINES": f"0.0,{-RINK_GOAL_LINE_OFFSET},{RINK_GOAL_LINE_OFFSET}",
        "RINK_MATCH_MAX_DIST": "1.5",
        "multi_cam_window_name": "Multi-cam view",
        "rink_window_name": "Rink view",
        "lap_window_name": "Lap Count"
    }

    config["OCR"] = {
        "BASE_URL": "http://127.0.0.1:1234",
        "MODEL": "glm-ocr",
        "PROMPT": "Return ONLY the numbers visible on this helmet. Output digits only, no punctuation or explanation. If no number is visible, output unknown.",
        "TIMEOUT": "5",
    }

    with open(path, "w", encoding="utf-8") as config_file:
        config.write(config_file)


def _parse_csv_floats(value):
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def load_config(path):
    """
    Load the config from path and return it as a nested dict.
    """

    create_config(path)
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")

    return {
        "Path": {
            "WIDE_SOURCE": config["Path"]["WIDE_SOURCE"],
            "CLOSE_SOURCE": config["Path"]["CLOSE_SOURCE"],
            "YOLO_ROI_PATH": config["Path"]["YOLO_ROI_PATH"],
            "OCR_ROI_PATH": config["Path"]["OCR_ROI_PATH"],
            "CLOSE_YOLO_ROI_PATH": config["Path"]["CLOSE_YOLO_ROI_PATH"],
            "CLOSE_OCR_ROI_PATH": config["Path"]["CLOSE_OCR_ROI_PATH"],
            "HOMOGRAPHY_PATH": config["Path"]["HOMOGRAPHY_PATH"],
            "RINK_WIDE_H_PATH": config["Path"]["RINK_WIDE_H_PATH"],
            "RINK_CLOSE_H_PATH": config["Path"]["RINK_CLOSE_H_PATH"],
            "FINISH_LINE_PATH": config["Path"]["FINISH_LINE_PATH"],
        },
        "Model": {
            "Model_OV_path": config["Model"]["Model_OV_path"],
            "Model_PT_path": config["Model"]["Model_PT_path"],
            "Tensor_engine_path": config["Model"]["Tensor_engine_path"],
            "USE_FP16": config.getboolean("Model", "USE_FP16"),
            "IMGSZ": config.getint("Model", "IMGSZ"),
        },
        "FrameSync": {
            "SYNC_MISS_LOG_INTERVAL": config.getfloat("FrameSync", "SYNC_MISS_LOG_INTERVAL"),
            "MAX_SYNC_DELTA": config.getfloat("FrameSync", "MAX_SYNC_DELTA"),
        },
        "ModelClass": {
            "HELMET_CLASS_ID": config.getint("ModelClass", "HELMET_CLASS_ID"),
            "PERSON_CLASS_ID": config.getint("ModelClass", "PERSON_CLASS_ID"),
        },
        "Inference": {
            "conf": config.getfloat("Inference", "conf"),
            "iou": config.getfloat("Inference", "iou"),
            "max_det": config.getint("Inference", "max_det"),
            "imgsz": config.getint("Inference", "imgsz"),
            "half": config.getboolean("Inference", "half"),
            "device": config["Inference"]["device"] or None,
            "verbose": config.getboolean("Inference", "verbose"),
        },
        "Runtime": {
            "CONF_THRESHOLD": config.getfloat("Runtime", "CONF_THRESHOLD"),
            "FRAME_SKIP": config.getint("Runtime", "FRAME_SKIP"),
            "OCR_VOTE": config.getint("Runtime", "OCR_VOTE"),
            "CLOSE_HELMET_PERSON_MAX_DIST": config.getfloat("Runtime", "CLOSE_HELMET_PERSON_MAX_DIST"),
            "CLOSE_HELMET_PERSON_MAX_BELOW_RATIO": config.getfloat("Runtime", "CLOSE_HELMET_PERSON_MAX_BELOW_RATIO"),
        },
        "Rink": {
            "RINK_BOUNDS": _parse_csv_floats(config["Rink"]["RINK_BOUNDS"]),
            "RINK_GOAL_LINE_OFFSET": config.getfloat("Rink", "RINK_GOAL_LINE_OFFSET"),
            "RINK_RED_LINES": _parse_csv_floats(config["Rink"]["RINK_RED_LINES"]),
            "RINK_MATCH_MAX_DIST": config.getfloat("Rink", "RINK_MATCH_MAX_DIST"),
            "multi_cam_window_name": config["Rink"]["multi_cam_window_name"],
            "rink_window_name": config["Rink"]["rink_window_name"],
            "lap_window_name": config["Rink"]["lap_window_name"],
        },
        "OCR": {
            "BASE_URL": config.get("OCR", "BASE_URL", fallback="http://127.0.0.1:1234/v1"),
            "MODEL": config.get("OCR", "MODEL", fallback="glm-ocr"),
            "PROMPT": config.get("OCR", "PROMPT", fallback="Return ONLY the numbers visible on this helmet. Output digits only, no punctuation or explanation. If no number is visible, output unknown."),
            "TIMEOUT": config.getfloat("OCR", "TIMEOUT", fallback=5.0),
        },
    }

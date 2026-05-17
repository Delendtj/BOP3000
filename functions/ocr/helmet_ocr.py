

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from . import helmet_ocr_llm
from .helmet_ocr_llm import _call_ocr, _parse_ocr_response, init_ocr_client

logger = logging.getLogger(__name__)

# Threshold for upscaling small images
UPSCALE_THRESH = 60
PREPROCESS_LOG_DIR = Path("output/ocr_debug_images")
PREPROCESS_LOG_MAX_IMAGES = 20
DEBUG_OCR_LOGS_DIR = Path("output/ocr_debug_images")

# Logging globals
_preprocess_log_index = 0
_debug_image_counter = 0


def register_helmet(
    helmets: List[dict],
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    debug: bool = False,
) -> List[dict]:
    # Loop over helmet crops, preprocess each, run OCR, parse results, return list of {track_id, helmet_number, ocr_conf}.
    results = []

    for helmet in helmets:
        img: np.ndarray = helmet["image"]  # numpy array (BGR crop)
        bbox = helmet["bbox"]
        tid = int(helmet.get("track_id", -1))

        if tid == -1:
            continue

        try:
            processed_img = preprocess_image(img)
            raw_text = _call_ocr(helmet_ocr_llm._ocr_client, processed_img, prompt, model, timeout)
            number_str, ocr_conf = _parse_ocr_response(raw_text)

            crop_h, crop_w = img.shape[:2]
            logger.debug(
                "OCR RESULT: track_id=%d, crop=%dx%d, extracted='%s', conf=%.1f%%, raw='%s'",
                tid, crop_w, crop_h, number_str, ocr_conf, raw_text,
            )

            if debug:
                _save_ocr_debug_images(
                    original=img,
                    preprocessed=processed_img,
                    track_id=tid,
                    raw_text=raw_text,
                    digits=number_str,
                    ocr_conf=ocr_conf,
                )

            if debug and number_str:
                logger.debug(f"Number accepted: {number_str} for track_id: {tid}")

        except TimeoutError:
            logger.warning(f"OCR timed out for track_id {tid}")
            number_str = ""
            ocr_conf = 0.0
            raw_text = "timeout"
        except Exception as e:
            logger.warning(f"OCR failed for track_id {tid}: {type(e).__name__}({e})")
            number_str = ""
            ocr_conf = 0.0
            raw_text = str(e)

        # Collect the result regardless of success/failure
        results.append({
            "track_id": tid,
            "bbox": bbox,
            "helmet_number": number_str,
            "ocr_conf": ocr_conf,
        })

    return results


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert BGR to gray, upscale if too small, apply Gaussian blur then unsharp mask, return processed BGR.
    if image.shape[0] < UPSCALE_THRESH or image.shape[1] < UPSCALE_THRESH:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _log_preprocessing_pair(
    before: np.ndarray,
    after: np.ndarray,
    track_id: int,
    raw_text: str,
    digits: str,
    ocr_conf: float,
) -> None:
    # Save a side-by-side before/after preprocessing image with metadata footer to the preprocess log directory.
    global _preprocess_log_index
    PREPROCESS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    before_bgr = before if before.ndim == 3 else cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    after_bgr = after if after.ndim == 3 else cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
    target_h = max(before_bgr.shape[0], after_bgr.shape[0])
    before_panel = _resize_to_height(before_bgr, target_h)
    after_panel = _resize_to_height(after_bgr, target_h)
    separator = np.full((target_h, 8, 3), 32, dtype=np.uint8)
    combined = np.hstack((before_panel, separator, after_panel))
    footer_h = 88
    footer = np.full((footer_h, combined.shape[1], 3), 18, dtype=np.uint8)
    raw_label = raw_text or "<none>"
    cv2.putText(footer, f"track_id: {track_id}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, f"raw: {raw_label[:140]}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(footer, f"digits: {digits or '<empty>'}   conf: {ocr_conf:.1f}", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    combined = np.vstack((combined, footer))
    file_index = _preprocess_log_index % PREPROCESS_LOG_MAX_IMAGES
    output_path = PREPROCESS_LOG_DIR / f"preprocess_{file_index:02d}.png"
    cv2.imwrite(str(output_path), combined)
    _preprocess_log_index += 1


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    # Resize an image to a target height while preserving aspect ratio.
    if image.shape[0] == target_h:
        return image
    scale = target_h / image.shape[0]
    target_w = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _save_ocr_debug_imagegeners(
    original: np.ndarray,
    preprocessed: np.ndarray,
    track_id: int,
    raw_text: str,
    digits: str,
    ocr_conf: float,
) -> None:
    # Save a debug overlay image showing original vs preprocessed crop with OCR metadata footer.
    global _debug_image_counter
    DEBUG_OCR_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    orig_bgr = original if original.ndim == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    proc_bgr = preprocessed if preprocessed.ndim == 3 else cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    target_h = max(orig_bgr.shape[0], proc_bgr.shape[0])
    orig_panel = _resize_to_height(orig_bgr, target_h)
    proc_panel = _resize_to_height(proc_bgr, target_h)
    separator = np.full((target_h, 16, 3), 80, dtype=np.uint8)
    combined = np.hstack((orig_panel, separator, proc_panel))
    footer_h = 120
    footer = np.full((footer_h, combined.shape[1], 3), 10, dtype=np.uint8)
    raw_label = raw_text or "<none>"
    cv2.putText(footer, f"track_id: {track_id}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, f"raw OCR: {raw_label[:180]}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(footer, f"digits: {digits or '<empty>'}   conf: {ocr_conf:.1f}%", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if digits else (0, 180, 180), 1, cv2.LINE_AA)
    orig_h, orig_w = original.shape[:2]
    proc_h, proc_w = preprocessed.shape[:2]
    cv2.putText(footer, f"orig: {orig_w}x{orig_h}, proc: {proc_w}x{proc_h}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    combined = np.vstack((combined, footer))
    output_path = DEBUG_OCR_LOGS_DIR / f"ocr_debug_{_debug_image_counter:04d}_tid{track_id}.png"
    cv2.imwrite(str(output_path), combined)
    _debug_image_counter += 1

"""
Helmet OCR processing module.

Processes cropped helmet images using PaddleOCR to extract helmet numbers.
Designed to work with the multiprocessing architecture in ocr_worker.py.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Apply Paddle runtime workarounds before importing paddleocr in the worker process.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from paddleocr import PaddleOCR

# Threshold for upscaling small images
UPSCALE_THRESH = 60
PREPROCESS_LOG_DIR = Path("output/ocr_preprocess_logs")
PREPROCESS_LOG_MAX_IMAGES = 20
DEBUG_OCR_LOGS_DIR = Path("output/ocr_debug_images")

# Configure logging
logger = logging.getLogger(__name__)

# Global PaddleOCR instance - initialized once per process
_ocr: Optional[PaddleOCR] = None
_preprocess_log_index = 0
_debug_image_counter = 0


def _get_ocr_instance() -> PaddleOCR:
    """Lazy initialization of PaddleOCR instance."""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr


def register_helmet(helmets: List[dict], debug: bool = False) -> List[dict]:
    """
    Process a list of helmet dicts and extract helmet numbers via OCR.

    Args:
        helmets: List of helmet dicts with keys 'image', 'bbox', 'conf', 'track_id'
        debug: If True, prints debug information about detected numbers

    Returns:
        List of result dicts with keys 'track_id', 'bbox', 'helmet_number', 'ocr_conf'

    Note:
        - track_id == -1 items are skipped (untracked detections)
        - Empty string and 0.0 confidence returned when OCR fails or finds no digits
    """
    results = []
    ocr = _get_ocr_instance()

    for helmet in helmets:
        img: np.ndarray = helmet['image']  # numpy array (BGR crop)
        bbox = helmet['bbox']
        tid = int(helmet.get('track_id', -1))

        if tid == -1:
            continue

        try:
            processed_img = preprocess_image(img)
            raw = ocr.predict(processed_img)

            number_str, ocr_conf, raw_texts, raw_scores = _extract_digits_from_ocr(raw)
            crop_h, crop_w = img.shape[:2]
            print(
                f"[OCR RESULT] track_id={tid}, crop={crop_w}x{crop_h}, "
                f"extracted='{number_str}', conf={ocr_conf:.1f}%, raw_texts={raw_texts}"
            )

            # Save detailed debug images for every frame (not just successful extractions)
            _save_ocr_debug_images(
                original=img,
                preprocessed=processed_img,
                track_id=tid,
                raw_texts=raw_texts,
                raw_scores=raw_scores,
                digits=number_str,
                ocr_conf=ocr_conf,
            )

            if debug and number_str:
                logger.debug(f"Number accepted: {number_str} for track_id: {tid}")

        except Exception as e:
            logger.warning(
                f"OCR failed for track_id {tid}: "
                f"{type(e).__name__}({e})"
            )
            number_str = ""
            ocr_conf = 0.0

        results.append({
            'track_id': tid,
            'bbox': bbox,
            'helmet_number': number_str,
            'ocr_conf': ocr_conf,
        })

    return results


def _extract_digits_from_ocr(
    raw: List[dict]
) -> tuple[str, float, List[str], List[float]]:
    """
    Extract digit-only text and confidence from PaddleOCR output.

    Args:
        raw: Raw output from PaddleOCR.predict()

    Returns:
        Tuple of (digit_string, average_confidence_percentage)
        Returns ("", 0.0) if no valid digits found
    """
    number_str = ""
    ocr_conf = 0.0
    valid_texts: List[str] = []
    valid_confs: List[float] = []
    raw_texts_all: List[str] = []
    raw_scores_all: List[float] = []


    for res in raw:
        if isinstance(res, dict):
            rec_texts = res.get("rec_texts") or []
            rec_scores = res.get("rec_scores") or []
        else:
            rec_texts = getattr(res, "rec_texts", []) or []
            rec_scores = getattr(res, "rec_scores", []) or []



        for text, score in zip(rec_texts, rec_scores):
            text_str = str(text).strip()
            raw_texts_all.append(text_str)
            raw_scores_all.append(float(score))
            if not text_str:
                continue
            # Filter out non-digit characters
            digits = "".join(ch for ch in text_str if ch.isdigit())
            if not digits:
                continue
            valid_texts.append(digits)
            valid_confs.append(float(score))

    if valid_texts:
        number_str = "".join(valid_texts).strip()
        ocr_conf = (sum(valid_confs) / len(valid_confs)) * 100.0

    print("Extracted digit from OCR: ", number_str)

    return number_str, ocr_conf, raw_texts_all, raw_scores_all

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped helmet image for OCR.

    1. Convert to grayscale
    2. Upscale if dimensions < UPSCALE_THRESH (60px) using CUBIC interpolation
    3. Apply sharpening via unsharp masking (original - gaussian_blur)
    4. Convert back to BGR for PaddleOCR input format

    Args:
        image: Input numpy array in BGR format

    Returns:
        Preprocessed image as BGR numpy array ready for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale if image is too small
    if image.shape[0] < UPSCALE_THRESH or image.shape[1] < UPSCALE_THRESH:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Unsharp masking for sharpening
    # Creates a blurred copy and subtracts it from original to enhance edges
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)

    # Convert back to BGR for PaddleOCR input format
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _log_preprocessing_pair(
    before: np.ndarray,
    after: np.ndarray,
    track_id: int,
    raw_texts: List[str],
    raw_scores: List[float],
    digits: str,
    ocr_conf: float,
) -> None:
    """Write one rotating debug image that shows raw and preprocessed crops side by side."""
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

    # cv2.putText(combined, "before", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    #cv2.putText(
    #    combined,
    #    "after",
    #    cv2.FONT_HERSHEY_SIMPLEX,
    #    0.7,
    #    (0, 255, 255),
    #    2,
    #    cv2.LINE_AA,
    #)

    raw_pairs = [f"{text or '<empty>'} ({score:.2f})" for text, score in zip(raw_texts, raw_scores)]
    raw_label = ", ".join(raw_pairs) if raw_pairs else "<none>"

    cv2.putText(footer, f"track_id: {track_id}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, f"raw: {raw_label[:140]}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"digits: {digits or '<empty>'}   conf: {ocr_conf:.1f}",
        (8, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack((combined, footer))

    file_index = _preprocess_log_index % PREPROCESS_LOG_MAX_IMAGES
    output_path = PREPROCESS_LOG_DIR / f"preprocess_{file_index:02d}.png"
    cv2.imwrite(str(output_path), combined)
    _preprocess_log_index += 1


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    """Resize an image to the target height while keeping aspect ratio."""
    if image.shape[0] == target_h:
        return image

    scale = target_h / image.shape[0]
    target_w = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _save_ocr_debug_images(
    original: np.ndarray,
    preprocessed: np.ndarray,
    track_id: int,
    raw_texts: List[str],
    raw_scores: List[float],
    digits: str,
    ocr_conf: float,
) -> None:
    """Save detailed debug images showing OCR pipeline stages."""
    global _debug_image_counter

    DEBUG_OCR_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to BGR if needed (grayscale -> BGR)
    orig_bgr = original if original.ndim == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    proc_bgr = preprocessed if preprocessed.ndim == 3 else cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

    # Resize to common height for side-by-side display
    target_h = max(orig_bgr.shape[0], proc_bgr.shape[0])
    orig_panel = _resize_to_height(orig_bgr, target_h)
    proc_panel = _resize_to_height(proc_bgr, target_h)

    # Create combined image: original | preprocessed
    separator = np.full((target_h, 16, 3), 80, dtype=np.uint8)
    combined = np.hstack((orig_panel, separator, proc_panel))

    footer_h = 120
    footer = np.full((footer_h, combined.shape[1], 3), 10, dtype=np.uint8)

    # Raw OCR output info
    raw_pairs = [f"{text or '<empty>'} ({score:.2f})" for text, score in zip(raw_texts, raw_scores)]
    raw_label = ", ".join(raw_pairs) if raw_pairs else "<none>"

    cv2.putText(footer, f"track_id: {track_id}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, f"raw OCR: {raw_label[:180]}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"digits: {digits or '<empty>'}   conf: {ocr_conf:.1f}%",
        (10, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if digits else (0, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Image dimensions
    orig_h, orig_w = original.shape[:2]
    proc_h, proc_w = preprocessed.shape[:2]
    cv2.putText(footer, f"orig: {orig_w}x{orig_h}, proc: {proc_w}x{proc_h}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    combined = np.vstack((combined, footer))

    # Save with unique filename
    output_path = DEBUG_OCR_LOGS_DIR / f"ocr_debug_{_debug_image_counter:04d}_tid{track_id}.png"
    cv2.imwrite(str(output_path), combined)
    _debug_image_counter += 1

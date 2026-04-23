"""
Helmet OCR processing module.

Processes cropped helmet images using PaddleOCR to extract helmet numbers.
Designed to work with the multiprocessing architecture in ocr_worker.py.
"""

import logging
import os
import re
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
PREPROCESS_LOG_MAX_IMAGES = 20
DEBUG_OCR_ORIGINAL_DIR = Path("output/ocr_debug_original")
DEBUG_OCR_PROCESSED_DIR = Path("output/ocr_debug_processed")

# Configure logging
logger = logging.getLogger(__name__)

# Global PaddleOCR instance - initialized once per process
_ocr: Optional[PaddleOCR] = None
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

            # Save original and preprocessed crops in separate rotating folders.
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


def _save_ocr_debug_images(
    original: np.ndarray,
    preprocessed: np.ndarray,
    track_id: int,
    raw_texts: List[str],
    raw_scores: List[float],
    digits: str,
    ocr_conf: float,
) -> None:
    """Save original and preprocessed OCR crops in separate rotating folders."""
    global _debug_image_counter

    DEBUG_OCR_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_OCR_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    orig_bgr = original if original.ndim == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    proc_bgr = preprocessed if preprocessed.ndim == 3 else cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

    file_index = _debug_image_counter % PREPROCESS_LOG_MAX_IMAGES
    metadata = _format_debug_suffix(track_id, raw_texts, raw_scores, digits, ocr_conf)

    original_path = DEBUG_OCR_ORIGINAL_DIR / f"ocr_orig_{file_index:02d}_tid{track_id}{metadata}.png"
    processed_path = DEBUG_OCR_PROCESSED_DIR / f"ocr_proc_{file_index:02d}_tid{track_id}{metadata}.png"

    cv2.imwrite(str(original_path), orig_bgr)
    cv2.imwrite(str(processed_path), proc_bgr)
    _debug_image_counter += 1


def _format_debug_suffix(
    track_id: int,
    raw_texts: List[str],
    raw_scores: List[float],
    digits: str,
    ocr_conf: float,
) -> str:
    raw_pairs = [f"{text or 'empty'}-{score:.2f}" for text, score in zip(raw_texts, raw_scores)]
    raw_label = "_".join(raw_pairs) if raw_pairs else "none"
    raw_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_label)[:80]
    digits_label = digits if digits else "empty"
    return f"_digits-{digits_label}_conf-{ocr_conf:.1f}_raw-{raw_label}"

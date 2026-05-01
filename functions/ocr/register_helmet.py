"""
Helmet OCR processing module.

Processes cropped helmet images via GLM-OCR (LM Studio) to extract helmet numbers.
Designed to work with the multiprocessing architecture in ocr_worker.py.
"""

import logging
import base64
import re
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

# Threshold for upscaling small images
UPSCALE_THRESH = 60
PREPROCESS_LOG_DIR = Path("output/ocr_debug_images")
PREPROCESS_LOG_MAX_IMAGES = 20
DEBUG_OCR_LOGS_DIR = Path("output/ocr_debug_images")

# Configure logging
logger = logging.getLogger(__name__)

# Lazy-init client for spawned worker process
_ocr_client: Optional[Any] = None
_ocr_model: Optional[str] = None
_preprocess_log_index = 0
_debug_image_counter = 0


def init_ocr_client(base_url: str, model: str) -> None:
    """Initialize OpenAI-compatible client for spawned worker process."""
    global _ocr_client, _ocr_model

    from openai import OpenAI

    _ocr_client = OpenAI(
        base_url=base_url,
        api_key="lm-studio",
    )
    _ocr_model = model


def _parse_ocr_response(raw_text: str) -> tuple[str, float]:
    """
    Extract digits and confidence from GLM-OCR response.

    Expected format: NUMBER|CONFIDENCE (e.g. '42|0.95' or 'NONE|0.0')
    Falls back to old cleanliness-based parsing if pipe-delimited format not found.

    Returns:
        Tuple of (digit_string, confidence_percentage)
    """
    if not raw_text or raw_text.strip().lower() == "unknown":
        return "", 0.0

    text = raw_text.strip()

    # Try pipe-delimited format: NUMBER|CONFIDENCE
    parts = text.split("|")
    if len(parts) >= 2:
        number_part = parts[0].strip()
        conf_part = parts[1].strip()

        # Handle NONE/UNKNOWN responses
        if number_part.lower() in ("none", "unknown"):
            return "", 0.0

        digits = re.sub(r"\D", "", number_part)
        if not digits:
            return "", 0.0

        try:
            conf = float(conf_part)
            # Convert 0-1 scale to percentage
            confidence = (conf * 100.0) if conf <= 1.0 else min(conf, 100.0)
        except ValueError:
            # If confidence can't be parsed, fall back to cleanliness ratio
            digits = re.sub(r"\D", "", text)
            if not digits:
                return "", 0.0
            confidence = (len(digits) / max(len(text), 1)) * 100.0

        return digits, confidence

    # Fallback: old cleanliness-based parsing for non-pipe responses
    digits = re.sub(r"\D", "", text)
    if not digits:
        return "", 0.0

    confidence = (len(digits) / max(len(text), 1)) * 100.0
    return digits, confidence


def register_helmet(
    helmets: List[dict],
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    debug: bool = False,
) -> List[dict]:
    """
    Process a list of helmet dicts and extract helmet numbers via GLM-OCR.

    Args:
        helmets: List of helmet dicts with keys 'image', 'bbox', 'conf', 'track_id'
        base_url: LM Studio API base URL
        model: Model name for API call
        prompt: Prompt to send with image
        timeout: Seconds to wait for response before giving up
        debug: If True, prints debug information about detected numbers

    Returns:
        List of result dicts with keys 'track_id', 'bbox', 'helmet_number', 'ocr_conf'
    """
    global _ocr_client, _ocr_model

    if _ocr_client is None:
        init_ocr_client(base_url, model)

    results = []

    for helmet in helmets:
        img: np.ndarray = helmet["image"]  # numpy array (BGR crop)
        bbox = helmet["bbox"]
        tid = int(helmet.get("track_id", -1))

        if tid == -1:
            continue

        try:
            processed_img = preprocess_image(img)
            raw_text = _call_ocr(_ocr_client, processed_img, prompt, model, timeout)
            number_str, ocr_conf = _parse_ocr_response(raw_text)

            crop_h, crop_w = img.shape[:2]
            logger.debug(
                "OCR RESULT: track_id=%d, crop=%dx%d, extracted='%s', conf=%.1f%%, raw='%s'",
                tid, crop_w, crop_h, number_str, ocr_conf, raw_text,
            )

            # Save detailed debug images
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
            logger.warning(
                f"OCR failed for track_id {tid}: "
                f"{type(e).__name__}({e})"
            )
            number_str = ""
            ocr_conf = 0.0
            raw_text = str(e)

        results.append({
            "track_id": tid,
            "bbox": bbox,
            "helmet_number": number_str,
            "ocr_conf": ocr_conf,
        })

    return results


def _call_ocr(client, image_bgr: np.ndarray, prompt: str, model: str, timeout: float) -> str:
    """
    Call GLM-OCR API with a preprocessed helmet crop.

    Converts BGR -> RGB -> JPEG base64 -> HTTP request.
    Returns raw text response from model.
    """
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Encode as JPEG base64
    ok, buffer = cv2.imencode(".jpg", image_rgb)
    if not ok:
        return "unknown"

    image_b64 = base64.b64encode(buffer).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            timeout=timeout,
        )

        return (response.choices[0].message.content or "").strip()

    except Exception:
        return "unknown"


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped helmet image for OCR.

    1. Convert to grayscale
    2. Upscale if dimensions < UPSCALE_THRESH (60px) using CUBIC interpolation
    3. Apply sharpening via unsharp masking (original - gaussian_blur)
    4. Convert back to BGR for API input format

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
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)

    # Convert back to BGR for API input format
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _log_preprocessing_pair(
    before: np.ndarray,
    after: np.ndarray,
    track_id: int,
    raw_text: str,
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

    raw_label = raw_text or "<none>"

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
    raw_text: str,
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

    raw_label = raw_text or "<none>"

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

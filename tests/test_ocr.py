
import os
import re
import sys

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import cv2
from paddleocr import PaddleOCR, TextRecognition


def build_variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variants = [("original", image)]

    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(("gray_upscaled", cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)))

    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))
    variants.append(("otsu_inverted", cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR)))

    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.5)
    sharpened = cv2.addWeighted(upscaled, 1.8, blurred, -0.8, 0)
    variants.append(("sharpened", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))

    return variants


def main():
    image_path = "../output/ocr_preprocess_logs/preprocess_03.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    print("Detected numbers:")
    found = False

    try:
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
        )

        for variant_name, variant_img in build_variants(image):
            results = ocr.predict(
                input=variant_img,
                text_rec_score_thresh=0.0,
            )
            for result in results:
                if isinstance(result, dict):
                    rec_texts = result.get("rec_texts") or []
                    rec_scores = result.get("rec_scores") or []
                else:
                    rec_texts = getattr(result, "rec_texts", []) or []
                    rec_scores = getattr(result, "rec_scores", []) or []

                for text, confidence in zip(rec_texts, rec_scores):
                    numbers = re.findall(r"\d+", str(text))
                    for number in numbers:
                        print(f"  {number}  (confidence: {float(confidence):.1%}, variant: {variant_name}, mode: detect+rec)")
                        found = True
    except Exception as exc:
        print(f"Full OCR pipeline failed, falling back to recognition-only mode: {exc}")

    best_raw_text = ""
    best_confidence = 0.0
    recognizer = TextRecognition(enable_mkldnn=False, device="cpu")
    for variant_name, variant_img in build_variants(image):
        results = recognizer.predict(input=variant_img, batch_size=1)
        for result in results:
            if isinstance(result, dict):
                payload = result.get("res") or {}
            else:
                payload = getattr(result, "res", None) or {}

            text = str(payload.get("rec_text", "") or "").strip()
            confidence = float(payload.get("rec_score", 0.0))
            if confidence > best_confidence:
                best_confidence = confidence
                best_raw_text = text

            numbers = re.findall(r"\d+", text)
            for number in numbers:
                print(f"  {number}  (confidence: {confidence:.1%}, variant: {variant_name}, mode: rec-only)")
                found = True

    if not found:
        print("  No numbers found.")
        print(f"Best raw text: '{best_raw_text}' (confidence: {best_confidence:.1%})")

if __name__ == "__main__":
    main()

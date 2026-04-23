from __future__ import annotations

# CLI examples:
# python tests/test_paddle_ocr.py output/ocr_debug_images
# python tests/test_paddle_ocr.py output/ocr_debug_images --limit 100 --print-every 10
# python tests/test_paddle_ocr.py output/ocr_debug_images --pause-ms 50

import argparse
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0" # Disabled because isn't included in original.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
from paddleocr import PaddleOCR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_REC_MODEL = "PP-OCRv5_mobile_rec"
RESULTS_DIR = Path("output/ocr_benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential benchmark for PaddleOCR helmet OCR on a directory of crops."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing OCR crop images.",
    )
    parser.add_argument(
        "--det-model",
        default=DEFAULT_DET_MODEL,
        help=f"PaddleOCR text detection model. Default: {DEFAULT_DET_MODEL}",
    )
    parser.add_argument(
        "--rec-model",
        default=DEFAULT_REC_MODEL,
        help=f"PaddleOCR text recognition model. Default: {DEFAULT_REC_MODEL}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process.",
    )
    parser.add_argument(
        "--pause-ms",
        type=float,
        default=0.0,
        help="Optional pause between images to better mimic a live queue.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print per-image progress every N images.",
    )
    return parser.parse_args()


def load_image_paths(image_dir: Path, limit: int | None) -> list[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_model(det_model: str, rec_model: str) -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        device="cpu",
    )


def read_image_bgr(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape {image.shape} for {path}")


def extract_digits_from_ocr(raw) -> tuple[str, float, list[str], list[float]]:
    number_str = ""
    ocr_conf = 0.0
    valid_texts: list[str] = []
    valid_confs: list[float] = []
    raw_texts_all: list[str] = []
    raw_scores_all: list[float] = []

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
            digits = "".join(ch for ch in text_str if ch.isdigit())
            if not digits:
                continue
            valid_texts.append(digits)
            valid_confs.append(float(score))

    if valid_texts:
        number_str = "".join(valid_texts).strip()
        ocr_conf = (sum(valid_confs) / len(valid_confs)) * 100.0

    return number_str, ocr_conf, raw_texts_all, raw_scores_all


def run_single_image(image_bgr, ocr) -> dict:
    prep_ms = 0.0

    infer_start = time.perf_counter()
    raw = ocr.predict(image_bgr)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    decode_start = time.perf_counter()
    digits_only, ocr_conf, raw_texts, raw_scores = extract_digits_from_ocr(raw)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    raw_text = " | ".join(
        f"{text or '<empty>'} ({score:.2f})" for text, score in zip(raw_texts, raw_scores)
    ).strip()
    parse_ok = digits_only.isdigit() if digits_only else False

    return {
        "raw_text": raw_text,
        "digits_only": digits_only,
        "unknown": not bool(digits_only),
        "parse_ok": parse_ok,
        "ocr_conf": ocr_conf,
        "prep_ms": prep_ms,
        "infer_ms": infer_ms,
        "decode_ms": decode_ms,
        "total_ms": prep_ms + infer_ms + decode_ms,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def format_summary(results: list[dict], elapsed_s: float) -> str:
    totals = sorted(r["total_ms"] for r in results)
    infers = sorted(r["infer_ms"] for r in results)
    prep = sorted(r["prep_ms"] for r in results)
    decode = sorted(r["decode_ms"] for r in results)

    parse_ok = sum(1 for r in results if r["parse_ok"])
    unknown = sum(1 for r in results if r["unknown"])
    with_digits = sum(1 for r in results if r["digits_only"])
    lines = [
        "Summary",
        f"images processed: {len(results)}",
        f"wall time: {elapsed_s:.2f}s",
        f"throughput: {len(results) / elapsed_s:.2f} img/s" if elapsed_s > 0 else "throughput: inf",
        f"strict digits-only outputs: {parse_ok}/{len(results)} ({(100.0 * parse_ok / len(results)):.1f}%)",
        f"outputs containing any digits: {with_digits}/{len(results)} ({(100.0 * with_digits / len(results)):.1f}%)",
        f"empty outputs: {unknown}/{len(results)} ({(100.0 * unknown / len(results)):.1f}%)",
        "latency total ms: "
        f"mean={statistics.mean(totals):.1f} "
        f"median={statistics.median(totals):.1f} "
        f"p95={percentile(totals, 0.95):.1f} "
        f"max={max(totals):.1f}",
        "latency infer ms: "
        f"mean={statistics.mean(infers):.1f} "
        f"median={statistics.median(infers):.1f} "
        f"p95={percentile(infers, 0.95):.1f} "
        f"max={max(infers):.1f}",
        "latency prep ms: "
        f"mean={statistics.mean(prep):.1f} "
        f"median={statistics.median(prep):.1f}",
        "latency decode ms: "
        f"mean={statistics.mean(decode):.1f} "
        f"median={statistics.median(decode):.1f}",
    ]
    return "\n".join(lines)


def format_result_details(results: list[dict]) -> str:
    lines = ["Per-image outputs"]
    for result in results:
        lines.extend([
            f"file: {result['image_name']}",
            f"raw_text: {result['raw_text']}",
            f"digits_only: {result['digits_only']}",
            f"ocr_conf: {result['ocr_conf']:.1f}",
            f"total_ms: {result['total_ms']:.1f} infer_ms: {result['infer_ms']:.1f}",
            "",
        ])
    return "\n".join(lines)


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("._") or "model"


def write_results(model_name: str, image_dir: Path, summary: str, details: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{timestamp}_{sanitize_name(model_name)}.txt"
    header = [
        f"model: {model_name}",
        f"image_dir: {image_dir}",
        f"timestamp: {timestamp}",
        "",
    ]
    output_path.write_text("\n".join(header) + summary + "\n\n" + details + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths(args.image_dir, args.limit)

    if not image_paths:
        raise RuntimeError(f"No image files found in {args.image_dir}")

    print(f"loading model: PaddleOCR det={args.det_model} rec={args.rec_model}")
    ocr = load_model(args.det_model, args.rec_model)

    results = []
    wall_start = time.perf_counter()

    for idx, path in enumerate(image_paths, start=1):
        image_bgr = read_image_bgr(path)
        result = run_single_image(image_bgr=image_bgr, ocr=ocr)
        result["image_name"] = path.name
        results.append(result)

        if args.print_every > 0 and (idx == 1 or idx % args.print_every == 0 or idx == len(image_paths)):
            print(
                f"[{idx}/{len(image_paths)}] {path.name} "
                f"raw={result['raw_text']!r} digits={result['digits_only']!r} "
                f"conf={result['ocr_conf']:.1f} "
                f"total_ms={result['total_ms']:.1f} infer_ms={result['infer_ms']:.1f}"
            )

        if args.pause_ms > 0:
            time.sleep(args.pause_ms / 1000.0)

    elapsed_s = time.perf_counter() - wall_start
    summary = format_summary(results, elapsed_s)
    details = format_result_details(results)
    print(f"\n{summary}")
    model_name = f"PaddleOCR_det={args.det_model}_rec={args.rec_model}"
    result_path = write_results(model_name, args.image_dir, summary, details)
    print(f"\nresults saved to: {result_path}")


if __name__ == "__main__":
    main()

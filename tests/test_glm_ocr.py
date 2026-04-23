from __future__ import annotations

# CLI examples:
# python tests/test_glm_ocr.py output/ocr_debug_original
# python tests/test_glm_ocr.py output/ocr_debug_original --device cuda
# python tests/test_glm_ocr.py output/ocr_debug_original --limit 100 --print-every 10
# python tests/test_glm_ocr.py output/ocr_debug_original --pause-ms 50
# python tests/test_glm_ocr.py output/ocr_debug_original --model zai-org/GLM-OCR

import argparse
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Text Recognition:"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RESULTS_DIR = Path("output/ocr_benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential benchmark for GLM-OCR on a directory of crops."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing OCR crop images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for each image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Generation limit per image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection. 'auto' prefers CUDA when available.",
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


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return requested


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


def load_model(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)

    model_kwargs = {
        "device_map": "auto" if device == "cuda" else device,
        "torch_dtype": "auto",
    }
    if device == "cuda":
        quant_config = BitsAndBytesConfig()
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_id,
        **model_kwargs,
    )
    model.eval()
    return processor, model


def get_model_input_device(model, requested_device: str):
    if requested_device == "cuda":
        return getattr(model, "device", torch.device("cuda"))
    return getattr(model, "device", torch.device("cpu"))


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def read_image_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image shape {image.shape} for {path}")


def run_single_image(
    image_path: Path,
    processor,
    model,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": str(image_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    prep_start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(get_model_input_device(model, device))
    inputs.pop("token_type_ids", None)
    sync_device(device)
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    infer_start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sync_device(device)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    decode_start = time.perf_counter()
    prompt_len = inputs["input_ids"].shape[1]
    raw_text = processor.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=False,
    ).strip()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    digits_only = "".join(ch for ch in raw_text if ch.isdigit())
    unknown = raw_text.strip().upper() == "UNKNOWN"
    parse_ok = bool(re.fullmatch(r"\d+", raw_text))

    return {
        "raw_text": raw_text,
        "digits_only": digits_only,
        "unknown": unknown,
        "parse_ok": parse_ok,
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
        f"unknown outputs: {unknown}/{len(results)} ({(100.0 * unknown / len(results)):.1f}%)",
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
    device = choose_device(args.device)
    image_paths = load_image_paths(args.image_dir, args.limit)

    if not image_paths:
        raise RuntimeError(f"No image files found in {args.image_dir}")

    print(f"loading model: {args.model}")
    print(f"device: {device}")
    processor, model = load_model(args.model, device)

    results = []
    wall_start = time.perf_counter()

    for idx, path in enumerate(image_paths, start=1):
        result = run_single_image(
            image_path=path,
            processor=processor,
            model=model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        result["image_name"] = path.name
        results.append(result)

        if args.print_every > 0 and (idx == 1 or idx % args.print_every == 0 or idx == len(image_paths)):
            print(
                f"[{idx}/{len(image_paths)}] {path.name} "
                f"raw={result['raw_text']!r} digits={result['digits_only']!r} "
                f"total_ms={result['total_ms']:.1f} infer_ms={result['infer_ms']:.1f}"
            )

        if args.pause_ms > 0:
            time.sleep(args.pause_ms / 1000.0)

    elapsed_s = time.perf_counter() - wall_start
    summary = format_summary(results, elapsed_s)
    details = format_result_details(results)
    print(f"\n{summary}")
    result_path = write_results(args.model, args.image_dir, summary, details)
    print(f"\nresults saved to: {result_path}")


if __name__ == "__main__":
    main()

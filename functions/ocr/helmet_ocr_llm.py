import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

DEFAULT_OCR_MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_OCR_MODEL_DIR = Path("models/ocr_model")
MODEL_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx")
logger = logging.getLogger(__name__)
_ocr_client: Optional[Any] = None
_ocr_model: Optional[str] = None
_DEFAULT_OCR_PROMPT = """Identify the 3-digit helmet number in this image.

Return EXACTLY this format, nothing else:
NUMBER

Where:
- NUMBER is exactly 3 digits (000-999).
- Return only the 3 digits. No words, punctuation, or extra text."""
_OCR_SYSTEM_PROMPT = "3 digits only"


def _has_model_weights(model_path: Path) -> bool:
    # Return True if dir contains any model weights
    return any(model_path.glob(pattern) for pattern in MODEL_WEIGHT_PATTERNS)


def _looks_like_default_ocr_path(model: str, model_path: Path) -> bool:
    # Return True if *model* refers to default model dir
    normalized = model.replace("\\", "/").strip("/")
    return normalized in {
        "models/ocr_model",
        "models/ocr_models",
        str(DEFAULT_OCR_MODEL_DIR).replace("\\", "/"),
    } or model_path.resolve() == DEFAULT_OCR_MODEL_DIR.resolve()


def _download_ocr_model(repo_id: str, target_dir: Path) -> str:
    # Download model from HF Hub to target_dir and return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the OCR model. "
            "Install project requirements, then run again."
        ) from exc
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading OCR model %s to %s", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return str(target_dir)


def _resolve_model_source(model: str) -> tuple[str, bool]:
    # Resolve model identifier to local path, downloading if needed
    """
    Resolve a model identifier to a local path.

    Returns (local_path, local_files_only) with exactly two outcomes:
      - (path, True)  → local path is populated (already on disk or just downloaded)
      - (model_id, False) → nothing on disk; let Transformers handle remote download
    """

    local_path = _resolve_to_local_path(model)

    if _is_local_path_populated(local_path, model):
        return str(local_path), True

    _download_ocr_model(model, local_path)
    return str(local_path), True


def _resolve_to_local_path(model: str) -> Path:
    # Map a model ID or path string to a local Path
    candidate = Path(model).expanduser()

    # Explicit local path provided by user
    if candidate.exists() or candidate.suffix:  # has a file extension → treat as path
        return candidate

    # Default model aliases
    if _looks_like_default_ocr_path(model, candidate):
        return DEFAULT_OCR_MODEL_DIR

    # Default model ID
    if model == DEFAULT_OCR_MODEL_ID:
        return DEFAULT_OCR_MODEL_DIR

    # HF-style repo ID → use default directory
    if "/" in model:
        return DEFAULT_OCR_MODEL_DIR

    # Fallback (shouldn't reach here, but be safe)
    return DEFAULT_OCR_MODEL_DIR


def _is_local_path_populated(path: Path, model: str) -> bool:
    # Return True if the local path already has model weights or an HF cache.
    # Direct weights
    if _has_model_weights(path):
        return True

    # HF cache snapshot (fallback when path is DEFAULT_OCR_MODEL_DIR but model differs)
    cache_path = _hf_cache_snapshot_path(model)
    if cache_path and cache_path.exists():
        logger.info("Using cached local OCR model snapshot: %s", cache_path)
        return True

    return False


def _hf_cache_snapshot_path(model: str) -> Optional[Path]:
    # Find the newest HuggingFace cache snapshot directory for *model*, or None if not cached.
    cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE", "")
    if cache_env:
        hub_cache = Path(cache_env).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        hub_cache = hf_home / "hub"

    cache_dir = hub_cache / f"models--{model.replace('/', '--')}"
    refs_main = cache_dir / "refs" / "main"
    snapshots_dir = cache_dir / "snapshots"

    snapshot_id: Optional[str] = None
    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
    elif snapshots_dir.exists():
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            snapshot_id = snapshots[0].name

    if snapshot_id:
        return snapshots_dir / snapshot_id
    return None


def _warmup_cuda_context() -> None:
    # Run a warmup matmul to pre-compile cuDNN kernels before inference.
    import torch
    if not torch.cuda.is_available():
        return
    a = torch.empty((16, 16), device="cuda", dtype=torch.float16)
    _ = a @ a
    torch.cuda.synchronize()


def init_ocr_client(
    base_url: str,
    model: str,
    load_in_4bit: bool = False,
) -> None:
    # Load the OCR model and processor, place in the correct dtype, and store global _ocr_client.
    global _ocr_client, _ocr_model

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    del base_url

    if not model:
        raise ValueError("OCR model must be a Hugging Face model ID or local path.")

    model_source, local_files_only = _resolve_model_source(model)
    logger.info("OCR model resolved: source=%s local=%s model_id=%r", model_source, local_files_only, model)

    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        _warmup_cuda_context()

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model_kwargs["dtype"] = torch.float16

    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        logger.info("Loading OCR model in 4-bit (requires bitsandbytes)")

    ocr_model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)
    ocr_model = ocr_model.to(device)
    ocr_model.eval()

    #model should load in GPU on default if CUDA is available
    if torch.cuda.is_available():
        logger.info("OCR model loaded on GPU (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("OCR model loaded on CPU")

    _ocr_client = {
        "processor": processor,
        "model": ocr_model,
        "device": device,
    }
    _ocr_model = model_source


def _parse_ocr_response(raw_text: str) -> tuple[str, float]:
    # Return the 3-digit string and 100 confidence if valid
    if not raw_text or raw_text.strip().lower() == "unknown":
        return "", 0.0

    text = raw_text.strip()
    match = re.fullmatch(r"(\d{3})", text)
    if not match:
        logger.debug("Rejected non-strict OCR response: raw=%r", raw_text)
        return "", 0.0

    return match.group(1), 100.0


def _call_ocr(client, image_bgr: np.ndarray, prompt: str, model: str, timeout: float) -> str:
    # Tokenize the helmet crop with the processor, run model.generate(), decode and return the first-line text.
    import torch

    del model, timeout

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    processor = client["processor"]
    ocr_model = client["model"]
    device = client["device"]
    model_dtype = next(ocr_model.parameters()).dtype if hasattr(ocr_model, "parameters") else None

    try:
        combined_prompt = _OCR_SYSTEM_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": combined_prompt},
                ],
            }
        ]

        if hasattr(processor, "apply_chat_template"):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(images=image_pil, text=combined_prompt, return_tensors="pt")

        normalized_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized_inputs[key] = value
                continue
            if device == "cuda" and getattr(value, "is_floating_point", lambda: False)():
                normalized_inputs[key] = value.to(device=device, dtype=model_dtype, non_blocking=True)
            else:
                normalized_inputs[key] = value.to(device=device, non_blocking=(device == "cuda"))
        inputs = normalized_inputs

        with torch.inference_mode():
            outputs = ocr_model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        tokens = outputs[0][prompt_len:] if prompt_len else outputs[0]

        if hasattr(processor, "decode"):
            decoded = processor.decode(tokens, skip_special_tokens=True).strip()
        else:
            decoded = processor.batch_decode([tokens], skip_special_tokens=True)[0].strip()

        first_line = next((line.strip() for line in decoded.splitlines() if line.strip()), "")
        logger.debug("OCR raw decoded=%r first_line=%r", decoded, first_line)
        return first_line
    except Exception as exc:
        import traceback
        logger.warning("Transformers OCR call failed: %s(%s)\n%s", type(exc).__name__, exc, traceback.format_exc())
        return "unknown"

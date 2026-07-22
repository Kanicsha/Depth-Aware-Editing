"""Shared on-disk and in-memory cache helpers for global (non-image) models."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "weights"
MODEL_CACHE_ROOT = REPO_ROOT / ".model_cache"

# Loaded once per Python process; inference per image uses these handles.
GLOBAL_MODELS = {
    "anydoor": {
        "path": WEIGHTS_DIR / "epoch=1-step=8687.ckpt",
        "kind": "disk + RAM (Gradio startup)",
        "image_specific": False,
    },
    "dinov2_vitg14": {
        "path": WEIGHTS_DIR / "dinov2_vitg14_pretrain.pth",
        "kind": "disk + RAM (inside AnyDoor)",
        "image_specific": False,
    },
    "depth_anything_local": {
        "paths": [
            WEIGHTS_DIR / "depth_anything_metric_depth_indoor.pt",
            WEIGHTS_DIR / "depth_anything_vitl14.pth",
        ],
        "kind": "disk (metric-depth CLI paths)",
        "image_specific": False,
    },
    "depth_anything_hf": {
        "id": "LiheYoung/depth-anything-small-hf",
        "kind": "HF cache + RAM (preprocess.depth_pipe)",
        "image_specific": False,
    },
    "sam_vit_huge": {
        "id": "facebook/sam-vit-huge",
        "kind": "HF cache + RAM (preprocess.sam_model)",
        "image_specific": False,
    },
    "sd_2_1_base": {
        "id": "Manojb/stable-diffusion-2-1-base",
        "kind": "HF cache + RAM (FeatureGuidance / null-text)",
        "image_specific": False,
    },
}

# Recomputed per background crop (+ prompt); not global.
IMAGE_SPECIFIC_CACHES = {
    "null_text_embed": REPO_ROOT / "examples/Gradio/null_embed",
    "depth_sam_inference": "recomputed each Analyze/Generate on crop",
    "colligo_clean_plate": "API call per removal mask",
}


def configure_model_cache(repo_root: Path | None = None) -> Path:
    """
    Optionally colocate HF/Torch caches under `<repo>/.model_cache/`.

    Off by default so existing `~/.cache/huggingface` is reused (no re-download).
    Enable with: export DEPTH_EDIT_MODEL_CACHE=1
    """
    root = Path(repo_root or REPO_ROOT).resolve()
    cache_root = root / ".model_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    if os.environ.get("DEPTH_EDIT_MODEL_CACHE", "").strip().lower() in ("1", "true", "yes"):
        hf = cache_root / "huggingface"
        torch_home = cache_root / "torch"
        hf.mkdir(parents=True, exist_ok=True)
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf / "transformers"))
        os.environ.setdefault("TORCH_HOME", str(torch_home))

    return cache_root


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f}{unit}" if unit != "B" else f"{num_bytes}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def verify_global_weight_files() -> list[str]:
    missing: list[str] = []
    for name, spec in GLOBAL_MODELS.items():
        if "path" in spec:
            if not Path(spec["path"]).is_file():
                missing.append(f"{name}: {spec['path']}")
        for p in spec.get("paths", []):
            if not Path(p).is_file():
                missing.append(f"{name}: {p}")
    return missing


def log_model_cache_status(*, device: str | None = None) -> None:
    missing = verify_global_weight_files()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))

    print("[model-cache] Global models (shared across all images in this session):", flush=True)
    for name, spec in GLOBAL_MODELS.items():
        extra = ""
        if "path" in spec and Path(spec["path"]).is_file():
            extra = f" ({_human_size(_dir_size(Path(spec['path'])))})"
        elif "paths" in spec:
            sizes = [p for p in spec["paths"] if Path(p).is_file()]
            if sizes:
                extra = f" ({len(sizes)} local files)"
        print(f"  - {name}: {spec['kind']}{extra}", flush=True)

    print("[model-cache] On-disk cache locations:", flush=True)
    print(f"  - weights/: {_human_size(_dir_size(WEIGHTS_DIR))} at {WEIGHTS_DIR}", flush=True)
    print(f"  - Hugging Face: {_human_size(_dir_size(hf_home))} at {hf_home}", flush=True)
    print(f"  - Torch hub: {_human_size(_dir_size(torch_home))} at {torch_home}", flush=True)
    if device:
        print(f"  - runtime device: {device}", flush=True)

    if missing:
        print("[model-cache] Missing local weight files (run download_weights.sh):", flush=True)
        for m in missing:
            print(f"  ! {m}", flush=True)
    else:
        print("[model-cache] All local weight files present.", flush=True)

    print("[model-cache] Image-specific (not global): null-text embeds, depth/SAM on crop, GenFill.", flush=True)

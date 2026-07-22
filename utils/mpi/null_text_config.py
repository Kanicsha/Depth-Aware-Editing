"""Null-text inversion settings (env-tunable, separate from AnyDoor DDIM steps)."""
from __future__ import annotations

import hashlib
import os
import re

import cv2
import numpy as np


def null_text_ddim_steps() -> int:
    return max(1, int(os.environ.get("NULL_TEXT_DDIM_STEPS", "50")))


def null_text_inner_steps() -> int:
    return max(1, int(os.environ.get("NULL_TEXT_INNER_STEPS", "5")))


def null_text_disk_cache_enabled() -> bool:
    return os.environ.get("NULL_TEXT_DISK_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _prompt_slug(prompt: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w]+", "_", (prompt or "prompt").strip().lower()).strip("_")
    return slug[:max_len] or "prompt"


def crop_cache_key(
    crop_rgb: np.ndarray,
    depth_value: int,
    inv_prompt: str,
    *,
    hash_size: int = 64,
) -> str:
    """Stable cache id for a specific crop + depth + prompt."""
    thumb = cv2.resize(crop_rgb, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    payload = thumb.tobytes() + f"|{int(depth_value)}|{inv_prompt}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{digest}_{_prompt_slug(inv_prompt)}"


def null_text_cache_paths(
    cache_dir: str,
    crop_rgb: np.ndarray,
    depth_value: int,
    inv_prompt: str,
) -> tuple[str, str]:
    key = crop_cache_key(crop_rgb, depth_value, inv_prompt)
    base = os.path.join(cache_dir, key)
    return f"{base}_null_text.pt", f"{base}_init_noise.pt"

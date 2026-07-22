"""Colligo / Firefly v2/images/fill helpers for clean-plate object removal."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROMPT_TEXTURED = (
    "Background inpainting only — remove masked text and fill with continuation of the "
    "existing background. Sample colors, lighting, shadows, blur, grain, gradient, and "
    "texture from the pixels immediately outside the mask and extend them seamlessly. "
    "The result must look like empty background with no text ever present. "
    "Do not invent or add anything: no objects, people, animals, plants, flowers, "
    "shapes, borders, lines, circles, frames, text, letters, numbers, symbols, logos, "
    "icons, watermarks, stickers, decorations, patterns, or new design elements. "
    "Do not change composition, hue, or content outside the mask. No creative generation."
)

PROMPT_FLAT = (
    "Solid flat color fill only. Paint the masked area with the exact same uniform "
    "background color as the paper immediately outside the mask. Plain empty paper or "
    "canvas — no texture, no gradient, no vignette, no pattern, no noise, no tint, "
    "no new hues, no objects, no shapes, no decorative elements."
)

PROMPT_FURNITURE_REMOVAL = (
    "Background inpainting only — remove the masked furniture or object completely. "
    "Fill the masked region with seamless continuation of the floor, wall, and "
    "background visible immediately outside the mask. Match colors, lighting, shadows, "
    "blur, grain, gradient, and texture from surrounding pixels. "
    "The result must look like the removed object was never there. "
    "Do not invent or add anything: no objects, people, animals, plants, furniture, "
    "shapes, borders, lines, text, logos, decorations, patterns, or new design elements. "
    "Do not change composition, hue, or content outside the mask. No creative generation."
)

NEGATIVE_PROMPT = (
    "new objects, people, animals, plants, flowers, furniture, shapes, borders, lines, "
    "circles, text, letters, numbers, symbols, logos, icons, watermark, sticker, "
    "decoration, creative, invented, hallucinated, patterned, textured, gradient, tinted"
)

REPO_TEMPLATE = Path(__file__).resolve().parent / "fill_request.json"
RUNTIME_CONFIG = Path(__file__).resolve().parent / "runtime_config.json"


def load_runtime_colligo_overrides() -> dict[str, Any]:
    """
    Optional hot-tunable settings (no Gradio restart).

    Edit `utils/colligo/runtime_config.json` and click Analyze/Generate again.
    Override path with env COLLIGO_RUNTIME_CONFIG.
    """
    path = Path(os.environ.get("COLLIGO_RUNTIME_CONFIG", str(RUNTIME_CONFIG)))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[colligo] ignoring invalid runtime config {path}: {exc}", flush=True)
        return {}


def normalize_colligo_host(host: str) -> str:
    h = (host or "").strip().rstrip("/")
    if h.endswith("/debug"):
        h = h[: -len("/debug")].rstrip("/")
    return h


def fill_prompt_for_furniture_removal(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> str:
    flat, _ = flat_background_stats(image_bgr, mask)
    if flat:
        base = PROMPT_FLAT
    else:
        base = PROMPT_FURNITURE_REMOVAL
    return f"{base} Avoid: {NEGATIVE_PROMPT}."


def fill_prompt_for_image(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> str:
    return fill_prompt_for_furniture_removal(image_bgr, mask)


def _background_sample_pixels(image_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    m = (mask > 0) if mask is not None else np.zeros((h, w), dtype=bool)
    border = np.zeros((h, w), dtype=bool)
    band = max(3, min(h, w) // 80)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    ring = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8), 1) > 0
    ring &= ~m
    sel = border | ring
    pts = image_bgr[sel]
    if len(pts) < 80:
        pts = image_bgr[~m]
    return pts.reshape(-1, 3) if len(pts) else image_bgr.reshape(-1, 3)


def flat_background_stats(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    std_thresh: float | None = None,
    mean_thresh: float | None = None,
) -> tuple[bool, np.ndarray]:
    std_thresh = float(os.environ.get("FIREFLY_FLAT_STD", std_thresh or 14.0))
    mean_thresh = float(os.environ.get("FIREFLY_FLAT_MEAN", mean_thresh or 188.0))
    pts = _background_sample_pixels(image_bgr, mask).astype(np.float32)
    std = float(np.std(pts, axis=0).max())
    mean_bgr = float(np.mean(pts))
    median_bgr = np.median(pts, axis=0).astype(np.uint8)
    return std <= std_thresh and mean_bgr >= mean_thresh, median_bgr


def solid_background_fill(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, color = flat_background_stats(image_bgr, mask)
    out = image_bgr.copy()
    out[mask > 0] = color
    return out


def paste_inpaint_on_source(
    source_bgr: np.ndarray,
    filled_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if filled_bgr.shape[:2] != source_bgr.shape[:2]:
        filled_bgr = cv2.resize(
            filled_bgr,
            (source_bgr.shape[1], source_bgr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    if mask.shape[:2] != source_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (source_bgr.shape[1], source_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    out = source_bgr.copy()
    out[mask > 0] = filled_bgr[mask > 0]
    return out


def load_fill_template(path: Path | None = None) -> dict[str, Any]:
    if path and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if REPO_TEMPLATE.is_file():
        return json.loads(REPO_TEMPLATE.read_text(encoding="utf-8"))
    return {
        "seeds": [1],
        "size": {"width": 2048, "height": 2048},
        "output": {
            "cai": {"directive": "sign_for_policy_mandate"},
            "storeInputs": False,
            "blendingMask": False,
        },
        "modelVersion": "image3",
        "detailLevel": "full",
        "similarity": 2,
        "inputImage": {
            "source": {"name": "source"},
            "fillArea": {"fillMask": {"name": "mask"}, "invertMask": False},
        },
        "useLegacyMd": False,
        "prompt": PROMPT_FURNITURE_REMOVAL,
    }


def _colligo_api_size(template: dict[str, Any], image_w: int, image_h: int) -> tuple[int, int]:
    preset = (template or {}).get("size") or {}
    aw = int(preset.get("width") or 0)
    ah = int(preset.get("height") or 0)
    if aw > 0 and ah > 0:
        return aw, ah
    return int(image_w), int(image_h)


def build_fill_request(
    *,
    width: int,
    height: int,
    prompt: str | None = None,
    flat: bool = False,
    template: dict[str, Any] | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    tmpl = template or load_fill_template()
    req = dict(tmpl)
    req["seeds"] = [seed]
    aw, ah = _colligo_api_size(tmpl, width, height)
    req["size"] = {"width": aw, "height": ah}
    if prompt is None:
        req["prompt"] = PROMPT_FLAT if flat else PROMPT_FURNITURE_REMOVAL
    else:
        req["prompt"] = prompt
    if "Avoid:" not in req["prompt"]:
        req["prompt"] = f"{req['prompt']} Avoid: {NEGATIVE_PROMPT}."
    req.pop("negativePrompt", None)
    if "similarity" in tmpl:
        default_sim = min(2, int(tmpl.get("similarity", 2)))
        req["similarity"] = min(
            2,
            int(os.environ.get("FIREFLY_SIMILARITY", default_sim)),
        )
    runtime = load_runtime_colligo_overrides()
    if "similarity" in runtime:
        req["similarity"] = min(2, int(runtime["similarity"]))
    if runtime.get("detailLevel"):
        req["detailLevel"] = str(runtime["detailLevel"])
    if runtime.get("modelVersion"):
        req["modelVersion"] = str(runtime["modelVersion"])
    if runtime.get("seed") is not None:
        req["seeds"] = [int(runtime["seed"])]
    if req.get("modelVersion") == "image3":
        req["detailLevel"] = runtime.get("detailLevel") or "full"
    out = dict(req.get("output") or {})
    if os.environ.get("FIREFLY_BLENDING_MASK") is not None:
        out["blendingMask"] = os.environ.get("FIREFLY_BLENDING_MASK", "1") != "0"
    req["output"] = out
    return req


def should_skip_generative_fill(image_bgr: np.ndarray, mask: np.ndarray) -> bool:
    mode = os.environ.get("FIREFLY_FLAT_FILL", "auto").lower()
    if mode in ("0", "false", "no", "off", "never"):
        return False
    if mode in ("1", "true", "yes", "always", "solid"):
        return True
    flat, _ = flat_background_stats(image_bgr, mask)
    return flat


def post_correct_flat_fill(
    source_bgr: np.ndarray,
    filled_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if os.environ.get("FIREFLY_FLAT_CORRECT", "1") == "0":
        return filled_bgr
    flat, _ = flat_background_stats(source_bgr, mask)
    if not flat:
        return filled_bgr
    return solid_background_fill(source_bgr, mask)

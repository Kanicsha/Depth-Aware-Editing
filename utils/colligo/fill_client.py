"""Colligo v2/images/fill HTTP client with Telea/solid fallbacks."""
from __future__ import annotations

import io
import json
import os
from typing import Callable

import cv2
import numpy as np
import requests
from PIL import Image

from utils.colligo.fill_utils import (
    build_fill_request,
    fill_prompt_for_furniture_removal,
    normalize_colligo_host,
    paste_inpaint_on_source,
    post_correct_flat_fill,
    should_skip_generative_fill,
    solid_background_fill,
)


def _resize_png_bytes(arr: np.ndarray, w: int, h: int, *, is_mask: bool) -> bytes:
    buf = io.BytesIO()
    if is_mask:
        Image.fromarray(arr).resize((w, h), Image.NEAREST).save(buf, format="PNG")
    else:
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).resize(
            (w, h), Image.LANCZOS
        ).save(buf, format="PNG")
    return buf.getvalue()


def run_colligo_fill(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    host: str | None = None,
    token: str | None = None,
    preserve_bgr: np.ndarray | None = None,
    telea_fallback: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, str]:
    """Return (filled image BGR, method tag).

    When ``telea_fallback`` is False (Gradio default), missing credentials or API
    failures return the source image unchanged instead of OpenCV Telea inpaint.
    """
    if mask is None or int((mask > 0).sum()) == 0:
        log("  clean-plate: empty removal mask, skipping")
        return image_bgr.copy(), "skipped"

    host = host or os.environ.get("COLLIGO_HOST", "")
    token = token or os.environ.get("COLLIGO_TOKEN", "")

    H, W = image_bgr.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    if should_skip_generative_fill(image_bgr, mask):
        log("  flat background → solid color fill")
        filled = solid_background_fill(image_bgr, mask)
        method = "solid"
    elif not host or not token:
        log("  COLLIGO_HOST/COLLIGO_TOKEN unset → skipping GenFill (no Telea)")
        return image_bgr.copy(), "skipped_no_credentials"
    else:
        host = normalize_colligo_host(host)
        prompt = fill_prompt_for_furniture_removal(image_bgr, mask)
        log(f"  Colligo host: {host}")
        log(f"  Colligo prompt: {prompt[:140]}…")
        req_body = build_fill_request(width=W, height=H, prompt=prompt)
        log(f"  Colligo request: similarity={req_body.get('similarity')}, detailLevel={req_body.get('detailLevel')}")
        api_size = req_body.get("size", {})
        aw = int(api_size.get("width", W))
        ah = int(api_size.get("height", H))

        src_bytes = _resize_png_bytes(image_bgr, aw, ah, is_mask=False) if (aw, ah) != (W, H) else cv2.imencode(".png", image_bgr)[1].tobytes()
        mask_bytes = _resize_png_bytes(mask, aw, ah, is_mask=True) if (aw, ah) != (W, H) else cv2.imencode(".png", mask)[1].tobytes()

        hdrs = {"Authorization": f"Bearer {token}" if not token.startswith("Bearer ") else token, "accept": "application/json"}
        ep = f"{host.rstrip('/')}/v2/images/fill"
        method = "colligo"
        try:
            r = requests.post(
                ep,
                headers=hdrs,
                timeout=180,
                files={
                    "source": ("source.png", src_bytes, "image/png"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                    "request": ("fill_request.json", json.dumps(req_body).encode(), "application/json"),
                },
            )
            if not r.ok:
                log(f"  Colligo fill failed HTTP {r.status_code}: {r.text[:800]}")
                r.raise_for_status()
            data = r.json()
            img_info = data["outputs"][0]["image"]
            log(f"  Colligo fill ID: {img_info.get('id', '—')}")
            filled = cv2.cvtColor(
                np.array(
                    Image.open(
                        io.BytesIO(requests.get(img_info["presignedUrl"], timeout=60).content)
                    )
                ),
                cv2.COLOR_RGB2BGR,
            )
            if (aw, ah) != (W, H):
                filled = cv2.resize(filled, (W, H), interpolation=cv2.INTER_LANCZOS4)
            filled = post_correct_flat_fill(image_bgr, filled, mask)
        except Exception as exc:
            if telea_fallback:
                log(f"  Colligo fill failed ({exc}) → Telea inpaint fallback")
                filled = cv2.inpaint(image_bgr, (mask > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
                method = "telea"
            else:
                log(f"  Colligo fill failed ({exc}) → skipping GenFill (no Telea)")
                return image_bgr.copy(), "colligo_failed"

    preserve = preserve_bgr if preserve_bgr is not None else image_bgr
    out = paste_inpaint_on_source(preserve, filled, mask)
    return out, method

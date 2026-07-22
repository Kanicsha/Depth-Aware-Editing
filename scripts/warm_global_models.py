#!/usr/bin/env python3
"""Load all global (non-image-specific) models once to verify cache + warm RAM."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_cache import configure_model_cache, log_model_cache_status


def main() -> None:
    configure_model_cache()
    t0 = time.time()
    print("[warm] Loading depth + SAM pipelines...", flush=True)
    from utils.lazy_models import ensure_preprocess_models, ensure_generation_models, get_device

    ensure_preprocess_models()

    print("[warm] Loading AnyDoor + DINOv2 + SD 2.1...", flush=True)
    device = get_device()
    ensure_generation_models(device)

    log_model_cache_status(device=str(device))
    print(f"[warm] Done in {time.time() - t0:.1f}s — keep this process running or use Gradio for the same session cache.", flush=True)


if __name__ == "__main__":
    main()

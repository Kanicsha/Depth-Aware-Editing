"""Lazy loaders for global models used by the Gradio demo."""
from __future__ import annotations

import gc
import threading
from typing import Any, Tuple

import torch
from omegaconf import OmegaConf

from cldm.ddim_hacked_mpi_featguidance import DDIMSampler
from cldm.hack import disable_verbosity
from cldm.model import create_model, load_state_dict

_generation_lock = threading.Lock()
_generation_loaded = False
_model: Any = None
_ddim_sampler: Any = None
_diff_handles: Any = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_preprocess_models() -> None:
    """Load SAM + Depth Anything (first Analyze or Generate)."""
    from utils.mpi import preprocess

    preprocess.ensure_preprocess_models()


def ensure_generation_models(
    device: torch.device | None = None,
) -> Tuple[Any, Any, Any]:
    """Load AnyDoor + DINOv2 + SD 2.1 FeatureGuidance (first Generate)."""
    global _generation_loaded, _model, _ddim_sampler, _diff_handles

    device = device or get_device()
    with _generation_lock:
        if _generation_loaded:
            return _model, _ddim_sampler, _diff_handles

        disable_verbosity()

        print("[lazy-load] Loading SD 2.1 FeatureGuidance...", flush=True)
        from src.featglac import FeatureGuidance

        _diff_handles = FeatureGuidance(conf=None).to(device)

        print("[lazy-load] Loading AnyDoor + DINOv2...", flush=True)
        config = OmegaConf.load("./configs/inference.yaml")
        _model = create_model(config.config_file).cpu()
        state_dict = load_state_dict(config.pretrained_model, location="cpu")
        _model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
        if device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _model = _model.to(device)
        _ddim_sampler = DDIMSampler(_model)
        _generation_loaded = True

        print("[lazy-load] AnyDoor + DINOv2 + SD 2.1 ready.", flush=True)
        return _model, _ddim_sampler, _diff_handles

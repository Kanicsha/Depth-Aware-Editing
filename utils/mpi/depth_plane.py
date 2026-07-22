"""Placement depth plane calibration in Depth Anything native units."""
from __future__ import annotations

import os

import cv2
import numpy as np


def depth_to_array(depth) -> np.ndarray:
    """Normalize HF PIL / numpy depth to float32 H×W."""
    if hasattr(depth, "mode"):
        arr = np.array(depth.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
    return arr


def depth_max_value(depth_arr: np.ndarray) -> int:
    mx = float(np.max(depth_arr)) if depth_arr.size else 255.0
    return max(1, int(round(mx)))


def placement_region_from_dict(image_dict: dict, shape: tuple[int, int]) -> np.ndarray:
    hint = image_dict["hint"]
    if hint.shape[-1] >= 4:
        region = (hint[:, :, -1] > 0.5).astype(np.uint8)
    else:
        region = np.zeros(shape[:2], dtype=np.uint8)
    h, w = shape[:2]
    if region.shape != (h, w):
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
    return region


def placement_depth_z_star(depth_arr: np.ndarray, placement_mask: np.ndarray) -> float:
    mask = placement_mask > 0
    vals = depth_arr[mask] if np.any(mask) else depth_arr[depth_arr > 0]
    if vals.size == 0:
        return float(np.median(depth_arr))
    return float(np.median(vals))


def depth_tolerance(depth_arr, placement_mask, *, z_star=None) -> float:
    env_tau = os.environ.get("DEPTH_LAYER_TAU")
    if env_tau is not None:
        return max(1.0, float(env_tau))
    mask = placement_mask > 0
    vals = depth_arr[mask] if np.any(mask) else depth_arr.ravel()
    if vals.size < 2:
        return 6.0
    z = z_star if z_star is not None else float(np.median(vals))
    mad = float(np.median(np.abs(vals - z)))
    span = float(np.percentile(vals, 90) - np.percentile(vals, 10))
    tau = max(mad * 1.5, span * 0.06, 4.0)
    # Avoid classifying every segment as SAME when the slider is far from bbox depths.
    tau_cap = float(os.environ.get("DEPTH_LAYER_TAU_MAX", "25"))
    return min(tau, max(4.0, tau_cap))


def classify_segment_depth(median_d: float, z_star: float, tau: float) -> str:
    # Depth-Anything relative depth: higher value = closer to camera.
    # A segment closer than the placement plane occludes the object -> "front".
    if median_d > z_star + tau:
        return "front"
    if median_d < z_star - tau:
        return "behind"
    return "same"


def mpi_depth_partition(z_star: float, depth_arr: np.ndarray) -> list[tuple[int, int]]:
    z = int(round(float(z_star)))
    dmax = depth_max_value(depth_arr)
    z = max(0, min(z, dmax))
    return [(0, z), (z, dmax)]

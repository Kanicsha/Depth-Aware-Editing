"""SAM-based layered removal masks for clean-plate object placement."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from utils.mpi.depth_plane import (
    classify_segment_depth,
    depth_tolerance,
    placement_region_from_dict,
)


def _sam_label_map(sam_mask) -> np.ndarray:
    arr = np.asarray(sam_mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def _segment_intersects_placement(seg: np.ndarray, placement: np.ndarray) -> bool:
    return int((seg & (placement > 0)).sum()) > 0


_GENFILL_MODE_ALIASES: dict[str, str] = {
    "same_only": "same_intersection",
    "same_intersection": "same_intersection",
    "full_bbox": "full_bbox",
    "full-bbox": "full_bbox",
    "bbox": "full_bbox",
    "entire_bbox": "full_bbox",
    "same_and_behind_intersection": "same_and_behind_intersection",
    "same_and_behind": "same_and_behind_intersection",
    "c3": "same_and_behind_intersection",
    "behind_intersection": "behind_intersection",
    "behind_only": "behind_intersection",
    "e2": "behind_intersection",
    "f3": "f3",
    "same_intersection_exclude_behind_anydoor": "f3",
    "same_exclude_behind_anydoor": "f3",
    "none": "none",
}


def genfill_mode() -> str:
    """Normalized GenFill / AnyDoor experiment mode from GENFILL_MODE env."""
    raw = os.environ.get("GENFILL_MODE", "same_intersection").strip().lower()
    return _GENFILL_MODE_ALIASES.get(raw, "same_intersection")


def anydoor_hint_excludes_behind(mode: str | None = None) -> bool:
    return genfill_mode() if mode is None else mode == "f3"


def _dilate_removal_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def resolve_genfill_removal_mask(
    mode: str,
    image_dict: dict,
    full_shape: tuple[int, int],
    same_mask: np.ndarray,
    behind_mask: np.ndarray,
    front_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Build GenFill removal mask for the selected experiment mode."""
    same_d = _dilate_removal_mask(same_mask)
    behind_d = _dilate_removal_mask(behind_mask)
    meta: dict = {
        "genfill_mode": mode,
        "same_layer_pixels": int((same_mask > 0).sum()),
        "behind_layer_pixels": int((behind_mask > 0).sum()),
        "front_layer_pixels": int((front_mask > 0).sum()),
    }

    if mode == "none":
        removal = np.zeros(full_shape[:2], dtype=np.uint8)
    elif mode == "full_bbox":
        removal = full_bbox_removal_mask(image_dict, full_shape, front_mask=front_mask)
    elif mode == "same_and_behind_intersection":
        union = ((same_mask > 0) | (behind_mask > 0)).astype(np.uint8) * 255
        removal = _dilate_removal_mask(union)
    elif mode == "behind_intersection":
        removal = behind_d
    elif mode == "f3":
        removal = same_d
        meta["anydoor_hint_excludes_behind"] = True
    else:  # same_intersection
        removal = same_d

    meta["genfill_removal_pixels"] = int((removal > 0).sum())
    return removal, meta


def anydoor_edit_mask(
    image_dict: dict,
    full_shape: tuple[int, int],
    *,
    behind_mask: np.ndarray,
    ref_alpha_crop: np.ndarray | None = None,
) -> np.ndarray:
    """Placement edit mask for AnyDoor: bbox minus BEHIND, plus reference alpha."""
    placement = placement_region_from_dict(image_dict, full_shape)
    edit = (placement > 0).astype(np.float32)
    bm = behind_mask
    if bm.shape[:2] != edit.shape[:2]:
        bm = cv2.resize(bm, (edit.shape[1], edit.shape[0]), interpolation=cv2.INTER_NEAREST)
    edit = edit * (bm == 0).astype(np.float32)
    if ref_alpha_crop is not None:
        alpha = ref_alpha_crop.astype(np.float32)
        if float(alpha.max()) > 1.5:
            alpha = alpha / 255.0
        if alpha.shape[:2] != edit.shape[:2]:
            alpha = cv2.resize(alpha, (edit.shape[1], edit.shape[0]), interpolation=cv2.INTER_NEAREST)
        edit = np.clip(np.maximum(edit, alpha), 0.0, 1.0)
    return edit


def patch_image_dict_anydoor_hint(image_dict: dict, edit_mask: np.ndarray) -> dict:
    """Replace hint alpha channel with a custom AnyDoor edit mask (512 workspace)."""
    out = dict(image_dict)
    hint = out["hint"].copy()
    em = edit_mask.astype(np.float32)
    if em.shape[:2] != hint.shape[:2]:
        em = cv2.resize(em, (hint.shape[1], hint.shape[0]), interpolation=cv2.INTER_NEAREST)
    hint[:, :, -1] = np.clip(em, 0.0, 1.0)
    out["hint"] = hint
    return out


def full_bbox_removal_mask(
    image_dict: dict,
    full_shape: tuple[int, int],
    front_mask: np.ndarray | None = None,
) -> np.ndarray:
    """GenFill the entire placement hint bbox (minus FRONT occluders if given)."""
    placement = placement_region_from_dict(image_dict, full_shape)
    removal = (placement > 0).astype(np.uint8) * 255
    if front_mask is not None and int((front_mask > 0).sum()) > 0:
        fm = front_mask
        if fm.shape[:2] != removal.shape[:2]:
            fm = cv2.resize(fm, (removal.shape[1], removal.shape[0]), interpolation=cv2.INTER_NEAREST)
        removal[fm > 0] = 0
    return removal


def sam_layered_removal_masks(
    sam_mask,
    image_dict: dict,
    depth_map=None,
    depth_value: int | float | None = None,
    *,
    z_star: float | None = None,
    tau: float | None = None,
    segment_intersection_thresh: float | None = None,
    bbox_coverage_thresh: float | None = None,
    border_floor_exclude_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build SAME-depth removal + FRONT/BEHIND restore masks (512, uint8 0/255)."""
    segment_intersection_thresh = float(
        os.environ.get("REMOVAL_SEG_IOU", segment_intersection_thresh or 0.15)
    )
    bbox_coverage_thresh = float(
        os.environ.get("REMOVAL_BBOX_COVER", bbox_coverage_thresh or 0.08)
    )
    border_floor_exclude_ratio = float(
        os.environ.get("REMOVAL_FLOOR_EXCLUDE", border_floor_exclude_ratio or 0.55)
    )

    labels = _sam_label_map(sam_mask)
    h, w = labels.shape[:2]
    placement = placement_region_from_dict(image_dict, (h, w))
    bbox_area = max(int((placement > 0).sum()), 1)

    depth_arr = None
    if depth_map is not None:
        depth_arr = np.asarray(depth_map, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.shape != (h, w):
            depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)

    if z_star is None:
        z_star = float(depth_value) if depth_value is not None else 128.0
    else:
        z_star = float(z_star)
    if tau is None and depth_arr is not None:
        tau = depth_tolerance(depth_arr, placement, z_star=z_star)
    elif tau is None:
        tau = 6.0

    same_mask = np.zeros((h, w), dtype=np.uint8)
    front_mask = np.zeros((h, w), dtype=np.uint8)
    behind_mask = np.zeros((h, w), dtype=np.uint8)
    classifications: dict[int, str] = {}
    kept_same: list[int] = []
    kept_front: list[int] = []
    kept_behind: list[int] = []

    for seg_id in [int(i) for i in np.unique(labels) if int(i) > 0]:
        seg = labels == seg_id
        seg_area = int(seg.sum())
        if seg_area == 0:
            continue

        intersection = seg & (placement > 0)
        inter_area = int(intersection.sum())
        if inter_area == 0:
            continue

        seg_in_bbox = inter_area / seg_area
        bbox_cover = inter_area / bbox_area
        if seg_in_bbox < segment_intersection_thresh and bbox_cover < bbox_coverage_thresh:
            continue

        border = np.zeros((h, w), dtype=bool)
        band = max(2, min(h, w) // 64)
        border[:band, :] = True
        border[-band:, :] = True
        border[:, :band] = True
        border[:, -band:] = True
        border_touch = int((seg & border).sum())
        if border_touch / max(seg_area, 1) > border_floor_exclude_ratio:
            continue

        if depth_arr is not None:
            depth_region = depth_arr[intersection] if inter_area else depth_arr[seg]
            median_depth = float(np.median(depth_region)) if depth_region.size else z_star
        else:
            median_depth = z_star

        layer = classify_segment_depth(median_depth, z_star, tau)
        classifications[seg_id] = layer

        # GenFill only SAME-depth clutter at the placement plane. BEHIND occluders
        # (e.g. table behind a chair) stay in the photo and are restored after AnyDoor.
        if layer == "front":
            front_mask[intersection] = 255
            kept_front.append(seg_id)
        elif layer == "same":
            same_mask[intersection] = 255
            kept_same.append(seg_id)
        else:  # "behind"
            behind_mask[intersection] = 255
            kept_behind.append(seg_id)

    debug = {
        "z_star": z_star,
        "tau": tau,
        "classifications": classifications,
        "kept_same_segment_ids": kept_same,
        "kept_behind_segment_ids": kept_behind,
        "kept_front_segment_ids": kept_front,
        "same_pixels": int((same_mask > 0).sum()),
        "removal_pixels": int((_dilate_removal_mask(same_mask) > 0).sum()),
        "front_pixels": int((front_mask > 0).sum()),
        "behind_pixels": int((behind_mask > 0).sum()),
        "bbox_pixels": bbox_area,
    }
    return same_mask, front_mask, behind_mask, debug


def sam_intersection_removal_mask(
    sam_mask,
    image_dict: dict,
    depth_map=None,
    depth_value: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Backward-compatible wrapper returning dilated SAME-layer removal mask only."""
    same_mask, _front, _behind, debug = sam_layered_removal_masks(
        sam_mask,
        image_dict,
        depth_map=depth_map,
        depth_value=depth_value,
        **kwargs,
    )
    removal = _dilate_removal_mask(same_mask)
    legacy = {
        "kept_segment_ids": debug.get("kept_same_segment_ids", []),
        "removal_pixels": debug.get("removal_pixels", 0),
        "bbox_pixels": debug.get("bbox_pixels", 0),
        **debug,
    }
    return removal, legacy


def save_removal_mask_debug(crop_rgb, removal_mask, save_path) -> None:
    vis = crop_rgb.copy()
    if vis.shape[:2] != removal_mask.shape[:2]:
        removal_mask = cv2.resize(
            removal_mask,
            (vis.shape[1], vis.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    overlay = vis.copy()
    overlay[removal_mask > 0] = (255, 64, 64)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def save_layer_debug_png(
    crop_rgb: np.ndarray,
    removal_mask: np.ndarray,
    front_mask: np.ndarray,
    save_path: str,
    debug_info: dict | None = None,
    behind_mask: np.ndarray | None = None,
) -> None:
    vis = crop_rgb.copy().astype(np.float32)
    rm = removal_mask
    fm = front_mask
    bm = behind_mask if behind_mask is not None else np.zeros_like(rm)
    if rm.shape[:2] != vis.shape[:2]:
        rm = cv2.resize(rm, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    if fm.shape[:2] != vis.shape[:2]:
        fm = cv2.resize(fm, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    if bm.shape[:2] != vis.shape[:2]:
        bm = cv2.resize(bm, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = vis.copy()
    overlay[bm > 0] = overlay[bm > 0] * 0.5 + np.array([64, 128, 255], dtype=np.float32) * 0.5
    overlay[fm > 0] = overlay[fm > 0] * 0.5 + np.array([64, 255, 64], dtype=np.float32) * 0.5
    overlay[rm > 0] = overlay[rm > 0] * 0.5 + np.array([255, 64, 64], dtype=np.float32) * 0.5
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    if debug_info is not None:
        meta_path = save_path.rsplit(".", 1)[0] + ".json"
        serializable = {k: v for k, v in debug_info.items()}
        serializable["classifications"] = {
            str(k): v for k, v in serializable.get("classifications", {}).items()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

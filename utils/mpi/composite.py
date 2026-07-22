"""Map 512 crop masks/images to full background and alpha-composite results."""
from __future__ import annotations

import os

import cv2
import numpy as np


def _unpad_square_patch(patch: np.ndarray, extra_sizes) -> np.ndarray:
    H1, W1, H2, W2 = [int(x) for x in extra_sizes]
    if patch.shape[0] != H2 or patch.shape[1] != W2:
        patch = cv2.resize(patch, (W2, H2), interpolation=cv2.INTER_LANCZOS4)
    if W1 == H1:
        return patch
    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        return patch[:, pad1 : W2 - pad2]
    pad1 = int((H2 - H1) / 2)
    pad2 = H2 - H1 - pad1
    return patch[pad1 : H2 - pad2, :]


def map_crop_mask_to_full(
    crop_mask: np.ndarray,
    image_dict: dict,
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Map 512 workspace mask to full-resolution background coordinates."""
    full_h, full_w = full_shape[:2]
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    patch = _unpad_square_patch(crop_mask, image_dict["extra_sizes"])
    y1, y2, x1, x2 = [int(v) for v in image_dict["tar_box_yyxx_crop"]]
    if patch.shape[0] != (y2 - y1) or patch.shape[1] != (x2 - x1):
        patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full_mask[y1:y2, x1:x2] = patch
    return full_mask


def apply_clean_plate_to_full_bg(
    full_bg_rgb: np.ndarray,
    cleaned_crop_rgb: np.ndarray,
    removal_mask_crop: np.ndarray,
    image_dict: dict,
) -> np.ndarray:
    """Paste cleaned pixels only where removal_mask is set (crop → full coords)."""
    full_mask = map_crop_mask_to_full(removal_mask_crop, image_dict, full_bg_rgb.shape[:2])
    cleaned_full = full_bg_rgb.copy()
    patch = _unpad_square_patch(cleaned_crop_rgb, image_dict["extra_sizes"])
    y1, y2, x1, x2 = [int(v) for v in image_dict["tar_box_yyxx_crop"]]
    if patch.shape[0] != (y2 - y1) or patch.shape[1] != (x2 - x1):
        patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    region = full_bg_rgb.copy()
    region[y1:y2, x1:x2] = patch
    cleaned_full[full_mask > 0] = region[full_mask > 0]
    return cleaned_full


def extract_placement_alpha(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    placement_mask: np.ndarray,
    *,
    diff_thresh: float = 18.0,
) -> np.ndarray:
    """Float alpha [0,1] for pixels that changed inside placement_mask."""
    if before_rgb.shape != after_rgb.shape:
        after_rgb = cv2.resize(
            after_rgb,
            (before_rgb.shape[1], before_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    if placement_mask.shape[:2] != before_rgb.shape[:2]:
        placement_mask = cv2.resize(
            placement_mask,
            (before_rgb.shape[1], before_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    diff = np.abs(after_rgb.astype(np.float32) - before_rgb.astype(np.float32)).sum(axis=2)
    alpha = ((diff > diff_thresh) & (placement_mask > 0)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return np.clip(alpha, 0.0, 1.0)


def reference_alpha_from_dict(image_dict: dict, full_shape: tuple[int, int]) -> np.ndarray | None:
    """Map reference-object alpha from crop coords to full image [0,1]."""
    ref_alpha_crop = image_dict.get("ref_alpha_crop")
    if ref_alpha_crop is None:
        return None
    full_alpha = map_crop_mask_to_full(ref_alpha_crop, image_dict, full_shape).astype(np.float32) / 255.0
    if float(full_alpha.max()) <= 0.0:
        return None
    return np.clip(full_alpha, 0.0, 1.0)


def _shadow_weight() -> float:
    return float(os.environ.get("COMPOSITE_SHADOW_WEIGHT", "0.35"))


def _shadow_dilate_px() -> int:
    return max(0, int(os.environ.get("COMPOSITE_SHADOW_DILATE", "12")))


def placement_alpha_full(
    image_dict: dict,
    full_shape: tuple[int, int],
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    placement_mask_full: np.ndarray,
) -> np.ndarray:
    """Prefer warped reference alpha; fall back to diff-based alpha."""
    ref_alpha = reference_alpha_from_dict(image_dict, full_shape)
    if ref_alpha is not None:
        alpha = ref_alpha.copy()
        k = _shadow_dilate_px()
        if k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
            dilated = cv2.dilate((alpha * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
            shadow = np.clip(dilated - alpha, 0.0, 1.0) * _shadow_weight()
            alpha = np.clip(alpha + shadow, 0.0, 1.0)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        placement = (placement_mask_full > 0).astype(np.float32)
        return np.clip(alpha * placement, 0.0, 1.0)
    return extract_placement_alpha(before_rgb, after_rgb, placement_mask_full)


def occlusion_aware_composite(
    original_bg_rgb: np.ndarray,
    bg_working_rgb: np.ndarray,
    anydoor_result_rgb: np.ndarray,
    image_dict: dict,
    *,
    removal_mask_crop: np.ndarray | None = None,
    front_mask_crop: np.ndarray | None = None,
    behind_mask_crop: np.ndarray | None = None,
    bg_clean_rgb: np.ndarray | None = None,
    save_dir: str | None = None,
    placement_mode: str | None = None,
) -> np.ndarray:
    """Alpha-composite AnyDoor object; restore BEHIND through holes and FRONT on top."""
    from utils.mpi.removal_mask import is_occlusion_placement_mode

    full_shape = original_bg_rgb.shape[:2]
    placement_mask_full = placement_mask_from_image_dict(image_dict, full_shape)
    removal = removal_mask_crop if removal_mask_crop is not None else np.zeros(full_shape, dtype=np.uint8)
    front = front_mask_crop if front_mask_crop is not None else np.zeros(full_shape, dtype=np.uint8)
    behind = behind_mask_crop if behind_mask_crop is not None else np.zeros(full_shape, dtype=np.uint8)
    touched = touched_region_full(image_dict, full_shape, removal, front)
    occlusion_mode = is_occlusion_placement_mode(placement_mode)

    if occlusion_mode:
        layer0 = original_bg_rgb.astype(np.float32)
    else:
        layer0 = (bg_clean_rgb if bg_clean_rgb is not None else bg_working_rgb).astype(np.float32)
    alpha_obj = placement_alpha_full(
        image_dict,
        full_shape,
        layer0.astype(np.uint8),
        anydoor_result_rgb,
        placement_mask_full,
    )

    a_obj = alpha_obj[:, :, None]
    layer1 = layer0 * (1.0 - a_obj) + anydoor_result_rgb.astype(np.float32) * a_obj

    if behind is not None and int((behind > 0).sum()) > 0:
        behind_full = map_crop_mask_to_full(behind, image_dict, full_shape).astype(np.float32) / 255.0
        behind_full = cv2.GaussianBlur(behind_full, (5, 5), 0)
        restore_behind = np.clip(behind_full * (1.0 - alpha_obj), 0.0, 1.0)
        a_behind = restore_behind[:, :, None]
        layer1 = layer1 * (1.0 - a_behind) + original_bg_rgb.astype(np.float32) * a_behind

    if front is not None and int((front > 0).sum()) > 0:
        front_full = map_crop_mask_to_full(front, image_dict, full_shape).astype(np.float32) / 255.0
        front_full = cv2.GaussianBlur(front_full, (9, 9), 0)
        alpha_front = np.clip(front_full * (1.0 - alpha_obj), 0.0, 1.0)
        a_front = alpha_front[:, :, None]
        layer2 = layer1 * (1.0 - a_front) + original_bg_rgb.astype(np.float32) * a_front
    else:
        layer2 = layer1

    original = original_bg_rgb.astype(np.float32)
    touched_f = (touched > 0).astype(np.float32)[:, :, None]
    out = layer2 * touched_f + original * (1.0 - touched_f)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(save_dir, "composite_alpha_obj.png"),
            (np.clip(alpha_obj, 0, 1) * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(save_dir, "composite_layer1_object.jpg"),
            cv2.cvtColor(np.clip(layer1, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(save_dir, "final_composite.jpg"),
            cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

    return np.clip(out, 0, 255).astype(np.uint8)


def composite_object_on_original(
    original_bg_rgb: np.ndarray,
    working_result_rgb: np.ndarray,
    before_working_rgb: np.ndarray,
    placement_mask_full: np.ndarray,
) -> np.ndarray:
    """Paste only the placed object (+ shadow) from working_result onto original_bg."""
    alpha = extract_placement_alpha(before_working_rgb, working_result_rgb, placement_mask_full)
    alpha3 = alpha[:, :, None]
    out = original_bg_rgb.astype(np.float32) * (1.0 - alpha3) + working_result_rgb.astype(np.float32) * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def placement_mask_from_image_dict(image_dict: dict, full_shape: tuple[int, int]) -> np.ndarray:
    hint = image_dict["hint"]
    region = (hint[:, :, -1] > 0.5).astype(np.uint8) * 255
    return map_crop_mask_to_full(region, image_dict, full_shape)


def _touch_dilate_px() -> int:
    return max(0, int(os.environ.get("COMPOSITE_TOUCH_DILATE", "24")))


def touched_region_full(
    image_dict: dict,
    full_shape: tuple[int, int],
    removal_mask_crop: np.ndarray,
    front_mask_crop: np.ndarray,
) -> np.ndarray:
    """Dilated union of placement, removal, and front masks in full coords."""
    placement = placement_mask_from_image_dict(image_dict, full_shape)
    removal = map_crop_mask_to_full(removal_mask_crop, image_dict, full_shape)
    front = map_crop_mask_to_full(front_mask_crop, image_dict, full_shape)
    touched = ((placement > 0) | (removal > 0) | (front > 0)).astype(np.uint8) * 255
    k = _touch_dilate_px()
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
        touched = cv2.dilate(touched, kernel, iterations=1)
    return touched


def layered_composite(
    original_bg_rgb: np.ndarray,
    bg_clean_rgb: np.ndarray,
    anydoor_result_rgb: np.ndarray,
    image_dict: dict,
    removal_mask_crop: np.ndarray,
    front_mask_crop: np.ndarray,
    *,
    behind_mask_crop: np.ndarray | None = None,
    bg_working_rgb: np.ndarray | None = None,
    save_dir: str | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper around occlusion_aware_composite."""
    return occlusion_aware_composite(
        original_bg_rgb,
        bg_working_rgb if bg_working_rgb is not None else bg_clean_rgb,
        anydoor_result_rgb,
        image_dict,
        removal_mask_crop=removal_mask_crop,
        front_mask_crop=front_mask_crop,
        behind_mask_crop=behind_mask_crop,
        bg_clean_rgb=bg_clean_rgb,
        save_dir=save_dir,
    )

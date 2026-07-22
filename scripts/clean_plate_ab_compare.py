#!/usr/bin/env python3
"""Gate 3: side-by-side clean-plate A/B on a crop (GenFill vs Telea-only baseline)."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.colligo.fill_client import run_colligo_fill
from utils.mpi.preprocess import get_depth_and_sam_mask
from utils.mpi.removal_mask import (
    genfill_mode,
    resolve_genfill_removal_mask,
    sam_layered_removal_masks,
    save_removal_mask_debug,
)
from run_inference_object_placement import process_pairs


def _prepare_reference(path: str):
    ref_np = np.array(Image.open(path).convert("RGBA"))
    ref_image = ref_np[:, :, :3]
    ref_mask = (ref_np[:, :, 3] > 128).astype(np.uint8) * 255
    return ref_image, ref_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean-plate A/B compare (Gate 3)")
    ap.add_argument("background", help="Background RGB image")
    ap.add_argument("reference", help="Reference RGBA object")
    ap.add_argument("bbox_mask", help="Full-size grayscale placement bbox mask")
    ap.add_argument("out_dir", help="Directory for A/B outputs")
    ap.add_argument("--depth", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bg_rgb = np.array(Image.open(args.background).convert("RGB"))
    ref_image, ref_mask = _prepare_reference(args.reference)
    tar_mask = cv2.imread(args.bbox_mask, cv2.IMREAD_GRAYSCALE)
    if tar_mask is None:
        raise SystemExit(f"Could not read bbox mask: {args.bbox_mask}")

    image_dict = process_pairs(ref_image, ref_mask, bg_rgb.copy(), tar_mask, shape_control=False)
    crop_rgb = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
    depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(crop_rgb), is_relative_depth=True)

    same_mask, front_mask, behind_mask, _dbg = sam_layered_removal_masks(
        sam_mask, image_dict, depth_map=np.array(depth), z_star=float(args.depth)
    )
    removal_mask, _meta = resolve_genfill_removal_mask(
        genfill_mode(),
        image_dict,
        crop_rgb.shape[:2],
        same_mask,
        behind_mask,
        front_mask,
    )
    save_removal_mask_debug(
        crop_rgb,
        removal_mask,
        os.path.join(args.out_dir, "removal_mask_debug.png"),
    )
    cv2.imwrite(os.path.join(args.out_dir, "removal_mask.png"), removal_mask)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    baseline_bgr, baseline_method = run_colligo_fill(
        crop_bgr,
        np.zeros_like(removal_mask),
        host="",
        token="",
    )
    cleaned_bgr, cleaned_method = run_colligo_fill(crop_bgr, removal_mask)

    cv2.imwrite(os.path.join(args.out_dir, "A_baseline_crop.jpg"), baseline_bgr)
    cv2.imwrite(os.path.join(args.out_dir, "B_clean_plate_crop.jpg"), cleaned_bgr)

    h, w = crop_bgr.shape[:2]
    panel = np.zeros((h, w * 2, 3), dtype=np.uint8)
    panel[:, :w] = baseline_bgr
    panel[:, w:] = cleaned_bgr
    cv2.imwrite(os.path.join(args.out_dir, "AB_compare.jpg"), panel)

    print(f"A (no removal mask, {baseline_method}) → {args.out_dir}/A_baseline_crop.jpg")
    print(f"B (SAM removal + {cleaned_method}) → {args.out_dir}/B_clean_plate_crop.jpg")
    print(f"Side-by-side → {args.out_dir}/AB_compare.jpg")


if __name__ == "__main__":
    main()

import os
import sys
import time
from datetime import datetime

sys.path.append(".")
from utils.model_cache import configure_model_cache, log_model_cache_status
from utils.run_log import capture_run_log, make_timestamped_run_dir

configure_model_cache()

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import cv2
import einops
import torch
import random
from pytorch_lightning import seed_everything
from cldm.hack import disable_verbosity, enable_sliced_attention, disable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from ldm.util import instantiate_from_config

from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from run_inference_object_placement import process_pairs, crop_back, inference_single_image
from utils.colligo.fill_client import run_colligo_fill
from utils.lazy_models import ensure_generation_models, ensure_preprocess_models, get_device
from utils.mpi.removal_mask import (
    anydoor_edit_mask,
    anydoor_hint_excludes_behind,
    genfill_mode,
    patch_image_dict_anydoor_hint,
    resolve_genfill_removal_mask,
    sam_layered_removal_masks,
    save_layer_debug_png,
    save_removal_mask_debug,
)
from utils.mpi.composite import (
    apply_clean_plate_to_full_bg,
    layered_composite,
    occlusion_aware_composite,
    placement_mask_from_image_dict,
)
from utils.mpi.depth_plane import (
    depth_max_value,
    depth_to_array,
    mpi_depth_partition,
    placement_depth_z_star,
    placement_region_from_dict,
)
from utils.mpi.null_text_config import (
    crop_cache_key,
    null_text_cache_paths,
    null_text_ddim_steps,
    null_text_disk_cache_enabled,
    null_text_inner_steps,
)

# Depth/SAM helpers — models load on first Analyze or Generate, not at import.
from utils.mpi.preprocess import get_depth_and_sam_mask, plot_depth_bins
from utils.mpi.postprocess import sam_postprocess, sam_postprocess2, get_sam_mask
from utils.mpi.mpi import get_mpi_rgb_and_alpha
from utils.mpi.null_text_inv import NullTextPipeline
from diffusers.schedulers import DDIMScheduler

device = get_device()

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

print(
    "[model-cache] Gradio ready — models load on first Analyze (~3 GB) or Generate (~20 GB).",
    flush=True,
)
log_model_cache_status(device=str(device))

def _prepare_reference(reference_image):
    ref_np = np.array(reference_image.convert("RGBA"))
    ref_image = ref_np[:, :, :3]  # already RGB (PIL)
    ref_mask = ref_np[:, :, 3]
    ref_mask = (ref_mask > 128).astype(np.uint8) * 255
    ref_mask = cv2.dilate(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    ref_mask = cv2.erode(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    return ref_image, ref_mask


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log_depth_threshold(stage: str, depth_value, *, note: str = "") -> None:
    msg = f"[gradio] {stage}: depth threshold = {int(depth_value)}"
    if note:
        msg = f"{msg} ({note})"
    print(f"{_ts()} {msg}", flush=True)


def _resolve_colligo_credentials(colligo_host=None, colligo_token=None):
    host = (colligo_host or "").strip() or os.environ.get("COLLIGO_HOST", "")
    token = (colligo_token or "").strip() or os.environ.get("COLLIGO_TOKEN", "")
    return host, token


def _clean_plate_active(enable_clean_plate, colligo_host=None, colligo_token=None) -> bool:
    """GenFill clean-plate runs only when enabled AND Colligo host+token are set."""
    if not _default_enable_clean_plate(enable_clean_plate):
        return False
    host, token = _resolve_colligo_credentials(colligo_host, colligo_token)
    return bool(host and token)


def _default_enable_clean_plate(enable_clean_plate):
    if enable_clean_plate is not None:
        return bool(enable_clean_plate)
    return True


def _run_clean_plate_pipeline(
    bg_np,
    ref_image,
    ref_mask,
    tar_mask,
    depth_value,
    enable_clean_plate,
    current_save_dir,
    is_relative_depth=True,
    colligo_host=None,
    colligo_token=None,
):
    """Build image_dict; optionally layered GenFill clean-plate."""
    image_dict = process_pairs(ref_image, ref_mask, bg_np.copy(), tar_mask, shape_control=False)
    gt_image_cropped = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
    depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(gt_image_cropped), is_relative_depth)

    depth_arr = depth_to_array(depth)
    placement = placement_region_from_dict(image_dict, gt_image_cropped.shape[:2])
    z_star = float(depth_value)

    removal_mask = np.zeros(gt_image_cropped.shape[:2], dtype=np.uint8)
    front_mask = np.zeros(gt_image_cropped.shape[:2], dtype=np.uint8)
    behind_mask = np.zeros(gt_image_cropped.shape[:2], dtype=np.uint8)
    layer_debug = {"z_star": z_star, "classifications": {}}
    sam_mask_original = sam_mask
    clean_method = "disabled"
    bg_working = bg_np.copy()

    _log_depth_threshold(
        "layer masks",
        z_star,
        note="SAM FRONT / SAME / BEHIND for composite (+ optional GenFill)",
    )
    same_mask, front_mask, behind_mask, layer_debug = sam_layered_removal_masks(
        sam_mask,
        image_dict,
        depth_map=depth_arr,
        z_star=z_star,
    )
    gf_mode = genfill_mode()
    removal_mask, gf_meta = resolve_genfill_removal_mask(
        gf_mode,
        image_dict,
        gt_image_cropped.shape[:2],
        same_mask,
        behind_mask,
        front_mask,
    )
    layer_debug.update(gf_meta)
    print(
        f"{_ts()} [clean-plate] GENFILL_MODE={gf_mode} — "
        f"GenFill removal {gf_meta['genfill_removal_pixels']} px "
        f"(same={gf_meta['same_layer_pixels']}, behind={gf_meta['behind_layer_pixels']})",
        flush=True,
    )
    layer_debug_path = os.path.join(current_save_dir, "layer_debug.png")
    save_layer_debug_png(
        gt_image_cropped,
        removal_mask,
        front_mask,
        layer_debug_path,
        layer_debug,
        behind_mask=behind_mask,
    )
    cv2.imwrite(os.path.join(current_save_dir, "behind_mask.png"), behind_mask)

    if not _default_enable_clean_plate(enable_clean_plate):
        print(f"{_ts()} [clean-plate] disabled (checkbox off) — AnyDoor on original background", flush=True)
    elif not _clean_plate_active(enable_clean_plate, colligo_host, colligo_token):
        host, token = _resolve_colligo_credentials(colligo_host, colligo_token)
        clean_method = "skipped_no_colligo"
        print(
            f"{_ts()} [clean-plate] skipped (no Colligo credentials: host={'set' if host else 'missing'}, "
            f"token={'set' if token else 'missing'}) — AnyDoor on original background",
            flush=True,
        )
    else:
        _log_depth_threshold(
            "clean-plate removal mask",
            z_star,
            note=f"GenFill on {gf_mode} removal mask",
        )

        if int((removal_mask > 0).sum()) > 0:
            crop_bgr = cv2.cvtColor(gt_image_cropped, cv2.COLOR_RGB2BGR)
            host, token = _resolve_colligo_credentials(colligo_host, colligo_token)
            cleaned_crop_bgr, clean_method = run_colligo_fill(
                crop_bgr,
                removal_mask,
                host=host,
                token=token,
                telea_fallback=False,
            )
            if clean_method in ("colligo", "solid"):
                cleaned_crop_rgb = cv2.cvtColor(cleaned_crop_bgr, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(current_save_dir, "clean_plate_crop.jpg"),
                    cleaned_crop_bgr,
                )
                bg_working = apply_clean_plate_to_full_bg(
                    bg_np, cleaned_crop_rgb, removal_mask, image_dict
                )
                image_dict = process_pairs(ref_image, ref_mask, bg_working.copy(), tar_mask, shape_control=False)
                gt_image_cropped = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
                depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(gt_image_cropped), is_relative_depth)
            else:
                print(
                    f"{_ts()} [clean-plate] GenFill did not run ({clean_method}) — AnyDoor on original background",
                    flush=True,
                )
                clean_method = "skipped_no_colligo"
        else:
            clean_method = "skipped_empty_mask"
            print(f"{_ts()} [clean-plate] empty removal mask — AnyDoor on original background", flush=True)

    artifact_dir = os.path.abspath(current_save_dir)
    cv2.imwrite(
        os.path.join(current_save_dir, "bg_original.jpg"),
        cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(current_save_dir, "bg_working.jpg"),
        cv2.cvtColor(bg_working, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(current_save_dir, "bg_working_crop.jpg"),
        cv2.cvtColor(gt_image_cropped, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(current_save_dir, "same_mask.png"), same_mask)
    cv2.imwrite(os.path.join(current_save_dir, "removal_mask.png"), removal_mask)
    cv2.imwrite(os.path.join(current_save_dir, "front_mask.png"), front_mask)
    save_removal_mask_debug(
        gt_image_cropped,
        removal_mask,
        os.path.join(current_save_dir, "removal_mask_debug.png"),
    )
    layer_debug["pipeline_artifacts"] = {
        "bg_original": "bg_original.jpg",
        "bg_working": "bg_working.jpg",
        "bg_working_crop": "bg_working_crop.jpg",
        "layer_debug": "layer_debug.png",
        "removal_mask": "removal_mask.png",
        "same_mask": "same_mask.png",
        "front_mask": "front_mask.png",
        "behind_mask": "behind_mask.png",
        "clean_plate_crop": "clean_plate_crop.jpg",
        "clean_method": clean_method,
        "genfill_mode": gf_mode,
    }
    print(
        f"{_ts()} [pipeline] saved intermediates → {artifact_dir}\n"
        f"  bg_original.jpg      untouched input photo\n"
        f"  bg_working.jpg       background AnyDoor receives (after optional GenFill)\n"
        f"  bg_working_crop.jpg  512×512 workspace crop from bg_working\n"
        f"  clean_method={clean_method}  genfill_mode={gf_mode}",
        flush=True,
    )

    return (
        bg_working,
        image_dict,
        gt_image_cropped,
        depth,
        sam_mask,
        removal_mask,
        front_mask,
        behind_mask,
        clean_method,
        layer_debug,
        sam_mask_original,
    )


def aug_data_mask(image, mask):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
    ])
    transformed = transform(image=image.astype(np.uint8), mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def extract_bbox_mask(annotation_data, base_image):
    """Extract bounding box and generate binary mask from annotation"""
    if annotation_data is None or "mask" not in annotation_data:
        return None, None

    bg_img = annotation_data["image"]  # Use the annotated image as background
    obj_mask = annotation_data["mask"]  # This is a PIL image
    ann_np = np.array(obj_mask.convert("L"))

    coords = np.argwhere(ann_np > 0)
    if coords.shape[0] == 0:
        return None, None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Generate binary mask from bbox
    mask = np.zeros((base_image.height, base_image.width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    return Image.fromarray(mask), bg_img

def analyze_depth_and_sam(
    background_image,
    reference_image,
    inv_prompt,
    depth_value=50,
    enable_clean_plate=True,
    colligo_host="",
    colligo_token="",
    run_dir=None,
    run_logger=None,
):
    """
    First step: Analyze depth and SAM segmentation to help user choose depth value
    
    Args:
        background_image: Annotated background image with mask
        reference_image: RGBA reference image
    
    Returns:
        Depth analysis plots and suggested depth value
    """
    
    # Input validation
    if background_image is None or reference_image is None:
        raise gr.Error("Please upload both background and reference images.")
    
    if "mask" not in background_image:
        raise gr.Error("Please draw a mask on the background image.")
    
    # Process background image and mask
    bg_image = background_image["image"]
    bbox_mask, bg_image = extract_bbox_mask(background_image, bg_image)
    
    if bbox_mask is None:
        raise gr.Error("Could not extract mask from background image.")

    ensure_preprocess_models()

    bg_np = np.array(bg_image.convert("RGB"))
    ref_image, ref_mask = _prepare_reference(reference_image)
    tar_mask = np.array(bbox_mask)

    current_save_dir = run_dir or make_timestamped_run_dir("analyze")
    os.makedirs(current_save_dir, exist_ok=True)
    print(f"{_ts()} [gradio] Analyze: saving artifacts to {os.path.abspath(current_save_dir)}", flush=True)
    if run_logger is not None:
        run_logger.set_manifest(
            inv_prompt=inv_prompt,
            depth_value=int(round(float(depth_value))),
            enable_clean_plate=bool(_default_enable_clean_plate(enable_clean_plate)),
        )

    enable_clean = _default_enable_clean_plate(enable_clean_plate)
    removal_depth = int(round(float(depth_value)))
    _log_depth_threshold("Analyze", removal_depth, note="from depth slider → removal mask")
    (
        bg_working,
        image_dict,
        gt_image_cropped,
        depth,
        sam_mask,
        removal_mask,
        front_mask,
        behind_mask,
        clean_method,
        layer_debug,
        sam_mask_original,
    ) = _run_clean_plate_pipeline(
        bg_np, ref_image, ref_mask, tar_mask, depth_value=removal_depth,
        enable_clean_plate=enable_clean, current_save_dir=current_save_dir,
        colligo_host=colligo_host, colligo_token=colligo_token,
    )
    print(f"{_ts()} clean-plate method: {clean_method}", flush=True)
    if run_logger is not None:
        run_logger.set_manifest(clean_plate_method=clean_method, layer_debug=layer_debug)
    tar_mask_mpi = image_dict["tar_mpi_mask"]

    tar_mask_mpi_copy = np.ones_like(tar_mask_mpi) * 255
    plot_depth_bins(
        depth,
        sam_mask,
        tar_mask_mpi_copy,
        input_img_name="gradio_infer",
        save_dir=current_save_dir,
        is_crop=True,
    )

    depth_top_path = os.path.join(current_save_dir, "gradio_infer_depth_bins_front_crop.png")
    depth_front_path = os.path.join(current_save_dir, "gradio_infer_depth_bins_crop.png")

    depth_3d_plot = Image.open(depth_top_path)
    depth_front_plot = Image.open(depth_front_path)

    layer_preview_path = os.path.join(current_save_dir, "layer_debug.png")
    removal_preview_path = os.path.join(current_save_dir, "removal_mask_debug.png")
    if os.path.exists(layer_preview_path):
        removal_mask_preview = Image.open(layer_preview_path)
    elif os.path.exists(removal_preview_path):
        removal_mask_preview = Image.open(removal_preview_path)
    else:
        removal_mask_preview = Image.fromarray(gt_image_cropped)

    depth_array = depth_to_array(depth)
    mask_region = np.array(tar_mask_mpi)
    
    # Get depth values in the mask region
    if len(mask_region.shape) == 3:
        masked_depth = depth_array * mask_region[:, :, 0]
    else:
        masked_depth = depth_array * mask_region
    
    # Calculate statistics for suggested depth
    non_zero_depths = masked_depth[masked_depth > 0]
    if len(non_zero_depths) > 0:
        suggested_depth = int(round(placement_depth_z_star(depth_array, mask_region[:, :, 0] if len(mask_region.shape) == 3 else mask_region)))
        dmax = depth_max_value(depth_array)
        suggested_depth = max(10, min(dmax, suggested_depth))
    else:
        suggested_depth = 50  # Default value

    _log_depth_threshold("Analyze", suggested_depth, note="auto-suggested from depth plot (informational only)")

    # Never overwrite a user-adjusted slider; only auto-fill on first run at default 50.
    if removal_depth == 50 and suggested_depth != 50:
        output_depth = suggested_depth
        _log_depth_threshold("Analyze", output_depth, note="applied auto-suggestion to depth slider")
    else:
        output_depth = removal_depth
        _log_depth_threshold("Analyze", output_depth, note="kept depth slider value unchanged")

    if run_logger is not None:
        run_logger.set_manifest(
            suggested_depth=int(suggested_depth),
            output_depth=int(output_depth),
        )

    print(f"{_ts()} [gradio] Analyze: finished; outputs in {os.path.abspath(current_save_dir)}", flush=True)
    return depth_3d_plot, depth_front_plot, removal_mask_preview, image_dict, output_depth

def gradio_infer(
    background_image,
    reference_image,
    depth_value,
    image_dict,
    inv_prompt,
    guidance_scale=5.0,
    num_samples=1,
    enable_clean_plate=True,
    colligo_host="",
    colligo_token="",
    run_dir=None,
    run_logger=None,
):
    """
    Main inference function for Gradio interface
    
    Args:
        background_image: Annotated background image with mask
        reference_image: RGBA reference image
        depth_value: Depth threshold for MPI
        guidance_scale: Guidance scale for generation
        num_samples: Number of samples to generate
    
    Returns:
        Generated image and depth visualization
    """

    model, ddim_sampler, diff_handles = ensure_generation_models(device)
    ensure_preprocess_models()

    # Input validation
    if background_image is None or reference_image is None:
        raise gr.Error("Please upload both background and reference images.")
    
    if "mask" not in background_image:
        raise gr.Error("Please draw a mask on the background image.")
    
    # Configuration
    DConf = OmegaConf.load('./configs/datasets.yaml')
    null_text_emb_path = "./examples/Gradio/null_embed"
    os.makedirs(null_text_emb_path, exist_ok=True)
    image_name = "gradio_inference"
    
    # MPI settings
    save_memory = False
    plot_depth = True
    do_mpi = True
    is_relative_depth = True
    mask_adjustment = False
    enable_shape_control = False
    sam_postprocess_dict = None
    anydoor_mpi_timetep = 20
    blending_timestep = 20
    do_null_text_again = False
    
    current_save_dir = run_dir or make_timestamped_run_dir("generate")
    os.makedirs(current_save_dir, exist_ok=True)
    print(f"{_ts()} [gradio] Generate: saving artifacts to {os.path.abspath(current_save_dir)}", flush=True)
    if run_logger is not None:
        run_logger.set_manifest(
            inv_prompt=inv_prompt,
            depth_value=int(round(float(depth_value))),
            guidance_scale=float(guidance_scale),
            num_samples=int(num_samples),
            enable_clean_plate=bool(_default_enable_clean_plate(enable_clean_plate)),
        )
    
    # Process background image and mask
    bg_image = background_image["image"]
    bbox_mask, bg_image = extract_bbox_mask(background_image, bg_image)
    
    if bbox_mask is None:
        raise gr.Error("Could not extract mask from background image.")
    
    bg_np = np.array(bg_image.convert("RGB"))
    ref_image, ref_mask = _prepare_reference(reference_image)
    tar_mask = np.array(bbox_mask)

    bg_original = bg_np.copy()

    depth_value = int(round(float(depth_value)))
    _log_depth_threshold("Generate", depth_value, note="from depth slider")

    (
        bg_working,
        image_dict,
        gt_image_cropped,
        depth,
        sam_mask,
        removal_mask,
        front_mask,
        behind_mask,
        clean_method,
        layer_debug,
        sam_mask_original,
    ) = _run_clean_plate_pipeline(
        bg_np,
        ref_image,
        ref_mask,
        tar_mask,
        depth_value=depth_value,
        enable_clean_plate=enable_clean_plate,
        current_save_dir=current_save_dir,
        colligo_host=colligo_host,
        colligo_token=colligo_token,
    )
    print(f"{_ts()} clean-plate method: {clean_method}", flush=True)
    if run_logger is not None:
        run_logger.set_manifest(clean_plate_method=clean_method, layer_debug=layer_debug)
    use_colligo_clean_plate = clean_method in ("colligo", "solid")
    bg_clean = bg_working.copy() if use_colligo_clean_plate else None
    tar_mask_mpi = image_dict["tar_mpi_mask"]

    depth_arr = depth_to_array(depth)
    depth_partition = mpi_depth_partition(depth_value, depth_arr)
    print(f"{_ts()} [gradio] Generate: MPI depth partition = {depth_partition}", flush=True)
    mpi_foreground_rgb, mpi_foreground_alpha = get_mpi_rgb_and_alpha(
        np.array(gt_image_cropped), depth_arr.astype(np.uint8), depth_partition
    )
    
    # Adjust MPI masks based on depth type
    if is_relative_depth:
        mpi_background_alpha, mpi_foreground_alpha = mpi_foreground_alpha[0], mpi_foreground_alpha[1]
        mpi_background_alpha = 1 - mpi_foreground_alpha
    else:
        mpi_background_alpha, mpi_foreground_alpha = mpi_foreground_alpha[1], mpi_foreground_alpha[0]
        mpi_background_alpha = 1 - mpi_foreground_alpha
    
    mpi_orig_mask = [mpi_background_alpha, mpi_foreground_alpha]
    
    # Resize MPI masks for model input
    mpi_foreground_alpha = cv2.resize(mpi_foreground_alpha, (64, 64), interpolation=cv2.INTER_NEAREST)
    mpi_background_alpha = cv2.resize(mpi_background_alpha, (64, 64), interpolation=cv2.INTER_NEAREST)
    mpi_foreground_alpha = torch.tensor(mpi_foreground_alpha, dtype=torch.float16).to(device).unsqueeze(0).unsqueeze(0)
    mpi_background_alpha = torch.tensor(mpi_background_alpha, dtype=torch.float16).to(device).unsqueeze(0).unsqueeze(0)

    # Prepare tensors for diffusion handles
    ten_img3 = torch.from_numpy(np.array(Image.fromarray(gt_image_cropped))).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    depth_fore = torch.tensor(np.array(depth)).unsqueeze(0).unsqueeze(0).to(device)

    # Null-text inversion (always fresh unless optional crop-hash disk cache enabled)
    cache_key = crop_cache_key(gt_image_cropped, depth_value, inv_prompt)
    use_disk_cache = null_text_disk_cache_enabled()
    null_text_cache_path, init_noise_cache_path = null_text_cache_paths(
        null_text_emb_path, gt_image_cropped, depth_value, inv_prompt
    )
    cache_hit = (
        use_disk_cache
        and os.path.exists(null_text_cache_path)
        and os.path.exists(init_noise_cache_path)
        and not do_null_text_again
    )
    inv_t0 = time.perf_counter()
    if cache_hit:
        print(
            f"{_ts()} [null-text cache HIT] key={cache_key} path={null_text_cache_path}",
            flush=True,
        )
        null_text_emb = torch.load(null_text_cache_path, map_location=device)
        ddim_latents = torch.load(init_noise_cache_path, map_location=device)
        init_noise = ddim_latents[-1].requires_grad_(True)
        cache_status = "hit"
    else:
        print(
            f"{_ts()} [null-text inversion RUN] key={cache_key} "
            f"ddim={null_text_ddim_steps()} inner={null_text_inner_steps()}",
            flush=True,
        )
        null_text_emb, ddim_latents = diff_handles.invert_input_image(ten_img3, depth_fore, prompt=inv_prompt)
        init_noise = ddim_latents[-1]
        cache_status = "run"
        torch.save(null_text_emb.detach().cpu(), os.path.join(current_save_dir, "null_text.pt"))
        ddim_latent = [latent.detach().cpu().numpy().tolist() for latent in ddim_latents]
        torch.save(torch.tensor(ddim_latent), os.path.join(current_save_dir, "init_noise.pt"))
        if use_disk_cache:
            torch.save(null_text_emb.detach().cpu(), null_text_cache_path)
            torch.save(torch.tensor(ddim_latent), init_noise_cache_path)
    inv_elapsed = time.perf_counter() - inv_t0
    print(f"{_ts()} [null-text inversion] {cache_status} in {inv_elapsed:.1f}s", flush=True)
    if run_logger is not None:
        run_logger.set_manifest(
            null_text_cache=cache_status,
            null_text_cache_key=cache_key,
            null_text_cache_path=null_text_cache_path if use_disk_cache else None,
            null_text_inversion_seconds=round(inv_elapsed, 2),
            null_text_ddim_steps=null_text_ddim_steps(),
            null_text_inner_steps=null_text_inner_steps(),
        )
    
    # Generate input image
    null_text_emb_fg, init_noise_fg, activations_fore, latent_image = diff_handles.generate_input_image(
        depth=depth_fore, prompt=inv_prompt, null_text_emb=null_text_emb, init_noise=init_noise
    )
    
    # Save reconstructed image
    with torch.no_grad():
        latent_image = diff_handles.diffuser.vae.decode(latent_image / diff_handles.diffuser.vae.config.scaling_factor, return_dict=False)[0]
        latent_image = VaeImageProcessor(vae_scale_factor=diff_handles.diffuser.vae.config.scaling_factor).postprocess(latent_image, output_type="pt")
        latent_image = latent_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        latent_image = (latent_image * 255).astype(np.uint8)
        cv2.imwrite(f"{current_save_dir}/reconstructed_image.jpg", cv2.cvtColor(latent_image, cv2.COLOR_RGB2BGR))
    
    # Prepare MPI data dictionary
    mpi_data_dict = {
        "do_mpi": do_mpi,
        "ddim_latents": ddim_latents,
        "mpi_masks": [mpi_background_alpha, mpi_foreground_alpha],
        "mpi_orig_mask": mpi_orig_mask,
        "activation_fore": activations_fore
        # "mpi_foreground_alpha": mpi_foreground_alpha,
        # "mpi_background_alpha": mpi_background_alpha,
        # "do_amodal_masking": False,
        # "do_latent_scaling": False,
        # "do_multi_diff": False,
        # "do_consistory": True,
        # "do_self_attn_masking": True,
        # "object_latents": None
    }

    if anydoor_hint_excludes_behind(genfill_mode()):
        edit_mask = anydoor_edit_mask(
            image_dict,
            gt_image_cropped.shape[:2],
            behind_mask=behind_mask,
            ref_alpha_crop=image_dict.get("ref_alpha_crop"),
        )
        image_dict = patch_image_dict_anydoor_hint(image_dict, edit_mask)
        cv2.imwrite(
            os.path.join(current_save_dir, "anydoor_edit_mask.png"),
            (np.clip(edit_mask, 0.0, 1.0) * 255).astype(np.uint8),
        )
        print(
            f"{_ts()} [anydoor] F3 hint excludes BEHIND — edit mask "
            f"{int((edit_mask > 0).sum())} px",
            flush=True,
        )
        if run_logger is not None:
            run_logger.set_manifest(anydoor_edit_mask_pixels=int((edit_mask > 0).sum()))
    
    # Generate final image
    generated_images = []
    for i in range(num_samples):
        gen_image = inference_single_image(
            ref_image, ref_mask, bg_working.copy(), tar_mask,
            mpi_data_dict, item=image_dict,
            sam_postprocess_dict=sam_postprocess_dict,
            guidance_scale=guidance_scale,
            curr_save_dir=current_save_dir,
            save_memory=save_memory,
            ddim_sampler=ddim_sampler,
            model=model,
            use_full_pred=use_colligo_clean_plate,
        )

        cv2.imwrite(
            os.path.join(current_save_dir, "anydoor_on_working_bg.jpg"),
            cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR),
        )

        gen_image = occlusion_aware_composite(
            bg_original,
            bg_working,
            gen_image,
            image_dict,
            removal_mask_crop=removal_mask,
            front_mask_crop=front_mask,
            behind_mask_crop=behind_mask,
            bg_clean_rgb=bg_clean,
            save_dir=current_save_dir,
        )
        cv2.imwrite(
            os.path.join(current_save_dir, "gradio_generated_image.png"),
            cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR),
        )
        
        # Convert to PIL image
        gen_image_pil = Image.fromarray(gen_image)
        generated_images.append(gen_image_pil)

    print(f"{_ts()} [gradio] Generate: finished; outputs in {os.path.abspath(current_save_dir)}", flush=True)
    return generated_images[0]

# Create Gradio interface
def create_demo():
    with gr.Blocks(title="MPI Object Placement Demo", css="""
        .gradio-container {
            max-width: 1600px !important;
        }
        .output-gallery {
            max-height: 600px;
        }
    """) as demo:
        gr.Markdown("# Zero shot Depth aware Object Placement")
        gr.Markdown("**Two-Step Process:** 1) Upload images and analyze depth. 2) Choose depth value and generate object placement.")
        gr.Markdown("**Instructions:** 1) Upload background image and draw a mask for bbox. 2) Upload reference object (RGBA). 3) Click 'Analyze Depth' to see depth distribution. 4) Adjust depth value based on analysis. 5) Generate the result.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Background Image & Mask")
                background_image = gr.Image(
                    label="Background Image (Draw mask where you want to place object preferably a box)", 
                    tool="sketch",
                    type="pil",
                    height=400
                )
                
                gr.Markdown("### Reference Object")
                reference_image = gr.Image(
                    label="Reference Object (RGBA format - white background, transparent object)", 
                    type="pil",
                    height=400
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Analyze Depth")
                analyze_button = gr.Button("Analyze Depth & SAM", variant="primary")
                
                gr.Markdown("### Step 2: Generation Parameters")
                depth_value = gr.Slider(
                    label="Depth Threshold",
                    minimum=10,
                    maximum=255,
                    value=50,
                    step=1,
                    info="Placement plane depth (Depth Anything units). SAME-layer removal uses this threshold."
                )
                
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    value=5.0,
                    step=0.5,
                    info="Controls how closely the generation follows the prompt"
                )
                
                num_samples = gr.Slider(
                    label="Number of Samples",
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    info="Number of variations to generate"
                )
                
                inv_prompt = gr.Textbox(
                    label="Prompt",
                    value="a photo of a sofa in a living room",
                    info="Prompt for null text inversion (used for generation and inversion)"
                )

                enable_clean_plate = gr.Checkbox(
                    label="Enable clean-plate (Colligo GenFill)",
                    value=False,
                    info="Requires Colligo host + token. If off or credentials missing, runs standard AnyDoor on the original photo (no Telea).",
                )

                colligo_host = gr.Textbox(
                    label="Colligo host",
                    value=os.environ.get(
                        "COLLIGO_HOST",
                        "https://clio-imaging-colligov2-stage.corp.ethos851-stage-or2.ethos.adobe.net",
                    ),
                    info="Stage root URL (no /debug). Uses COLLIGO_HOST env if left blank.",
                )

                colligo_token = gr.Textbox(
                    label="Colligo bearer token",
                    value=os.environ.get("COLLIGO_TOKEN", ""),
                    type="password",
                    info="Bearer token for v2/images/fill. Uses COLLIGO_TOKEN env if left blank.",
                )
                
                generate_button = gr.Button("Generate Object Placement", variant="primary")
                # clear_button = gr.Button("Clear All", variant="secondary")
        
        with gr.Row():
            gr.Markdown("### Step 1 Results: Depth Analysis")
        
        with gr.Row():
            with gr.Column():
                depth_3d_plot = gr.Image(
                    label="3D Depth Distribution (Top View)",
                    show_label=True,
                    height=300
                )
            
            with gr.Column():
                depth_front_plot = gr.Image(
                    label="Depth Distribution (Front View)",
                    show_label=True,
                    height=300
                )

            with gr.Column():
                removal_mask_preview = gr.Image(
                    label="Layer mask preview (red = GenFill removal, blue = BEHIND keep, green = FRONT)",
                    show_label=True,
                    height=300
                )
        
        # with gr.Row():
        #     with gr.Column():
        #         sam_visualization = gr.Image(
        #             label="SAM Segmentation Mask",
        #             show_label=True,
        #             height=300
        #         )
        
        with gr.Row():
            gr.Markdown("### Step 2 Results: Generated Images")
            # output_gallery = gr.Gallery(
            #     label="Generated Images",
            #     show_label=True,
            #     elem_id="gallery",
            #     columns=1,
            #     height=512
            # )
            output_image = gr.Image(label="Generated Image", 
                                    show_label=True,
                                    height=512)
        
        # Hidden component for image_dict
        image_dict = gr.State()
        
        # Status indicator
        status_text = gr.Textbox(
            label="Status",
            value="Ready — models load on first Analyze or Generate",
            interactive=False
        )
        
        # Tips section
        gr.Markdown("### Understanding the Depth Analysis")
        gr.Markdown("""
        **3D Depth Distribution (Top View)**: Shows the object depth distribution from top view above.
        
        **Depth Distribution (Front View)**: Shows the depth distribution from the front. Use this to understand object placement.
   
        """)
        
        gr.Markdown("### Tips for Best Results")
        gr.Markdown("""
        - **Background Image**: Use high-quality images with clear depth information
        - **Mask Drawing**: Draw the mask in the area where you want to place the object
        - **Reference Object**: Use RGBA images with transparent backgrounds for best results
        - **Depth Value**: 
          - empty space where object cl=an be place in the scene at required depth layer
        - **Guidance Scale**: Higher values (8-12) for more faithful generation, lower values (3-6) for more creative results
        """)
        
        # Connect the buttons
        def analyze_with_status(
            bg_img,
            ref_img,
            inv_prompt_val,
            depth_val,
            clean_plate_on,
            colligo_host_val,
            colligo_token_val,
        ):
            run_dir = make_timestamped_run_dir("analyze")
            metadata = {
                "inv_prompt": inv_prompt_val,
                "depth_value": int(round(float(depth_val))) if depth_val is not None else None,
                "enable_clean_plate": bool(clean_plate_on),
            }
            try:
                with capture_run_log(run_dir, stage="analyze", metadata=metadata) as logger:
                    logger.log("Analyze button clicked")
                    depth_3d, depth_front, removal_preview, img_dict, out_depth = analyze_depth_and_sam(
                        bg_img,
                        ref_img,
                        inv_prompt_val,
                        depth_value=depth_val,
                        enable_clean_plate=clean_plate_on,
                        colligo_host=colligo_host_val,
                        colligo_token=colligo_token_val,
                        run_dir=run_dir,
                        run_logger=logger,
                    )
                status = (
                    f"Depth analysis completed. Artifacts: {run_dir}  Log: {run_dir}/run.log"
                )
                return depth_3d, depth_front, removal_preview, img_dict, out_depth, status
            except Exception as e:
                raise gr.Error(f"{str(e)}  Log: {run_dir}/run.log") from e
        
        def generate_with_status(
            bg_img,
            ref_img,
            depth_val,
            img_dict,
            inv_prompt_val,
            guidance_val,
            num_samp,
            clean_plate_on,
            colligo_host_val,
            colligo_token_val,
        ):
            run_dir = make_timestamped_run_dir("generate")
            metadata = {
                "inv_prompt": inv_prompt_val,
                "depth_value": int(round(float(depth_val))) if depth_val is not None else None,
                "guidance_scale": float(guidance_val),
                "num_samples": int(num_samp),
                "enable_clean_plate": bool(clean_plate_on),
            }
            try:
                with capture_run_log(
                    run_dir,
                    stage="generate",
                    metadata=metadata,
                    mark_latest=True,
                ) as logger:
                    logger.log("Generate button clicked")
                    result = gradio_infer(
                        bg_img,
                        ref_img,
                        depth_val,
                        None,
                        inv_prompt_val,
                        guidance_val,
                        num_samp,
                        clean_plate_on,
                        colligo_host_val,
                        colligo_token_val,
                        run_dir=run_dir,
                        run_logger=logger,
                    )
                status = (
                    f"Generation completed. Artifacts: {run_dir}  Log: {run_dir}/run.log"
                )
                return result, status
            except Exception as e:
                raise gr.Error(f"{str(e)}  Log: {run_dir}/run.log") from e
        
        # Step 1: Analyze depth
        analyze_button.click(
            fn=analyze_with_status,
            inputs=[
                background_image,
                reference_image,
                inv_prompt,
                depth_value,
                enable_clean_plate,
                colligo_host,
                colligo_token,
            ],
            outputs=[depth_3d_plot, depth_front_plot, removal_mask_preview, image_dict, depth_value, status_text]
        )
        
        # Step 2: Generate images
        generate_button.click(
            fn=generate_with_status,
            inputs=[
                background_image,
                reference_image,
                depth_value,
                image_dict,
                inv_prompt,
                guidance_scale,
                num_samples,
                enable_clean_plate,
                colligo_host,
                colligo_token,
            ],
            outputs=[output_image, status_text]
        )
        
        # Clear function
        def clear_all():
            return None, None, 50, 5.0, 1, "a photo of a sofa in a livingroom", None, None, None, None
        
        # clear_button.click(
        #     fn=clear_all,
        #     outputs=[
        #         background_image,
        #         reference_image,
        #         depth_value,
        #         guidance_scale,
        #         num_samples,
        #         inv_prompt,
        #         depth_3d_plot,
        #         depth_front_plot,
        #         # sam_visualization,
        #         output_gallery,
        #         status_text
        #     ]
        # )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Change from 0.0.0.0 to your server ip
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        inbrowser=True,  # Changed to True to automatically open browser
        prevent_thread_lock=True
    )


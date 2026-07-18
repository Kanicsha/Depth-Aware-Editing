import os
import sys
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import cv2
import einops
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked_mpi_featguidance import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention, disable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from ldm.util import instantiate_from_config

from diffusers import DDIMScheduler
import sys
sys.path.append(".")
from src.featglac import FeatureGuidance
from diffusers.image_processor import VaeImageProcessor
from run_inference_object_placement import process_pairs, crop_back, inference_single_image

# Load depth anything and SAM for preprocessing
from utils.mpi.preprocess import *
from utils.mpi.postprocess import sam_postprocess, sam_postprocess2, get_sam_mask
from utils.mpi.mpi import get_mpi_rgb_and_alpha
from utils.mpi.null_text_inv import NullTextPipeline
from diffusers.schedulers import DDIMScheduler

# Initialize device and models
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize Diffusion Handles
diff_handles = FeatureGuidance(conf=None)
diff_handles.to(device)

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Load configuration and models
config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = config.pretrained_model
model_config = config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location=str(device)))
model = model.to(device)
ddim_sampler = DDIMSampler(model)

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

def analyze_depth_and_sam(background_image, reference_image, inv_prompt):
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
    
    bg_np = np.array(bg_image.convert("RGB"))
    
    # Process reference image (RGBA)
    ref_np = np.array(reference_image.convert("RGBA"))
    
    # Extract RGB and mask from reference
    ref_image = ref_np[:, :, :3]  # RGB channels
    ref_image = ref_image[:,:,::-1]
    ref_mask = ref_np[:, :, 3]    # Alpha channel
    ref_mask = (ref_mask > 128).astype(np.uint8) * 255
    
    # Clean up reference mask
    ref_mask = cv2.dilate(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    ref_mask = cv2.erode(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Convert target mask
    tar_mask = np.array(bbox_mask)
    
    # Process image pairs
    image_dict = process_pairs(ref_image, ref_mask, bg_np.copy(), tar_mask, shape_control=False)
    
    # Get cropped target image
    gt_image_cropped = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
    tar_mask_mpi = image_dict["tar_mpi_mask"]
    
    # Generate depth and SAM mask
    depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(gt_image_cropped), is_relative_depth=True)
    
    # Create depth analysis plots
    # depth_3d, depth_front, sam_viz = create_depth_analysis_plots(depth, sam_mask, tar_mask_mpi)
    tar_mask_mpi_copy = np.ones_like(tar_mask_mpi) * 255
    plot_depth_bins(depth, sam_mask, tar_mask_mpi_copy, input_img_name="gradio_infer", save_dir="./results/object_placement/gradio_infer", is_crop=True)

    depth_top_path = f"./results/object_placement/gradio_infer/gradio_infer_depth_bins_front_crop.png"
    depth_front_path = f"./results/object_placement/gradio_infer/gradio_infer_depth_bins_crop.png"

    depth_3d_plot = Image.open(depth_top_path)
    depth_front_plot = Image.open(depth_front_path)
    
    # Calculate suggested depth value based on depth distribution
    depth_array = np.array(depth)
    mask_region = np.array(tar_mask_mpi)
    
    # Get depth values in the mask region
    if len(mask_region.shape) == 3:
        masked_depth = depth_array * mask_region[:, :, 0]
    else:
        masked_depth = depth_array * mask_region
    
    # Calculate statistics for suggested depth
    non_zero_depths = masked_depth[masked_depth > 0]
    if len(non_zero_depths) > 0:
        suggested_depth = int(np.percentile(non_zero_depths, 75))  # 75th percentile
        suggested_depth = max(10, min(200, suggested_depth))  # Clamp to valid range
    else:
        suggested_depth = 50  # Default value
    
    return depth_3d_plot, depth_front_plot, image_dict, suggested_depth

def gradio_infer(background_image, reference_image, depth_value, image_dict, inv_prompt, guidance_scale=5.0, num_samples=1):
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
    
    # Input validation
    if background_image is None or reference_image is None:
        raise gr.Error("Please upload both background and reference images.")
    
    if "mask" not in background_image:
        raise gr.Error("Please draw a mask on the background image.")
    
    # Configuration
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './results/object_placement'
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
    
    # Create save directory
    current_save_dir = os.path.join(save_dir, 'gradio_infer')
    os.makedirs(current_save_dir, exist_ok=True)
    
    # Process background image and mask
    bg_image = background_image["image"]
    bbox_mask, bg_image = extract_bbox_mask(background_image, bg_image)
    
    if bbox_mask is None:
        raise gr.Error("Could not extract mask from background image.")
    
    bg_np = np.array(bg_image.convert("RGB"))
    
    # Process reference image (RGBA)
    ref_np = np.array(reference_image.convert("RGBA"))
    
    # Extract RGB and mask from reference
    ref_image = ref_np[:, :, :3]  # RGB channels
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = ref_np[:, :, 3]    # Alpha channel
    ref_mask = (ref_mask > 128).astype(np.uint8) * 255
    
    # Clean up reference mask
    ref_mask = cv2.dilate(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    ref_mask = cv2.erode(ref_mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Convert target mask
    tar_mask = np.array(bbox_mask)
    
    # Process image pairs
    if image_dict is None:
        image_dict = process_pairs(ref_image, ref_mask, bg_np.copy(), tar_mask, shape_control=enable_shape_control)
    
    # Get cropped target image
    gt_image_cropped = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
    tar_mask_mpi = image_dict["tar_mpi_mask"]
    
    # Generate depth and SAM mask
    depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(gt_image_cropped), is_relative_depth)
    
    # MPI processing
    depth_partition = [(0, depth_value), (depth_value, 300)]
    mpi_foreground_rgb, mpi_foreground_alpha = get_mpi_rgb_and_alpha(
        np.array(gt_image_cropped), np.array(depth), depth_partition
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

    # Load or create null text embeddings
    if os.path.exists(f"{null_text_emb_path}/{image_name}_{inv_prompt}_null_text.pt") and not do_null_text_again:
        null_text_emb = torch.load(f"{null_text_emb_path}/{image_name}_{inv_prompt}_null_text.pt").to(device)
        ddim_latents = torch.load(f"{null_text_emb_path}/{image_name}_{inv_prompt}_init_noise.pt").to(device)
        init_noise = ddim_latents[-1].requires_grad_(True)
    else:
        null_text_emb, ddim_latents = diff_handles.invert_input_image(ten_img3, depth_fore, prompt=inv_prompt)
        init_noise = ddim_latents[-1]
        torch.save(null_text_emb.detach().cpu(), f"{null_text_emb_path}/{image_name}_{inv_prompt}_null_text.pt")
        ddim_latent = [latent.detach().cpu().numpy().tolist() for latent in ddim_latents]
        torch.save(torch.tensor(ddim_latent), f"{null_text_emb_path}/{image_name}_{inv_prompt}_init_noise.pt")
    
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
    
    # Generate final image
    generated_images = []
    for i in range(num_samples):
        gen_image = inference_single_image(
            ref_image, ref_mask, bg_np.copy(), tar_mask, 
            mpi_data_dict, item=image_dict, 
            sam_postprocess_dict=sam_postprocess_dict, 
            guidance_scale=guidance_scale,
            curr_save_dir=current_save_dir,
            save_memory=save_memory,
            ddim_sampler=ddim_sampler,
            model=model
        )
        
        # Convert to PIL image
        gen_image_pil = Image.fromarray(gen_image)
        gen_image_pil.save("./results/object_placement/gradio_infer/gradio_generated_image.png")
        generated_images.append(gen_image_pil)

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
                    maximum=300,
                    value=50,
                    step=5,
                    info="Controls the depth separation between foreground and background"
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
            value="Ready to generate",
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
        def analyze_with_status(bg_img, ref_img, inv_prompt_val):
            try:
                status_text.update("Analyzing depth and SAM segmentation...")
                result = analyze_depth_and_sam(bg_img, ref_img, inv_prompt_val)
                status_text.update("Depth analysis completed! Adjust depth value and generate.")
                return result
            except Exception as e:
                status_text.update(f"Error: {str(e)}")
                raise e
        
        def generate_with_status(bg_img, ref_img, depth_val, img_dict, inv_prompt_val, guidance_val, num_samp):
            try:
                status_text.update("Generating object placement...")
                result = gradio_infer(bg_img, ref_img, depth_val, img_dict, inv_prompt_val, guidance_val, num_samp)
                status_text.update("Generation completed successfully!")
                return result
            except Exception as e:
                status_text.update(f"Error: {str(e)}")
                raise e
        
        # Step 1: Analyze depth
        analyze_button.click(
            fn=analyze_with_status,
            inputs=[background_image, reference_image, inv_prompt],
            outputs=[depth_3d_plot, depth_front_plot, image_dict, depth_value]
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
                num_samples
            ],
            # outputs=output_gallery
            outputs=output_image
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


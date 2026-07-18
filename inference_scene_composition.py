import os
import json
from collections import OrderedDict
import argparse
import tempfile

import sys
sys.path.append('.')
import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)  # Triggers fresh download of MiDaS repo
import numpy as np
from omegaconf import OmegaConf
from src.featglac.feat_guidance import FeatureGuidance
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import cv2
from src.featglac.get_depth import get_depth_map
from transformers import pipeline

config_path = "./src/featglac/config/default_sc.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
diff_handles = FeatureGuidance(conf=diff_handles_config)
diff_handles.to(device)

# load image and mask
input_img_name = "ab26_07"
bg_name = "bg07.png"
bg_img = Image.open(f"./examples/Gradio/scene_comp_data/{bg_name}").convert("RGB").resize((512,512))
bg_name = bg_name.split(".")[0]

fg_img = Image.open("./examples/Gradio/scene_comp_data/ab26.jpg").convert("RGB").resize((512,512))
fg_mask = Image.open("./examples/Gradio/scene_comp_data/ab26_mask.png").convert("RGB").resize((512,512))
fg_mask = np.array(fg_mask) // 255

def colorise_depth(depth_map):
    max_depth = np.max(depth_map)
    min_depth = np.min(depth_map)
    depth_map = ((depth_map - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_map


compose_prompt = "a photo of old car in a dark garage with blue tint"
inv_prompt_bg = "a photo a empty dark garage with blue tint"
inv_prompt_fg = "a photo of old car in a empty road"
num_of_samples = 3

save_dir = f"./results/scene_comp/{input_img_name}"
null_text_emb_path = "./examples/Gradio/null_embed"
os.makedirs(null_text_emb_path, exist_ok=True)
print("Null text emb path: ", null_text_emb_path)
os.makedirs(save_dir, exist_ok=True)


# depth_fore = depth_pipe(fg_img)["depth"]
depth_fore = Image.fromarray(get_depth_map(fg_img))

depth_fore_color = colorise_depth(np.array(depth_fore))
cv2.imwrite("./{}/{}_depth_fore.jpg".format(save_dir,input_img_name), depth_fore_color)

depth_fore = np.array(depth_fore)

# depth_back = depth_pipe(bg_img)["depth"]
depth_back = Image.fromarray(get_depth_map(bg_img))
depth_back_color = colorise_depth(np.array(depth_back))
cv2.imwrite("./{}/{}_depth_back.jpg".format(save_dir,input_img_name), depth_back_color)
# depth_back_color.save("./{}/{}_depth_back.jpg".format(save_dir,input_img_name))
depth_back = np.array(depth_back)

depth_control = depth_fore * fg_mask[:,:,0] + depth_back * (1 - fg_mask[:,:,0])
depth_control_img = depth_control.copy()
depth_control_col = colorise_depth(depth_control)
cv2.imwrite("./{}/{}_depth_control.jpg".format(save_dir,input_img_name), depth_control_col)
depth_control = torch.from_numpy(depth_control).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)

image_mpi = np.array(bg_img) * (1 - fg_mask) + np.array(fg_img) * fg_mask
cv2.imwrite("./{}/{}_image_mpi.jpg".format(save_dir,input_img_name), cv2.cvtColor(image_mpi, cv2.COLOR_RGB2BGR))

depth_back = torch.tensor(depth_back).unsqueeze(0).unsqueeze(0).to(device)
ten_img2 = torch.tensor(np.array(bg_img)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

if(os.path.exists(f"{null_text_emb_path}/{bg_name}_{inv_prompt_bg}_null_text.pt")):
    null_text_emb = torch.load(f"{null_text_emb_path}/{bg_name}_{inv_prompt_bg}_null_text.pt").to(device)
    init_noise = torch.load(f"{null_text_emb_path}/{bg_name}_{inv_prompt_bg}_init_noise.pt").to(device)
else:
    null_text_emb, init_noise = diff_handles.invert_input_image(ten_img2, depth_back, inv_prompt_bg)
    init_noise = init_noise[-1]
    print("null_text_emb shape: ", null_text_emb.shape, null_text_emb.dtype)
    print("init_noise shape: ", init_noise.shape, init_noise.dtype)
    torch.save(null_text_emb.detach().cpu(), f"{null_text_emb_path}/{bg_name}_{inv_prompt_bg}_null_text.pt")
    torch.save(init_noise.detach().cpu(), f"{null_text_emb_path}/{bg_name}_{inv_prompt_bg}_init_noise.pt")
null_text_emb_bg, init_noise_bg, activations_back, latent_image = diff_handles.generate_input_image(
                depth=depth_back, prompt=inv_prompt_bg, null_text_emb=null_text_emb, init_noise=init_noise)

# save image reconstructed from inversion
with torch.no_grad():
    latent_image = diff_handles.diffuser.vae.decode(latent_image / diff_handles.diffuser.vae.config.scaling_factor, return_dict=False)[0]
    latent_image = VaeImageProcessor(vae_scale_factor=diff_handles.diffuser.vae.config.scaling_factor).postprocess(latent_image, output_type="pt")
    latent_image = latent_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    latent_image = (latent_image * 255).astype(np.uint8)
    cv2.imwrite("./{}/{}_recon_bg.jpg".format(save_dir,input_img_name), cv2.cvtColor(latent_image, cv2.COLOR_RGB2BGR))

depth_fore = torch.tensor(depth_fore).unsqueeze(0).unsqueeze(0).to(device)
ten_img3 = torch.tensor(np.array(fg_img)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
if(os.path.exists(f"{null_text_emb_path}/{input_img_name}_{inv_prompt_fg}_null_text.pt")):
    null_text_emb = torch.load(f"{null_text_emb_path}/{input_img_name}_{inv_prompt_fg}_null_text.pt").to(device)
    init_noise = torch.load(f"{null_text_emb_path}/{input_img_name}_{inv_prompt_fg}_init_noise.pt").to(device)
else:
    null_text_emb, init_noise = diff_handles.invert_input_image(ten_img3, depth_fore, inv_prompt_fg)
    init_noise = init_noise[-1]
    torch.save(null_text_emb.detach().cpu(), f"{null_text_emb_path}/{input_img_name}_{inv_prompt_fg}_null_text.pt")
    torch.save(init_noise.detach().cpu(), f"{null_text_emb_path}/{input_img_name}_{inv_prompt_fg}_init_noise.pt")

null_text_emb_fg, init_noise_fg, activations_fore, latent_image = diff_handles.generate_input_image(
                depth=depth_fore, prompt=inv_prompt_fg, null_text_emb=null_text_emb, init_noise=init_noise)

down_sampled_mask = torch.nn.functional.interpolate(torch.tensor(fg_mask[:,:,0]).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device),
                                                     size=(64, 64), mode='nearest')

# save image reconstructed from inversion
with torch.no_grad():
    latent_image = diff_handles.diffuser.vae.decode(latent_image / diff_handles.diffuser.vae.config.scaling_factor, return_dict=False)[0]
    latent_image = VaeImageProcessor(vae_scale_factor=diff_handles.diffuser.vae.config.scaling_factor).postprocess(latent_image, output_type="pt")
    latent_image = latent_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    latent_image = (latent_image * 255).astype(np.uint8)
    cv2.imwrite("./{}/{}_recon_fg.jpg".format(save_dir,input_img_name), cv2.cvtColor(latent_image, cv2.COLOR_RGB2BGR))

# save activations
mpi_masks = [1 - fg_mask, fg_mask]
activations = [activations_back, activations_fore]

# do scene composition
for i in range(num_of_samples):
    results = diff_handles.mpi_scene_comp(
                    depth=depth_control, prompt=compose_prompt,
                    mpi_masks=mpi_masks,
                    null_text_emb=None, init_noise=init_noise_bg,
                    activations=activations,
                    use_input_depth_normalization=False)

    if diff_handles.conf.guided_diffuser.save_denoising_steps:
        edited_img, edited_disparity, denoising_steps = results
    else:
        edited_img, edited_disparity = results
        denoising_steps = None

    edited_img = edited_img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    edited_img = (edited_img * 255).astype(np.uint8)
    cv2.imwrite("./{}/{}_scene_comp_{}.png".format(save_dir,input_img_name,i), cv2.cvtColor(edited_img, cv2.COLOR_RGB2BGR))

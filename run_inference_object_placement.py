import os
import cv2
import einops
import numpy as np
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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_models():
    diff_handles = FeatureGuidance(conf=None)
    diff_handles.to(device)

    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()

    config = OmegaConf.load('./configs/inference.yaml')
    model_ckpt =  config.pretrained_model
    model_config = config.config_file

    model = create_model(model_config ).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location=str(device)))
    model = model.to(device)
    ddim_sampler = DDIMSampler(model)

    return diff_handles, model, ddim_sampler, save_memory

def aug_data_mask(image, mask):
    transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def plot_intermediate(intermediates, save_path):
    pred_x0_list = intermediates["pred_x0"]
    for i, pred_x0 in enumerate(pred_x0_list):
        x_samples = model.decode_first_stage(pred_x0)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

        result = x_samples[0][:,:,::-1]
        result = np.clip(result,0,255)
        cv2.imwrite("./{}/{}.png".format(save_path, i), result)
    print("intermediate images saved")
        

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, shape_control=False):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = (np.stack([ref_mask,ref_mask,ref_mask],-1) > 0).astype(ref_image.dtype)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]
    # cv2.imwrite("masked_ref_image.png", masked_ref_image)

    ratio = np.random.randint(11, 12) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = (np.stack([ref_mask,ref_mask,ref_mask],-1) > 0).astype(ref_image.dtype)
    # cv2.imwrite("masked_ref_image_expand.png", masked_ref_image)

    ### added
    masked_ref_img_for_collage = masked_ref_image.copy()
    ref_mask_for_collage = ref_mask.copy()

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)
    # cv2.imwrite("masked_ref_image_resized.png", masked_ref_image)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]
    # cv2.imwrite("ref_mask_resized.png", ref_mask)

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    ## cahnegd
    masked_ref_image_compose = masked_ref_img_for_collage
    ref_mask_compose = (ref_mask_for_collage > 0).astype(np.uint8) * 255
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)


    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) # 1.1,1.2

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3]) # 1.5, 3   #1.2 1.6
    tar_box_yyxx_crop = new_box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    # transforming mask for mpi according to tar_image
    tar_mask_mpi = tar_mask.copy()
    tar_mask_mpi_cropped = tar_mask_mpi[y1:y2,x1:x2]
    tar_mask_mpi_cropped = cv2.cvtColor(tar_mask_mpi_cropped, cv2.COLOR_GRAY2RGB)

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx
    # cv2.imwrite("cropped_target_image.png", cropped_target_image)

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
    # cv2.imwrite("ref_image_collage.png", ref_image_collage)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    if(shape_control):
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    ref_alpha_crop = np.zeros((collage.shape[0], collage.shape[1]), dtype=np.uint8)
    ref_alpha_crop[y1:y2, x1:x2] = (ref_mask_compose * 255).astype(np.uint8)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)
    # cv2.imwrite("cropped_target_image_pad.png", cropped_target_image)
    
    # transforming mask for mpi according to tar_image
    tar_mask_mpi_cropped = pad_to_square(tar_mask_mpi_cropped, pad_value = 0, random = False).astype(np.uint8)
    tar_mask_mpi_cropped = cv2.resize(tar_mask_mpi_cropped, (512,512), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop), tar_mpi_mask = tar_mask_mpi_cropped, object_bbox_for_sam = np.array(tar_box_yyxx),
                ref_alpha_crop=ref_alpha_crop.copy())
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 10 # maigin_pixel 5

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image

def del_model_and_sampler(model, ddim_sampler):
    del model
    del ddim_sampler
    torch.cuda.empty_cache()

def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, mpi_data_dict, item=None, sam_postprocess_dict=None, guidance_scale = 5.0,
                           curr_save_dir=None, save_memory=False, ddim_sampler=None, model=None, use_full_pred=False,
                           ddim_steps=None):
    if item is None:
        item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    cv2.imwrite("ref_image.png", ref[:,:,::-1])
    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().to(device)
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().to(device)
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    #amodal conditioning

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    if ddim_steps is None:
        ddim_steps = 50
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    mpi_data=mpi_data_dict,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond,
                                                    log_every_t=1)

    mpi_data_dict["object_latents"] = intermediates["x_inter"]

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    # result = x_samples[0][:,:,::-1]
    # result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[:,:,:]

    tag = "w_mpi" if mpi_data_dict["do_mpi"] else "wo_mpi"
    cv2.imwrite(os.path.join(curr_save_dir, f"anydoor_orig_gen_{tag}.png"), pred[:,:,::-1])
    orig_pred = pred.copy()

    ## saving ours anydoor results
    pred_anydoor = orig_pred if use_full_pred else orig_pred[1:, :, :]
    
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image_anydoor = crop_back(pred_anydoor, tar_image, sizes, tar_box_yyxx_crop) 
    # cv2.imwrite(os.path.join(curr_save_dir, f"anydoor_gen_{tag}.png"), gen_image_anydoor[:,:,::-1])


    return gen_image_anydoor


if __name__ == '__main__': 
  
    # load depthanything and sam and do necessary preprocessing
    from utils.mpi.preprocess import *
    from utils.mpi.mpi import get_mpi_rgb_and_alpha


    diff_handles, model, ddim_sampler, save_memory = load_models()

    plot_depth = True
    depth_value = 50
    do_mpi = True
    is_relative_depth = True
    enable_shape_control = False
    sam_postprocess_dict = None
    inv_prompt = "a photo of a white sofa in a living room" # used in multi diff only check where


    # dict containing image name and its corresponding depth value were object can be placed
    relative_depth_dict = {"z1":160, "z2":50, "z3":170, "z4":100, "z5":50, "z6":100, "z7":160, "z8":60,"c38":30, "c39":30,"c16":110,"c9":55, "c18":80,
                           "c5":1400, "sofa11":50, "c10":60,"c15":50,"t2":120,"t4":90, "t5":90,"c27":90,"t6":150,"t8":100,
                           "t10":90, "t7":70, "c45":110, "c4":125, "t1":60, "t3":90, "t5_new":90, "t3_new":90, "c32":100,"c40":75, # 100
                           "c36":120, "c25":120, "c30":110, "ab2":60, "t51":120, "t52":90, "c3":150, "empty_room":200,"t13":80,  # 90
                           "t14":70, "t15":100, "16":100, "t14":70, "beanbag":110, "20":100, "21":70, "19":100, "22":120, "23":100,  # 21:70
                           "24":90, "c27_toy4":90, "27":50, "28":150,"29":160, "30":300, "31":120, "32":120, "33":100, "34":300,"36":70, # 34:90
                           "35":110, "37":60, "38":70, "39":90, "40":80, "41":120, "45": 70, "46":50, "47":90, "48":80, "39_new":300, # 42_new1:35
                           "47_new":300, "35_new":100, "44":40, "44_new":300, "20_new":300, "42":100, "42_new1":35, "35_new":300,
                           "46_new":300, "43":300, "10":300, "12":120, "13":150, "14":100, "3":50, "5":30, "2":100, "7":120, "8":70, "11":80} # 23: 40, 24:80 t5: 70, 40, 90, t6:50
    #'''
    # ==== Example for inferring VITON-HD Test dataset ===

    from omegaconf import OmegaConf
    import os 
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './results/object_placement'
    null_text_emb_path = "./examples/Gradio/null_embed"
    os.makedirs(null_text_emb_path, exist_ok=True)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    test_bg_dir = os.path.join(test_dir,"new_samples_1")
    image_names = []
    target_mask_list = []
    for file_name in sorted(os.listdir(test_bg_dir)):
        if("mask" not in file_name):
            image_names.append(file_name)

    print("image names", image_names)
    idx_ = -1
    image_names = image_names[idx_:]
    print("background image name :", image_names)

    for image_name in os.listdir(test_bg_dir.replace('/new_samples_1/', '/new_samples_1_mask/')):
        if("mask" in image_name):
            target_mask_list.append(image_name)

    ref_image_path = os.path.join(test_dir, 'FG_object1')
    ref_img_list = sorted(os.listdir(ref_image_path))
    print(" ref)img_list : ", ref_img_list)
    ob_id_ = -25   # -29 bluw vase, 34 statue, 27 lamp, chair 19 -41 back truck, -36 school bus f2, 40 toy1
    do_null_text_again = False
    ref_img_list = ref_img_list[ob_id_:ob_id_+1]
    ref_img_path = os.path.join(ref_image_path, ref_img_list[np.random.randint(0, len(ref_img_list))])
    
    video_list = []
    for image_name in image_names:
        print(image_name)
        depth_value = relative_depth_dict.get(image_name.split(".")[0], 300)

        current_save_sir = os.path.join(save_dir, image_name.split(".")[0])
        os.makedirs(current_save_sir, exist_ok=True)
        seed_everything(42)

        tar_image_path = os.path.join(test_bg_dir, image_name)
        tar_mask_path = tar_image_path.replace('/new_samples_1/', '/new_samples_1_mask/')
        tar_mask_name = tar_mask_path.split("/")[-1].split(".")[0]
        tar_mask_path = os.path.join(test_bg_dir.replace('/new_samples_1', '/new_samples_1_mask'), tar_mask_name + "_mask_0.jpg")
        

        if(os.path.exists(tar_mask_path) == False):
            if(tar_mask_path.endswith('.png')):
                tar_mask_path = tar_mask_path.replace('.png','.jpg')
            else:
                tar_mask_path = tar_mask_path.replace('.jpg','.png')
        else:
            print("mask path exists")

        ref_image = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)

        ref_image = ref_image[:,:,:-1]
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = ref_image[:,:,-1]
        ref_mask = (ref_mask > 10).astype(np.uint8) * 255
        ref_mask = cv2.dilate(ref_mask, np.ones((5,5), np.uint8), iterations=1)
        ref_mask = cv2.erode(ref_mask, np.ones((5,5), np.uint8), iterations=1) // 255

        gt_image = cv2.imread(tar_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        tar_mask = Image.open(tar_mask_path ).convert('L')
        tar_mask= np.array(tar_mask)

        # do preprocessing
        image_dict = process_pairs(ref_image, ref_mask, gt_image.copy(), tar_mask, shape_control=enable_shape_control)
        y1, y2, x1, x2 = image_dict["tar_box_yyxx_crop"]

        oy1, oy2, ox1, ox2 = image_dict["object_bbox_for_sam"]
        # resize sam mask to 512
        r = 512 / (y2 - y1)
        image_dict["object_bbox_for_sam"] = [oy1 * r, oy2 * r, ox1 * r, ox2 * r]


        gt_image_cropped = ((image_dict["jpg"] * 127.5) + 127.5).astype(np.uint8)
        tar_mask_mpi = image_dict["tar_mpi_mask"]


        depth, sam_mask = get_depth_and_sam_mask(Image.fromarray(gt_image_cropped), is_relative_depth)
            # depth.save("{}/depth.png".format(current_save_sir))

        if(plot_depth):
            print("tar_mask shape: ", tar_mask_mpi.shape)
            tar_mask_mpi_copy = np.ones_like(tar_mask_mpi) * 255
            plot_depth_bins(depth, sam_mask, tar_mask_mpi_copy, image_name, current_save_sir, is_crop=True)
        
        # depth_partition = [(0, depth_value), (depth_value, 300)]
        depth_partition = [(0, depth_value), (depth_value, 300)]

        # Get layered depth mask
        mpi_foreground_rgb, mpi_foreground_alpha = get_mpi_rgb_and_alpha(np.array(gt_image_cropped), np.array(depth), depth_partition)

        cv2.imwrite("{}/mpi_foreground_alpha.png".format(current_save_sir), np.array(gt_image_cropped) * mpi_foreground_alpha[1][:,:,None])
        cv2.imwrite("{}/mpi_background_alpha.png".format(current_save_sir), np.array(gt_image_cropped) * mpi_foreground_alpha[0][:,:,None])
  
        if(is_relative_depth):
            mpi_background_alpha, mpi_foreground_alpha = mpi_foreground_alpha[0], mpi_foreground_alpha[1]
            mpi_background_alpha = 1 - mpi_foreground_alpha
        else:
            mpi_background_alpha, mpi_foreground_alpha = mpi_foreground_alpha[1], mpi_foreground_alpha[0]
            mpi_background_alpha = 1 - mpi_foreground_alpha

        mpi_orig_mask = [mpi_background_alpha, mpi_foreground_alpha]
        mpi_foreground_alpha = cv2.resize(mpi_foreground_alpha, (64, 64), interpolation=cv2.INTER_NEAREST)

        mpi_background_alpha = cv2.resize(mpi_background_alpha, (64, 64), interpolation=cv2.INTER_NEAREST)
        mpi_foreground_alpha = torch.tensor(mpi_foreground_alpha, dtype=torch.float16).to(device).unsqueeze(0).unsqueeze(0)
        mpi_background_alpha = torch.tensor(mpi_background_alpha, dtype=torch.float16).to(device).unsqueeze(0).unsqueeze(0)

        # Do null text inversion
        ten_img3 = torch.from_numpy(np.array(Image.fromarray(gt_image_cropped))).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        depth_fore = torch.tensor(np.array(depth)).unsqueeze(0).unsqueeze(0).to(device)

        from utils.mpi.null_text_config import (
            null_text_cache_paths,
            null_text_disk_cache_enabled,
        )

        gt_crop_rgb = np.array(Image.fromarray(gt_image_cropped))
        nt_path, in_path = null_text_cache_paths(
            null_text_emb_path, gt_crop_rgb, depth_value, inv_prompt
        )
        if null_text_disk_cache_enabled() and os.path.exists(nt_path) and os.path.exists(in_path) and not do_null_text_again:
            null_text_emb = torch.load(nt_path, map_location=device)
            ddim_latents = torch.load(in_path, map_location=device)
            init_noise = ddim_latents[-1]
        else:
            null_text_emb, ddim_latents = diff_handles.invert_input_image(ten_img3, depth_fore, prompt=inv_prompt)
            init_noise = ddim_latents[-1]
            if null_text_disk_cache_enabled():
                torch.save(null_text_emb.detach().cpu(), nt_path)
                ddim_latent = [latent.detach().cpu().numpy().tolist() for latent in ddim_latents]
                torch.save(torch.tensor(ddim_latent), in_path)

        null_text_emb_fg, init_noise_fg, activations_fore, latent_image = diff_handles.generate_input_image(
                        depth=depth_fore, prompt=inv_prompt, null_text_emb=null_text_emb, init_noise=init_noise)

        # recontruct image to check if inversion is correct
        with torch.no_grad():
            latent_image = diff_handles.diffuser.vae.decode(latent_image / diff_handles.diffuser.vae.config.scaling_factor, return_dict=False)[0]
            latent_image = VaeImageProcessor(vae_scale_factor=diff_handles.diffuser.vae.config.scaling_factor).postprocess(latent_image, output_type="pt")
            latent_image = latent_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            latent_image = (latent_image * 255).astype(np.uint8)
            cv2.imwrite("./{}/recon_fg.jpg".format(current_save_sir), cv2.cvtColor(latent_image, cv2.COLOR_RGB2BGR))


        fg_object_latents = None
        

        mpi_data_dict = {"ddim_latents": ddim_latents, "mpi_masks": [mpi_background_alpha, mpi_foreground_alpha], "do_mpi": do_mpi,
                        "mpi_orig_mask": mpi_orig_mask,
                        "fg_object_latents": fg_object_latents,
                        "activation_fore": activations_fore}

        gen_path = os.path.join(current_save_sir, image_name.split(".")[0] + "_mpi.png")
        gen_single = os.path.join(current_save_sir, 
            image_name.split(".")[0] + f"_obj_place_result.png")
   
        gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask, mpi_data_dict, 
                                        item=image_dict, sam_postprocess_dict=sam_postprocess_dict, guidance_scale=5.0,
                                        curr_save_dir=current_save_sir, save_memory=save_memory, ddim_sampler=ddim_sampler, model=model)
        cv2.imwrite(gen_single, gen_image[:,:,::-1])

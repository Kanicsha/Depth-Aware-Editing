import os
import numpy as np
import cv2
from transformers import pipeline, SamModel, SamProcessor
import torch
from PIL import Image
import sys
sys.path.append(".")
from utils.mpi.get_depth import get_depth_map

if torch.cuda.is_available():
    _device = 0
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cpu")

sam_model = pipeline("mask-generation", model="facebook/sam-vit-huge", device="cpu",
                     torch_dtype=torch.float32)

def get_ddim_inverted_latents(nt_pipeline, image, prompt, num_inference_steps=50):
    latent = nt_pipeline.image2latent(image)
    context = nt_pipeline.get_context(prompt)
    latents = nt_pipeline.ddim_inversion_loop(latent, context, num_inference_steps=num_inference_steps)
    return latents

def get_null_text_latents(nt_pipeline, image, prompt, num_inference_steps=50, num_of_optimization_steps=10, early_stop_epsilon=1e-5):
    ddim_latents = get_ddim_inverted_latents(nt_pipeline, image, prompt, num_inference_steps)
    context = nt_pipeline.get_context(prompt)
    null_text_emb = nt_pipeline.null_optimization(ddim_latents, context, num_of_optimization_steps, early_stop_epsilon)
    null_text_emb = torch.stack(null_text_emb, 0)
    null_text_reconstruction, null_text_latents = nt_pipeline(prompt, null_text_emb, ddim_latents[-1])
    null_text_reconstruction.images[0].save("null_text_reconstruction.jpg")
    return null_text_latents

def get_depth_and_sam_mask(image, is_relative_depth=True):
    if(is_relative_depth):
        depth = depth_pipe(image)["depth"]
    else:
        actual_depth, visualise_depth = get_depth_map(image)
        depth = (actual_depth, visualise_depth)
    # ensure image is a plain PIL RGB image (no MPS tensors attached)
    image = Image.fromarray(np.array(image).astype(np.uint8))
    outputs = sam_model(image, points_per_batch=64)
    masks = outputs["masks"]
    final_mask = torch.zeros_like(torch.tensor(masks[0]))
    for i, mask in enumerate(masks):
        final_mask += mask * (i+1)
    final_mask = final_mask.cpu().numpy()
    final_mask = Image.fromarray((final_mask).astype(np.uint8))
    return depth, final_mask
        
def get_bins_for_depth(sam_mask, depth_map, mask_region):
    # get the depth values for the mask region in the image and group them into bins based on sam mask
    if(len(mask_region.shape) == 3):
        masked_sam = cv2.bitwise_and(sam_mask, sam_mask, mask=mask_region[:,:,0])
    else:
        cv2.imwrite("mask_region.jpg", mask_region)
        cv2.imwrite("sam_mask.jpg", sam_mask)
        print("sam mask size, mask size :", sam_mask.shape, mask_region.shape)
        masked_sam = cv2.bitwise_and(sam_mask, sam_mask, mask=mask_region)
    # number of unique values in sam mask
    possible_object_id = np.unique(sam_mask)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray']
    print("possible_object_id", possible_object_id)
    num_bins = len(possible_object_id)
    bins = [[] for _ in range(num_bins)]
    for j in range(num_bins):
        if(possible_object_id[j] == 0):
            continue
        object_mask = (masked_sam == possible_object_id[j])
        object_mask = cv2.erode(object_mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=2)

        # if(np.sum(np.array(object_mask)) < 1000):
        #     # print("object {} is too small".format(j))
        #     continue
        
        color = colors[np.random.randint(0, len(colors))]
        object_depth = depth_map * object_mask
        bins[j] = {"pc": np.where(object_depth != 0), "color": color}
    return bins

#get bbox coord and size from depth map
def get_correct_dimensions(x_close, x_far, depth_close, depth_far, depth_value, is_coord_x=False, num_of_bins=40, is_realtive_depth=True, debug=False):
    if(is_realtive_depth):
        depth_bin = [depth_close - i * (depth_close - depth_far) / num_of_bins for i in range(num_of_bins)]
    else:
        depth_bin = [depth_close + i * (depth_far - depth_close) / num_of_bins for i in range(num_of_bins)][::-1]

    print("depth_bin", depth_bin)
    if(not is_coord_x or (is_coord_x and x_close > x_far)):
        depth_index = np.argmin(np.abs(np.array(depth_bin) - depth_value)) - 1
        coeff = [1.0 / i for i in range(1, num_of_bins+1)]
        alpha = (x_close - x_far) / sum(coeff)
        curr_coff = [1.0/i for i in range(1, depth_index+1)]
        curr_x = x_close - sum([alpha * i for i in curr_coff])
        y_coor_list = []
        if(debug):
            for j in range(num_of_bins):
                curr_coff = [1.0/i for i in range(1, j+1)]
                curr_x = x_close - (sum([alpha * i for i in curr_coff]))
                y_coor_list.append(int(curr_x) - 1)
            print("y_coor_list", y_coor_list)
        return int(curr_x), (depth_index, y_coor_list)
    else:
        depth_index = np.argmin(np.abs(np.array(depth_bin) - depth_value)) - 1
        coeff = [1.0/i for i in range(1, num_of_bins+1)]
        alpha = (x_far - x_close) / sum(coeff)
        curr_coff = [1.0/i for i in range(1, depth_index+1)]
        curr_x = x_close + sum([alpha * i for i in curr_coff])
        y_coor_list = []
        if(debug):
            for j in range(num_of_bins):
                curr_coff = [1.0/i for i in range(1, j+1)]
                curr_x = x_close + (sum([alpha * i for i in curr_coff]))
                y_coor_list.append(int(curr_x) - 1)
            print("y_coor_list", y_coor_list)
        return int(curr_x), (depth_index, y_coor_list)


# plot the depth bins with each bin in diff color
def plot_depth_bins(depth, sam_mask, mask_image, input_img_name, save_dir, is_crop=False):
    import matplotlib.pyplot as plt
    input_img_name = input_img_name.split("/")[-1].split(".")[0]
    depth = np.array(depth)
    sam_mask = np.array(sam_mask)
    mask_region = np.array(mask_image)
    bins = get_bins_for_depth(sam_mask, depth, mask_region)

    # plot the depth bins with each bin in diff color in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # give 30 colors
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray']
    ax.view_init(elev=0, azim=-90)

    for i, bin in enumerate(bins):
        if(len(bin) == 0):
            continue
        ax.scatter(bin["pc"][1], bin["pc"][0], -1 * depth[bin["pc"]], c=bin["color"])            
        # ax.scatter(bin[1], bin[0], -1 * depth[bin], c=colors[np.random.randint(0, len(colors))])
    # plt.xlim(0, 1024)
    # plt.ylim(0, 1024)
    plt.show()
    if(is_crop):
        plt.savefig("{}/{}_depth_bins_crop.png".format(save_dir,input_img_name))
    else:
        plt.savefig("{}/{}_depth_bins.png".format(save_dir,input_img_name))
    plt.close()
    # plot front view of depth bins
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # front view
    # ax.view_init(elev=0, azim=0)
    for i, bin in enumerate(bins):
        if(len(bin) == 0):
            continue
        ax.scatter(bin["pc"][1], 800 - bin["pc"][0], c=bin["color"])
    # plt.xlim(0, 1024)
    # plt.ylim(0, 1024)
    plt.show()
    if(is_crop):
        plt.savefig("{}/{}_depth_bins_front_crop.png".format(save_dir,input_img_name))
    else:
        plt.savefig("{}/{}_depth_bins_front.png".format(save_dir,input_img_name))



def draw_depth_lines(image, y_coor_list, save_dir, input_img_name, chosen_bin=None):
    image = np.array(image)
    for l, y_coor in enumerate(y_coor_list):
        if(chosen_bin is not None and l == chosen_bin):
            image[y_coor] = [0, 255, 0]
        else:
            image[y_coor] = [255, 0, 0]
    cv2.imwrite("{}/{}_depth_lines_debug.jpg".format(save_dir, input_img_name), image[:,:,::-1])


if(__name__ == "__main__"):
    from ldm.modules.mpi.null_text_inv import NullTextPipeline
    from ldm.modules.mpi.stable_null_inverter import StableNullInverter
    from diffusers.schedulers import DDIMScheduler

    scheduler = DDIMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # ntp = NullTextPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", scheduler = scheduler).to("cuda")
    ntp = NullTextPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler = scheduler, 
                                           torch_dtype=torch.float16, variant="fp16").to("cuda")
    

    image = Image.open("/mnt/data/rishubh/sachi/AnyDoorV2/examples/Gradio/new_sample/c9.jpg").convert("RGB").resize((512,512))
    prompt = "a sofa set in a living room"

    
    null_text_latents = get_null_text_latents(ntp, image, prompt)
    
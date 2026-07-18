import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_mpi_rgb_and_alpha(input_img, depth_img, depth_partition):

    mpi_rgb = []
    mpi_alpha = []
    for i in range(len(depth_partition)):
        mask = cv2.inRange(depth_img, depth_partition[i][0], depth_partition[i][1])
        # mask = cv2.bitwise_not(mask)
        mpi_rgb.append(cv2.bitwise_and(input_img, input_img, mask=mask))
        mask = mask / 255
        mpi_alpha.append(mask)
        
    return mpi_rgb, mpi_alpha

def render_scene_from_mpi(mpi_rgbs, mpi_alphas):
    n_planes = len(mpi_alphas)

    # Creating a list of accumulated product values - prod(1-alpha)_i=1->i=D
    cum_alpha_ar = [np.ones(mpi_alphas[0].shape)] # The deepest layer or mpi, having all the values to be 1. 
    for idx in range(0,n_planes):
        cum_alpha = cum_alpha_ar[-1]*(1-mpi_alphas[n_planes-1-idx]) 
        cum_alpha_ar.append(cum_alpha) 

    # To accumulate the pixel values in the combined image 
    combined_img = np.zeros(mpi_rgbs[0].shape)
    for idx in range(0, n_planes):
        rgb = mpi_rgbs[idx]
        alpha = mpi_alphas[idx] 

        # As we are going from front to the back 
        weight = alpha * cum_alpha_ar[n_planes-idx-1]
        weight_expand = np.dstack((weight, weight, weight))

        combined_img += weight_expand*rgb

    return combined_img.astype(np.uint8)

def render_scene_from_mpi_torch(mpi_rgbs, mpi_alphas):
    n_planes = len(mpi_alphas)

    # Creating a list of accumulated product values - prod(1-alpha)_i=1->i=D
    _dev = mpi_alphas[0].device
    cum_alpha_ar = [torch.ones(mpi_alphas[0].shape).to(_dev)]
    for idx in range(0,n_planes):
        cum_alpha = cum_alpha_ar[-1]*(1-mpi_alphas[n_planes-1-idx])
        cum_alpha_ar.append(cum_alpha)

    # To accumulate the pixel values in the combined image
    combined_img = torch.zeros(mpi_rgbs[0].shape).to(_dev)
    for idx in range(0, n_planes):
        rgb = mpi_rgbs[idx]
        alpha = mpi_alphas[idx] 

        # As we are going from front to the back 
        weight = alpha * cum_alpha_ar[n_planes-idx-1]
        # weight_expand = torch.cat((weight, weight, weight), dim=2)

        combined_img += weight*rgb

    return combined_img


def render_scene_from_two_mpi(background_rgb, foreground_rgb, background_alpha, foreground_alpha, depth_partition, partition_bin_idx):
    rgb_final = []
    alpha_final = []
    for i in range(len(depth_partition)):
        if(i <= partition_bin_idx):
            rgb_final.append(background_rgb[i])
            alpha_final.append(background_alpha[i])
        else:
            rgb_final.append(foreground_rgb[i])
            alpha_final.append(foreground_alpha[i])
    
    return render_scene_from_mpi(rgb_final, alpha_final)


if __name__ == "__main__":
    # test the function
    input_image = cv2.imread("./img_dir/sample/sofa11.jpg")
    input_image = cv2.resize(input_image, (1024, 1024))
    depth_img = cv2.imread("./sofa11_depth.jpg", cv2.IMREAD_GRAYSCALE)
    print("max and min depth values: ", depth_img.max(), depth_img.min())
    depth_partition = [(0, 20), (21, 50), (51, 80), (81, 120), (121,270)]
    mpi_rgb, mpi_alpha = get_mpi_rgb_and_alpha(input_image, depth_img, depth_partition)

    for i in range(len(mpi_rgb)):
        plt.imshow(np.concatenate([mpi_rgb[i][:,:,::-1], cv2.cvtColor((mpi_alpha[i]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)], axis=1))
        plt.savefig("./mpi_{}.jpg".format(i))

    combined_img = render_scene_from_mpi(mpi_rgb, mpi_alpha)
    print(combined_img.max(), combined_img.min())
    plt.imshow(np.concatenate([input_image, combined_img], axis=1))
    plt.savefig("./combined_img.jpg")
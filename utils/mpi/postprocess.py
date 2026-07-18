import os
import numpy as np
import cv2
from PIL import Image

def _get_segment_object():
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.grounded_sam.grounded_sam_demo1 import main as segment_object
    return segment_object


def sam_postprocess(src_img, tar_img, text_prompt, mpi_foreground):
    segment_object = _get_segment_object()
    sam_json = segment_object(Image.fromarray(cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)), None, text_prompt=text_prompt, output_dir="./sam_output_dir")
    sam_mask = np.array(sam_json[1]["mask"][0])[:,:,None]
    sam_mask = np.concatenate([sam_mask]*3, axis=2)
    mpi_foreground = mpi_foreground // 255
    mpi_foreground = cv2.resize(mpi_foreground, (sam_mask.shape[1], sam_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    sam_mpi_mask = np.clip(sam_mask + mpi_foreground, 0, 1)
    final_image = src_img * (1 - sam_mpi_mask) + tar_img * sam_mpi_mask
    return final_image


def sam_postprocess2(src_img, tar_img, tar_mpi_img, text_prompt, mpi_foreground):
    segment_object = _get_segment_object()
    sam_json = segment_object(Image.fromarray(cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)), None, text_prompt=text_prompt, output_dir="./sam_output_dir")
    sam_mask = np.array(sam_json[1]["mask"][0])[:,:,None]
    sam_mask = np.concatenate([sam_mask]*3, axis=2)
    mpi_foreground = mpi_foreground // 255
    mpi_foreground = cv2.resize(mpi_foreground, (sam_mask.shape[1], sam_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    sam_mpi_mask = np.clip(sam_mask + mpi_foreground, 0, 1)
    final_image = src_img * (1 - sam_mpi_mask) + tar_mpi_img * sam_mpi_mask
    return final_image

def get_sam_mask(tar_img, text_prompt, object_bbox=None):
    segment_object = _get_segment_object()
    sam_json = segment_object(Image.fromarray(cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)), None, text_prompt=text_prompt, obj_bbox=object_bbox,
                               output_dir="./sam_output_dir")
    if(len(sam_json) == 0 or len(sam_json) == 1):
        return None
    sam_mask = np.array(sam_json[1]["mask"][0])[:,:,None]
    sam_mask = np.concatenate([sam_mask]*3, axis=2)
    return sam_mask


if(__name__ == "__main__"):
    src_img = cv2.imread('test.jpg')
    tar_img = cv2.imread('test_ori_gen.png')
    tar_mpi_gen = cv2.imread('test_mpi_gen.png')
    mpi_fore = cv2.imread('test_mpi_mask.png')
    final_postprocessed_image = sam_postprocess2(src_img, tar_img, tar_mpi_gen, "floor lamp", mpi_fore)
    cv2.imwrite('test_postprocessed_image2.png', final_postprocessed_image)
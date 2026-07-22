import pathlib
import threading
from typing import Callable, List, Tuple, TypeVar

import torch
import scipy.ndimage
from omegaconf import OmegaConf, DictConfig

from src.featglac.stable_null_inverter import StableNullInverter
from src.featglac.guided_stable_diffuser import GuidedStableDiffuser
# from src.diffhandles.depth_transform import transform_depth, normalize_depth
from src.featglac.utils import solve_laplacian_depth
from utils.mpi.null_text_config import null_text_ddim_steps, null_text_inner_steps

import numpy as np

T = TypeVar("T")


def _run_with_autograd(fn: Callable[[], T]) -> T:
    """
    Run code that needs autograd outside Gradio's thread-local inference_mode.

    Gradio 3.x wraps UI callbacks in ``torch.inference_mode()``, which breaks
    null-text optimization (``loss.backward()``). Autograd is thread-local, so
    a worker thread is enough to escape that context.
    """
    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        torch.set_grad_enabled(True)
        try:
            result["value"] = fn()
        except BaseException as exc:
            error["exc"] = exc

    worker = threading.Thread(target=_target, name="null-text-inversion")
    worker.start()
    worker.join()
    if "exc" in error:
        raise error["exc"]
    return result["value"]

class FeatureGuidance:

    def __init__(self, conf: DictConfig=None):

        if conf is None:
            conf = OmegaConf.load(f'{pathlib.Path(__file__).parent.resolve()}/config/default.yaml')
        
        self.conf = conf

        self.diffuser = GuidedStableDiffuser(conf=self.conf.guided_diffuser)
        self.inverter = StableNullInverter(self.diffuser, num_ddim_steps=null_text_ddim_steps())

        self.device = torch.device('cuda')

    def to(self, device: torch.device = None):

        self.diffuser.to(device=device)
        self.inverter.to(device=device)

        self.device = device

        return self

    def invert_input_image(self, img: torch.Tensor, depth: torch.Tensor, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Invert an input image to allow applying diffusion handles to that input image.
        Use the outputs of this function as inputs to generate_input_image to work with a reconstructed version of the given input image.

        Args:
            img: The input image.
            depth: Depth of the input image.

        Returns:
            null_text_emb: The null text of the inverted input image.
            init_noise: The initial noise of the inverted input image.
        """

        # disparity = normalize_depth(1.0/(depth + 1e-6))
        disparity = (1.0/(depth + 1e-6))

        def _invert():
            return self.inverter.invert(
                target_img=img,
                depth=disparity,
                prompt=prompt,
                num_inner_steps=null_text_inner_steps(),
                verbose=True,
            )

        _, init_noise, null_text_emb = _run_with_autograd(_invert)

        return null_text_emb, init_noise

    def generate_input_image(
            self, depth: torch.Tensor, prompt: str, null_text_emb: torch.Tensor = None, init_noise: torch.Tensor = None
            ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Generates the input image from a prompt and a depth map,
        and stores intermdiate activations that descibe the identity of objects in the image.
        Optionally accepts a null text embedding and a noise tensor that can be used
        to reconstruct an inverted input image.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            null_text_emb: A null text embedding (may come from image inversion).
            init_noise: A starting noise (may come from image inversion).

        Returns:
            inverted_null_text: A null text embedding that can be used to reconstruct the input image
            init_noise: A starting noise that can be used to reconstruct the input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            latent_image: latent encoding of the input image
        """

        # disparity = normalize_depth(1.0/depth + 1e-6)
        disparity = (1.0/depth + 1e-6)

        # perform first diffusion inference pass to get intermediate features
        with torch.no_grad():
            activations, latent_image, null_text_emb, init_noise = self.diffuser.initial_inference(
                init_latents=init_noise, depth=disparity, uncond_embeddings=null_text_emb,
                prompt=prompt)

        return null_text_emb, init_noise, activations, latent_image
    
    def set_foreground(self, depth: torch.Tensor, fg_mask: torch.Tensor, bg_depth: torch.Tensor) -> torch.Tensor:
        """
        Select the foreground object in the input image. 

        Args:
            depth: Depth of the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).

        Returns:
            bg_depth: An updated background depth that has been adjusted to better match the depth of the input image.
        """

        # update the background depth by copying the input depth, but infilling the hole in the input depth
        # (where the foreground object used to be) with the background depth
        bg_depth = solve_laplacian_depth(
            depth[0, 0].cpu().numpy(),
            bg_depth[0, 0].cpu().numpy(),
            scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
        bg_depth = torch.from_numpy(bg_depth).to(device=self.device)[None, None]

        return bg_depth

    def transform_foreground(
            self, depth: torch.Tensor, prompt: str,
            fg_mask:torch.Tensor, bg_depth: torch.Tensor,
            null_text_emb: torch.Tensor, init_noise: torch.Tensor, 
            activations: List[torch.Tensor],
            rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
            fg_weight: float = None, bg_weight: float = None, use_input_depth_normalization=False):
        """
        Transform the foreground object. The following steps are performed:
        1) The depth of the foreground object is 3D-transformed, giving us a an updated depth map and corresondences between old and new 2D image coordinates.
        2) The edited image is generated guided by intermediate features that are warped with the correspondences from the 3D transformation.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            inverted_null_text: The null text of the inverted input image.
            inverted_noise: The noise of the inverted input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.
            use_input_depth_normalization: Use the same normalization factor and bias as the input depth for the edited depth, to make the edited depth as similar to the input depth as possible.
        
        Returns:
            output_img: The edited image.
        """
        
        # 3d-transform depth
        with torch.no_grad():
            edited_disparity, correspondences = transform_depth(
                depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
                intrinsics=self.diffuser.get_depth_intrinsics(device=depth.device),
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=use_input_depth_normalization,
                depth_transform_mode=self.conf.depth_transform_mode)

        with torch.no_grad():
            results = self.diffuser.guided_inference(
                latents=init_noise, depth=edited_disparity, uncond_embeddings=null_text_emb,
                prompt=prompt,
                activations_orig=activations,
                correspondences=correspondences,
                fg_weight=fg_weight, bg_weight=bg_weight,
                save_denoising_steps=self.conf.guided_diffuser.save_denoising_steps)

        if self.conf.guided_diffuser.save_denoising_steps:
            edited_img, denoising_steps = results
            return edited_img, edited_disparity, denoising_steps
        else:
            edited_img = results
            return edited_img, edited_disparity


    def mpi_blending(
            self, depth: torch.Tensor, prompt: str,
            mpi_masks:torch.Tensor,
            null_text_emb: torch.Tensor, init_noise: torch.Tensor, 
            activations: List[torch.Tensor],
            fg_weight: float = None, bg_weight: float = None, use_input_depth_normalization=False):
        """
        Transform the foreground object. The following steps are performed:
        1) The depth of the foreground object is 3D-transformed, giving us a an updated depth map and corresondences between old and new 2D image coordinates.
        2) The edited image is generated guided by intermediate features that are warped with the correspondences from the 3D transformation.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            inverted_null_text: The null text of the inverted input image.
            inverted_noise: The noise of the inverted input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.
            use_input_depth_normalization: Use the same normalization factor and bias as the input depth for the edited depth, to make the edited depth as similar to the input depth as possible.
        
        Returns:
            output_img: The edited image.
        """

        if(len(mpi_masks) == 2):
            bg_mask, fg_mask = mpi_masks[0], mpi_masks[1]
            correspondences = np.where(fg_mask == 1)
            corr_y, corr_x = correspondences[0], correspondences[1]
            corr_x, corr_y = torch.tensor(corr_x), torch.tensor(corr_y)
            correspondences = torch.stack((corr_x, corr_y, corr_x, corr_y), dim=-1)
        else:
            bg_mask, mid_mask, fg_mask = mpi_masks[0], mpi_masks[1], mpi_masks[2]
            bg_correspondences = np.where((mid_mask + fg_mask) == 1)
            bg_corr_y, bg_corr_x = bg_correspondences[0], bg_correspondences[1]
            bg_corr_x, bg_corr_y = torch.tensor(bg_corr_x), torch.tensor(bg_corr_y)
            bg_correspondences = torch.stack((bg_corr_x, bg_corr_y, bg_corr_x, bg_corr_y), dim=-1)

            fg_correspondences = np.where(fg_mask == 1)
            fg_corr_y, fg_corr_x = fg_correspondences[0], fg_correspondences[1]
            fg_corr_x, fg_corr_y = torch.tensor(fg_corr_x), torch.tensor(fg_corr_y)
            fg_correspondences = torch.stack((fg_corr_x, fg_corr_y, fg_corr_x, fg_corr_y), dim=-1)

            mid_correspondences = np.where(mid_mask == 1)
            mid_corr_y, mid_corr_x = mid_correspondences[0], mid_correspondences[1]
            mid_corr_x, mid_corr_y = torch.tensor(mid_corr_x), torch.tensor(mid_corr_y)
            mid_correspondences = torch.stack((mid_corr_x, mid_corr_y, mid_corr_x, mid_corr_y), dim=-1)

            correspondences = [bg_correspondences, mid_correspondences, fg_correspondences]
        
        # perform second diffusion inference pass guided by the 3d-transformed features
        with torch.no_grad():
            results = self.diffuser.guided_mpi_inference(
                latents=init_noise, depth=depth, uncond_embeddings=null_text_emb,
                prompt=prompt,
                activations_orig=activations,
                correspondences=correspondences,
                fg_weight=fg_weight, bg_weight=bg_weight,
                save_denoising_steps=self.conf.guided_diffuser.save_denoising_steps)

        if self.conf.guided_diffuser.save_denoising_steps:
            edited_img, denoising_steps = results
            return edited_img, depth, denoising_steps
        else:
            edited_img = results
            return edited_img, depth
        
    
    def mpi_scene_comp(
            self, depth: torch.Tensor, prompt: str,
            mpi_masks:torch.Tensor,
            null_text_emb: torch.Tensor, init_noise: torch.Tensor, 
            activations: List[torch.Tensor],
            fg_weight: float = None, bg_weight: float = None, use_input_depth_normalization=False):
        """
        Transform the foreground object. The following steps are performed:
        1) The depth of the foreground object is 3D-transformed, giving us a an updated depth map and corresondences between old and new 2D image coordinates.
        2) The edited image is generated guided by intermediate features that are warped with the correspondences from the 3D transformation.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            inverted_null_text: The null text of the inverted input image.
            inverted_noise: The noise of the inverted input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.
            use_input_depth_normalization: Use the same normalization factor and bias as the input depth for the edited depth, to make the edited depth as similar to the input depth as possible.
        
        Returns:
            output_img: The edited image.
        """
        
        # 3d-transform depth
        # with torch.no_grad():
        #     edited_disparity, correspondences = transform_depth(
        #         depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
        #         intrinsics=self.diffuser.get_depth_intrinsics(device=depth.device),
        #         rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
        #         use_input_depth_normalization=use_input_depth_normalization,
        #         depth_transform_mode=self.conf.depth_transform_mode)
        if(len(mpi_masks) == 2):
            bg_mask, fg_mask = mpi_masks[0], mpi_masks[1]
            correspondences = np.where(fg_mask == 1)
            corr_y, corr_x = correspondences[0], correspondences[1]
            corr_x, corr_y = torch.tensor(corr_x), torch.tensor(corr_y)
            correspondences = torch.stack((corr_x, corr_y, corr_x, corr_y), dim=-1)
        else:

            bg_mask, mid_mask, fg_mask = mpi_masks[0], mpi_masks[1], mpi_masks[2]
            bg_correspondences = np.where((mid_mask + fg_mask) == 1)
            bg_corr_y, bg_corr_x = bg_correspondences[0], bg_correspondences[1]
            bg_corr_x, bg_corr_y = torch.tensor(bg_corr_x), torch.tensor(bg_corr_y)
            bg_correspondences = torch.stack((bg_corr_x, bg_corr_y, bg_corr_x, bg_corr_y), dim=-1)

            fg_correspondences = np.where(fg_mask == 1)
            fg_corr_y, fg_corr_x = fg_correspondences[0], fg_correspondences[1]
            fg_corr_x, fg_corr_y = torch.tensor(fg_corr_x), torch.tensor(fg_corr_y)
            fg_correspondences = torch.stack((fg_corr_x, fg_corr_y, fg_corr_x, fg_corr_y), dim=-1)

            mid_correspondences = np.where(mid_mask == 1)
            mid_corr_y, mid_corr_x = mid_correspondences[0], mid_correspondences[1]
            mid_corr_x, mid_corr_y = torch.tensor(mid_corr_x), torch.tensor(mid_corr_y)
            mid_correspondences = torch.stack((mid_corr_x, mid_corr_y, mid_corr_x, mid_corr_y), dim=-1)

            correspondences = [bg_correspondences, mid_correspondences, fg_correspondences]
        

        # perform second diffusion inference pass guided by the 3d-transformed features
        with torch.no_grad():
            results = self.diffuser.guided_mpi_scene_comp(
                latents=init_noise, depth=depth, uncond_embeddings=null_text_emb,
                prompt=prompt,
                activations_orig=activations,
                correspondences=correspondences,
                fg_weight=fg_weight, bg_weight=bg_weight,
                save_denoising_steps=self.conf.guided_diffuser.save_denoising_steps)

        if self.conf.guided_diffuser.save_denoising_steps:
            edited_img, denoising_steps = results
            return edited_img, depth, denoising_steps
        else:
            edited_img = results
            return edited_img, depth
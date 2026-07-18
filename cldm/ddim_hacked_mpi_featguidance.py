"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

from utils.mpi.mpi import render_scene_from_mpi_torch
import math
import scipy.ndimage
from src.featglac.losses import compute_foreground_loss

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.is_floating_point() and self.model.device.type == "mps":
                attr = attr.to(torch.float32)
            attr = attr.to(self.model.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               mpi_data=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    mpi_data=mpi_data,
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, mpi_data=None):
        device = self.model.betas.device
        b = shape[0]
        #x_T 1,4,64,64
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        ddim_latents = mpi_data['ddim_latents']
        mpi_foreground_alpha = mpi_data['mpi_masks'][1]
        mpi_masks = mpi_data['mpi_masks']
        # mpi_timesteps = mpi_data["mpi_timesteps"]
        do_mpi = mpi_data["do_mpi"]
        # amodal_mask = mpi_data.get("amodal_mask",None)
        bg_change_latent = mpi_data.get("bg_change_latent", None)
        activations_fg = mpi_data.get("activation_fore")
        
        # get corrrespondences
        bg_mask, fg_mask = mpi_data.get("mpi_orig_mask")
        correspondences = np.where(fg_mask == 1)
        corr_y, corr_x = correspondences[0], correspondences[1]
        corr_x, corr_y = torch.tensor(corr_x), torch.tensor(corr_y)
        correspondences = torch.stack((corr_x, corr_y, corr_x, corr_y), dim=-1)

        processed_correspondences = self.process_correspondences(correspondences, img_res=512, bg_erosion=0)
        bg_correspondences = processed_correspondences
        fg_correspondences = processed_correspondences

        guidance_max_step = 51
        fg_weight = 1.5 * 30     # 30
        bg_weight = 1.25 * 30    # 30
        denoising_weight_schedule = []

        fg_weight_falloff = np.linspace(fg_weight, fg_weight, guidance_max_step)
        bg_weight_falloff = np.linspace(bg_weight, bg_weight, guidance_max_step)

        for t_idx in range(guidance_max_step):
            fg_weights = [0.0, 0.2, 8.5]
            bg_weights = [0.0, 0.2, 1.5]
            
            denoising_weight_schedule.append((
                t_idx,
                (np.array(fg_weights)*fg_weight_falloff[t_idx]).tolist(),
                (np.array(bg_weights)*bg_weight_falloff[t_idx]).tolist()))
        denoising_weight_schedule.append((guidance_max_step, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
 
        optimization_weight_schedule = [
                (0, [3.5, 3.5, 3.5], [2.5, 2.5, 2.5]),
                (1, [1.25, 1.25, 1.25], [2.5, 2.5, 2.5]),
                (2, [2.5, 2.5, 2.5], [2.5, 2.5, 2.5]),
                (3, [2.5, 2.5, 2.5], [2.5, 2.5, 2.5]),
                (4, [1.5, 1.5, 1.5], [1.25, 1.25, 1.25]),
            ]
        self.guidance_weight_schedule = StepGuidanceWeightSchedule(
            denoising_steps=denoising_weight_schedule,
            optimization_steps=optimization_weight_schedule)


        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            # if(i == 30):
            if(i == 30):
                img = img * mpi_masks[0] + ddim_latents[-i-1].to(torch.float16) * mpi_masks[1]  

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,
                                      t_idx =i, correspondance=fg_correspondences, orig_activations=activations_fg)
            img, pred_x0 = outs
            # print("latents mean and std :", img.mean(), img.std())                    
            
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    # @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, t_idx = 0, correspondance=None, orig_activations=None):
        b, *_, device = *x.shape, x.device

        # latent update as per diffusion handles
        torch.set_grad_enabled(True)
        for param in self.model.model.diffusion_model.parameters():
            param.requires_grad = True
        for param in self.model.model.diffusion_model.parameters():
            param.grad = None


        x.requires_grad = True
        # x = x.to(torch.float16)
        with torch.enable_grad():
            # if(t_idx > 30 and t_idx < 48):
            if(t_idx > 30 and t_idx < 48):
            # if(t_idx < 39):
                for num_iter in range(3): # 3
                    fgw, bgw = self.guidance_weight_schedule(t_idx, num_iter)
                    x = torch.nn.parameter.Parameter(x.detach(), requires_grad=True)
                    model_output, activations = self.model.apply_model(x, t, c, return_activations=True)

                    activations_size = (orig_activations[2][t_idx].shape[-2], orig_activations[2][t_idx].shape[-1])

                    loss = 0.0
                    for act_idx in range(len(orig_activations)):
                        if fgw != 0.0:
                            loss += fgw[act_idx] * compute_foreground_loss(
                                activations=activations[act_idx][0], activations_orig=orig_activations[act_idx][t_idx],
                                processed_correspondences=correspondance,
                                patch_size=1, activations_size=activations_size)

                    loss.backward()
                    grad_cond = x.grad
                    x = x - 0.2 * grad_cond # 0.2
        torch.set_grad_enabled(False)
        ############################

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    # @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    # @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    # @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    def process_correspondences(self, correspondences, img_res, bg_erosion=0):

        original_x, original_y, transformed_x, transformed_y = torch.split(correspondences, 1, dim=-1)

        # Since np.split creates arrays of shape (N, 1), we'll squeeze them to get back to shape (N,)
        original_x = original_x.squeeze()
        original_y = original_y.squeeze()
        transformed_x = transformed_x.squeeze()
        transformed_y = transformed_y.squeeze()
        
        # bg_mask_orig = np.zeros((img_res, img_res))
        
        # bg_mask_trans = np.zeros((img_res, img_res))
        
        visible_orig_x = []
        visible_orig_y = []
        visible_trans_x = []
        visible_trans_y = []    
        
        for x, y, tx, ty in zip(original_x, original_y, transformed_x, transformed_y):
            if((tx >= 0 and tx < img_res) and (ty >= 0 and ty < img_res)):
                visible_orig_x.append(x)
                visible_orig_y.append(y)
                visible_trans_x.append(tx)
                visible_trans_y.append(ty)
        
        # for x, y in zip(visible_orig_x, visible_orig_y):
        #     bg_mask_orig[y,x] = 1

        # for x, y in zip(visible_trans_x, visible_trans_y):
        #     bg_mask_trans[y,x] = 1        
        
        original_x, original_y, transformed_x, transformed_y = (
            np.array(visible_orig_x, dtype=np.int64), np.array(visible_orig_y, dtype=np.int64),
            np.array(visible_trans_x, dtype=np.int64), np.array(visible_trans_y, dtype=np.int64))

        original_x, original_y = original_x // (img_res // 64), original_y // (img_res // 64)
        transformed_x, transformed_y = transformed_x // (img_res // 64), transformed_y // (img_res // 64)

        bg_mask_orig = np.ones(shape=[64, 64], dtype=np.bool_)
        if len(original_x) > 0:
            bg_mask_orig[original_y, original_x] = False

        bg_mask_trans = np.ones(shape=[64, 64], dtype=np.bool_)
        if len(transformed_x) > 0:
            bg_mask_trans[transformed_y, transformed_x] = False

        if bg_erosion > 0:
            bg_mask_orig = scipy.ndimage.binary_erosion(bg_mask_orig, iterations=bg_erosion)
            bg_mask_trans = scipy.ndimage.binary_erosion(bg_mask_trans, iterations=bg_erosion)

        bg_y, bg_x = np.nonzero(bg_mask_orig & bg_mask_trans)
        bg_y_orig, bg_x_orig = np.nonzero(bg_mask_orig)
        bg_y_trans, bg_x_trans = np.nonzero(bg_mask_trans)

        
        
        # # Create sets for original and transformed pixels
        # original_pixels = set(zip(original_x, original_y))
        # transformed_pixels = set(zip(transformed_x, transformed_y))

        # # Create a set of all pixels in a 64x64 image
        # all_pixels = {(x, y) for x in range(64) for y in range(64)}

        # # Find pixels not in either of the original or transformed sets
        # bg_pixels = all_pixels - (original_pixels | transformed_pixels)

        # # Extract background_x and background_y
        # bg_x = np.array([x for x, y in bg_pixels])
        # bg_y = np.array([y for x, y in bg_pixels])

        # bg_pixels_orig = all_pixels - (original_pixels)

        # bg_x_orig = np.array([x for x, y in bg_pixels_orig])
        # bg_y_orig = np.array([y for x, y in bg_pixels_orig])

        # bg_pixels_trans = all_pixels - (transformed_pixels)

        # bg_x_trans = np.array([x for x, y in bg_pixels_trans])
        # bg_y_trans = np.array([y for x, y in bg_pixels_trans])

        processed_correspondences = {
            'original_x': original_x,
            'original_y': original_y,
            'transformed_x': transformed_x,
            'transformed_y': transformed_y,
            'background_x': bg_x,
            'background_y': bg_y,
            'background_x_orig': bg_x_orig,
            'background_y_orig': bg_y_orig,
            'background_x_trans': bg_x_trans,
            'background_y_trans': bg_y_trans,
        }

        return processed_correspondences

class GuidanceWeightSchedule:

    def __init__(self):
        pass

    def __call__(self, denoising_step: int, optimization_step: int):
        fg_weights = [1.0]*3
        bg_weights = [1.0]*3
        return fg_weights, bg_weights

class StepGuidanceWeightSchedule(GuidanceWeightSchedule):

    def __init__(
            self,
            denoising_steps,
            optimization_steps):

        super().__init__()
        
        if not all(len(fg_weights) == len(bg_weights) for _, fg_weights, bg_weights in denoising_steps):
            raise ValueError("Number of foreground and background weights do not match.")
        if not all(len(fg_weights) == len(bg_weights) for _, fg_weights, bg_weights in optimization_steps):
            raise ValueError("Number of foreground and background weights do not match.")
        if len(denoising_steps[0][1]) != len(optimization_steps[0][1]):
            raise ValueError("Number of denoising and optimization weights do not match.")

        self.denoising_steps = sorted(denoising_steps, key=lambda step: step[0])
        self.optimization_steps = sorted(optimization_steps, key=lambda step: step[0])

    def __call__(self, denoising_step: int, optimization_step: int):
        
        denoising_fg_weights = None
        denoising_bg_weights = None
        optimization_fg_weights = None
        optimization_bg_weights = None
        
        for step, fg_weights, bg_weights in reversed(self.denoising_steps):
            if denoising_step >= step:
                denoising_fg_weights = fg_weights
                denoising_bg_weights = bg_weights
                break
        for step, fg_weights, bg_weights in reversed(self.optimization_steps):
            if optimization_step >= step:
                optimization_fg_weights = fg_weights
                optimization_bg_weights = bg_weights
                break

        if any(weight is None for weight in [denoising_fg_weights, denoising_bg_weights, optimization_fg_weights, optimization_bg_weights]):
            raise ValueError(f"Could not find weights for denoising step {denoising_step} and optimization step {optimization_step}.")

        fg_weights = [dw * ow for dw, ow in zip(denoising_fg_weights, optimization_fg_weights)]
        bg_weights = [dw * ow for dw, ow in zip(denoising_bg_weights, optimization_bg_weights)]
        
        return fg_weights, bg_weights
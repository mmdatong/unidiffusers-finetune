import torch
from diffusers import UniDiffuserPipeline
from diffusers.utils import load_image, PIL_INTERPOLATION, randn_tensor
import PIL
import numpy as np
import warnings
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from typing import Dict, Union






# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    warnings.warn(
        "The preprocess method is deprecated and will be removed in a future version. Please"
        " use VaeImageProcessor.preprocess instead",
        FutureWarning,
    )
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]


    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = 512, 512

        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def _get_noise_pred(
        pipe,
        latents,
        prompt_embeds,
        t,
        data_type=1
        ):


    
    height = pipe.unet_resolution * pipe.vae_scale_factor
    width = pipe.unet_resolution * pipe.vae_scale_factor
    
    img_vae_latents, img_clip_latents = pipe._split(latents, height, width)

    img_vae_out, img_clip_out, text_out = pipe.unet(
            img_vae_latents, 
            img_clip_latents,
            prompt_embeds,
            timestep_img=t, 
            timestep_text=0,
            data_type=data_type
            )

    img_out = pipe._combine(img_vae_out, img_clip_out)
    text_T = randn_tensor(
            prompt_embeds.shape, 
            generator=None, 
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype
            )

    img_vae_out_uncond, img_clip_out_uncond, text_out_uncond = pipe.unet(
        img_vae_latents,
        img_clip_latents,
        text_T,
        timestep_img=t,
        timestep_text=pipe.scheduler.config.num_train_timesteps,
        data_type=data_type
        )
    img_out_uncond = pipe._combine(img_vae_out_uncond, img_clip_out_uncond)

    guidance_scale = 8.0

    return guidance_scale * img_out + (1.0 - guidance_scale) * img_out_uncond
    

def encode_prompts(pipe, prompts):
    """
    prompts is list of str
    """

    prompt_embeds = pipe._encode_prompt(
            prompt=prompts,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False, 
            negative_prompt=None)
    reduce_text_emb_dim = pipe.text_intermediate_dim < pipe.text_encoder_hidden_size 

    if reduce_text_emb_dim:
        prompt_embeds = pipe.text_decoder.encode(prompt_embeds)

    return prompt_embeds


def encode_image_vae_latents(pipe, images, dtype):
    image_vae_latents = pipe.encode_image_vae_latents(
            image=images,
            batch_size=len(images),
            num_prompts_per_image=1,
            dtype=dtype,
            device=pipe.device,
            do_classifier_free_guidance=False, 
            generator=None)
    return image_vae_latents

def encode_image_clip_latents(pipe, images, dtype):
    image_clip_latents = pipe.encode_image_clip_latents(
            image=images,
            batch_size=len(images),
            num_prompts_per_image=1,
            dtype=dtype,
            device=pipe.device,
            generator=None,
            )

    image_clip_latents = image_clip_latents.unsqueeze(1)

    return image_clip_latents







if __name__ == "__main__":

    model_id_or_path = "thu-ml/unidiffuser-v1"
    pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    device = "cuda"
    pipe.to(device)
    
    
    prompts = [
            "this is a dog",
           # "this is a cat"
            ]
    prompt_embeds = encode_prompts(pipe, prompts)
    
    
    
    
    image_path = "dog/alvan-nee-eoqnr8ikwFE-unsplash.jpeg"
    image = load_image(image_path).resize((512, 512))
    image_vae = preprocess(image)
    
    
    image_vaes = torch.cat([image_vae], dim=0).to(prompt_embeds.dtype)
    images = [image]
    
    
    image_vae_latents = encode_image_vae_latents(pipe, image_vaes, dtype=prompt_embeds.dtype)
    image_clip_latents = encode_image_clip_latents(pipe, images, dtype=prompt_embeds.dtype)
    latents = pipe._combine(image_vae_latents, image_clip_latents)
    
    noise_pred = _get_noise_pred(
            latents,
            prompt_embeds,
            100
            )
    
    noise_scheduler = pipe.scheduler
    
    import pdb; pdb.set_trace()
    
    

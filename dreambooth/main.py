import os
import math
from accelerate.utils import ProjectConfiguration, set_seed
import torch
from diffusers import UniDiffuserPipeline
import argparse
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger

from dream_booth_dataset import build_dataloader, preprocess
from diffusers.optimization import get_scheduler
from utils import encode_prompts, encode_image_vae_latents, encode_image_clip_latents, _get_noise_pred

from torch.nn import functional as F
from tqdm.auto import tqdm

import itertools



def create_optimizer(
        args,
        unet, 
        text_encoder
        ):
    optimizer_class = torch.optim.AdamW
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    return optimizer


def create_lr_scheduler(args, optimizer, accelerator):
    print("args.lr_warmup_steps", args.lr_warmup_steps)


    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    return lr_scheduler

def create_accelarator(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1)


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin,
    )

    return accelerator


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
            "--train_text_encoder", 
            action="store_true",
            help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
            )
    parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-6,
            help="Initial learning rate (after the potential warmup period) to use.",
        )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")


    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )



    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    

    args = parser.parse_args()
    return args



def save_models(unet, text_encoder, output_dir, model_id_or_path):
    pipeline_args = {}
    if text_encoder is not None:
        pipeline_args["text_encoder"] = text_encoder

    pipe = UniDiffuserPipeline.from_pretrained(
            model_id_or_path, 
            unet = unet,
            **pipeline_args,
            )

    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
            scheduler_args["variance_type"] = variance_type
    pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config, **scheduler_args)

    pipe.save_pretrained("/root/autodl-tmp/output_diffusers_nocliplatent_{}".format(epoch))



def main():
    args = parse_args()

    model_id_or_path = args.model_id_or_path

    pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)

    # device = "cuda"
    # pipe.to(device)


    instance_data_root = "dog"
    instance_prompt = "a photo of sks dog"

    dataloader = build_dataloader(
            instance_data_root,
            instance_prompt)


    noise_scheduler = pipe.scheduler
    unet = pipe.unet
    text_encoder = pipe.text_encoder






    vae = pipe.vae
    image_encoder = pipe.image_encoder
    text_decoder = pipe.text_decoder 

    clip_tokenizer = pipe.clip_tokenizer 
    image_processor = pipe.image_processor
    text_tokenizer = pipe.text_tokenizer 


    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    text_decoder.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)


    import xformers
    pipe.unet.enable_xformers_memory_efficient_attention()
    #pipe.unet.enable_gradient_checkpointing()






    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = args.max_train_steps // num_update_steps_per_epoch





    accelerator = create_accelarator(args)

    if args.train_text_encoder:

        pipe.unet, pipe.text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
                pipe.unet, pipe.text_encoder, optimizer, dataloader, lr_scheduler
                )
    else:
        optimizer = create_optimizer(args, unet, text_encoder=None)

    lr_scheduler = create_lr_scheduler(args, optimizer, accelerator)
    pipe.unet,  optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet, optimizer, dataloader, lr_scheduler
            )

    weight_dtype = torch.float32
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.unet.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_decoder.to(accelerator.device, dtype=weight_dtype)
    pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)


    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")



    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()


        for step, batch in enumerate(dataloader):

            with torch.no_grad():
                instance_prompt_embeds = encode_prompts(pipe, batch['instance_prompts'])
                instance_image_vaes = [preprocess(image) for image in batch['instance_images']]
                instance_image_vaes = torch.cat(instance_image_vaes, dim=0).to(instance_prompt_embeds.dtype)
                instance_images = batch['instance_images']

                instance_image_vae_latents = encode_image_vae_latents(
                        pipe, 
                        instance_image_vaes,
                        dtype=instance_prompt_embeds.dtype)

                instance_image_clip_latents = encode_image_clip_latents(
                        pipe, 
                        instance_images,
                        dtype=instance_prompt_embeds.dtype)


            latents = pipe._combine(instance_image_vae_latents, instance_image_clip_latents)
            model_input = latents


            if args.offset_noise:
                noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                        model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                        )
            else:
                noise = torch.randn_like(model_input)

            bsz = model_input.shape[0]

            timesteps = torch.randint(
                     0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                     )
            timesteps = timesteps.long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = _get_noise_pred(
                    pipe, 
                    noisy_model_input.to(pipe.device), 
                    instance_prompt_embeds.to(pipe.device), 
                    timesteps.to(pipe.device)
                    )



            vae_latents_len = target.shape[-1]

            loss = F.mse_loss(
                    model_pred.float()[..., :vae_latents_len], 
                    target.float()[..., :vae_latents_len], 
                    reduction="mean")

            # loss.backward()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            #  print(idx, loss)
            progress_bar.update(1)
            global_step += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
        if (epoch+1)%100 == 0:
            output_dir = os.path.join(args.output_dir, str(epoch))
            save_models(
                    unet=accelerator.unwrap_model(pipe.unet),
                    text_encoder=accelerator.unwrap_model(pipe.text_encoder),
                    output_dir=args.output_dir,
                    model_id_or_path=args.model_id_or_path
                    )

            # prompts = [

if __name__ == "__main__":
    main()



import torch
import sys
import argparse
import os
sys.path.append("./diffusers-main/src")
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        required=True,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        required=True,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    model_path = args.model_path
    output_path = args.output_path
    text_prompt = args.text_prompt
    device = args.device
    seed = args.seed
    num_images = args.num_images
    num_inference_steps = args.num_inference_steps

    pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch.float16
            )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    pipe.unet.load_attn_procs(model_path, weight_name="pytorch_custom_diffusion_weights.safetensors", _pipeline=pipe)
    pipe.load_textual_inversion(model_path, weight_name="<new1>.safetensors")
    pipe.load_textual_inversion(model_path, weight_name="<new2>.safetensors")

    generator = torch.Generator(device=device).manual_seed(seed)
    images = [
        pipe(text_prompt, num_inference_steps=num_inference_steps, generator=generator, eta=1.0).images[0]
        for _ in range(num_images)
    ]

    output_path = os.path.join(output_path, text_prompt)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(len(images)):
        images[i].save(os.path.join(output_path, f'{seed}-{i}.jpg'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

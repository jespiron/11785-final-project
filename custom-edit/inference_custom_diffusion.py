import torch
import sys
sys.path.append("./diffusers-main/src")
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler
)

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
model_path = './output'
text_prompt = "a photo of <new1> man and <new2> woman."
device = 'cuda'
seed = 47
num_images = 4

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
    pipe(text_prompt, num_inference_steps=25, generator=generator, eta=1.0).images[0]
    for _ in range(num_images)
]

for i in range(len(images)):
    images[i].save(f"./output/output_img_{i}.png")

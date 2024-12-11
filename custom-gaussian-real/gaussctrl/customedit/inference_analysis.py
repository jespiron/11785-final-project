import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import sys
from PIL import Image
import wandb
from io import BytesIO

sys.path.append("./diffusers-main/src")
import diffusers
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import (
    AnalysisAttnProcessor3,
)


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def load_model(pipe, path):
    st = torch.load(path)
    if 'modifier_token' in st:
        modifier_tokens = list(st['modifier_token'].keys())
        modifier_token_id = []
        for modifier_token in modifier_tokens:
            num_added_tokens = pipe.tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " 'modifier_token' that is not already in the tokenizer."
                )
            modifier_token_id.append(pipe.tokenizer.convert_tokens_to_ids(modifier_token))
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        for i, id_ in enumerate(modifier_token_id):
            token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]

    for name, params in pipe.unet.named_parameters():
        if name in st['unet']:
            params.data.copy_(st['unet'][f'{name}'])


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--path', type=str)
    parser.add_argument('--prompt', default='<new1> man and <new2> woman', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--output_dir', default="/data/MultiCustom/outputs/inference_analysis/")
    parser.add_argument('--num_iters', default=10, type=int)
    parser.add_argument('--seed', default=4444, type=int)
    parser.add_argument('--num_idx', default=0, type=int)
    parser.add_argument('--algorithm', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.is_wandb = True
    # args.is_wandb = None
    args.algorithm = "CD"
    # args.algorithm = "ours"

    args.batch_size = 8
    # args.seed = 6000
    # args.num_idx = 0

    args.attention_processor = AnalysisAttnProcessor3
    args.class_names = "man+woman"
    args.instant_names = "Harry_Potter+Hermione_Granger"
    # args.instant_names = "biden+harris"
    # args.class_names = "chair+backpack"
    # args.instant_names = "furniture_chair1+luggage_backpack1"
    c1, c2 = args.class_names.split("+")
    m1, m2 = args.instant_names.split("+")
    args.prompt = f"<new1> {c1} and <new2> {c2}"

    # if args.attention_processor == AnalysisAttnProcessor:
    #     proj_name = "Analysis - Q=all, K=V=partial"
    # elif args.attention_processor == AnalysisAttnProcessor2:
    #     proj_name = "Analysis - Q=K=all, V=partial"
    # elif args.attention_processor == AnalysisAttnProcessor3:
    #     proj_name = "Analysis on squared attn map - Q=all, K=V=partial"

    # use attn processor 3
    proj_name = "Analysis on squared attn map - Q=all, K=V=partial"

    if args.algorithm == "CD":
        args.path = f"./output/hp_hg/"
        if args.is_wandb:
            wandb.init(project=proj_name, name=f"CD-gpt_prompts-{args.instant_names}, seed-{args.seed}")
    elif args.algorithm == "ours":
        args.path = f"/data/MultiCustom/outputs/ours/{m1}_and_{m2}-6001.bin"
        if args.is_wandb:
            wandb.init(project=proj_name, name=f"ours-{args.instant_names}, seed-{args.seed}")

    # load model
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')

    pipe.unet.load_attn_procs(args.path, weight_name="pytorch_custom_diffusion_weights.safetensors", _pipeline=pipe)
    pipe.load_textual_inversion(args.path, weight_name="<new1>.safetensors")
    pipe.load_textual_inversion(args.path, weight_name="<new2>.safetensors")

    pipe = pipe.to(torch_dtype=torch.float16)

    # switch attention processor
    attention_procs = {}
    train_kv = True
    train_q_out = False

    st = pipe.unet.state_dict()
    for name, _ in pipe.unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipe.unet.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            weights = {
                "to_k_custom_diffusion.weight": st[name + ".to_k_custom_diffusion.weight"],
                "to_v_custom_diffusion.weight": st[name + ".to_v_custom_diffusion.weight"],
            }
            attention_procs[name] = args.attention_processor(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(pipe.unet.device)
            attention_procs[name].load_state_dict(weights)
        else:
            attention_procs[name] = args.attention_processor(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    pipe.unet.set_attn_processor(attention_procs)

    # get text embs for analysis
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # "<new1> c1 and <new2> c2"
    #     1    2  3     4    5
    prompt_list = [
        '<new1>', f'{c1}', '<new2>', f'{c2}',
        f'<new1> {c1}', f'<new1> {c2}', f'<new2> {c1}', f'<new2> {c2}',
        f'<new1> {c1} and <new2> {c2}',
                   ]
    key_idx_list = [
        [1], [2], [4], [5],
        [1, 2], [1, 5], [4, 2], [4, 5],
        [1, 2, 3, 4, 5],
    ]
    value_idx_list = [
        [1], [1], [1], [1],
        [1, 2], [1, 2], [1, 2], [1, 2],
        [1, 2, 3, 4, 5],
    ]

    text_embs_list = []
    for prompt in prompt_list:
        text_ids = tokenize_prompt(tokenizer, [prompt], tokenizer_max_length=77)['input_ids'][0]
        text_ids = text_ids.unsqueeze(0).to(text_encoder.device)
        text_embs = text_encoder(text_ids).last_hidden_state
        text_embs = torch.cat([text_embs] * 2)
        text_embs_list.append(text_embs)

    # if args.attention_processor == AnalysisAttnProcessor:
    #     cross_attention_kwargs = {'text_embs_list': text_embs_list,
    #                               'prompt_list': prompt_list,
    #                               'result_dict': {}}
    # elif args.attention_processor == AnalysisAttnProcessor2:
    #     cross_attention_kwargs = {'text_embs_list': text_embs_list,
    #                               'prompt_list': prompt_list,
    #                               'key_idx_list': key_idx_list,
    #                               'value_idx_list': value_idx_list,
    #                               'result_dict': {}}
    # elif args.attention_processor == AnalysisAttnProcessor3:
    #     cross_attention_kwargs = {'text_embs_list': text_embs_list,
    #                               'prompt_list': prompt_list,
    #                               'value_idx_list': value_idx_list,
    #                               'num_idx': args.num_idx,
    #                               'result_dict': {}}

    cross_attention_kwargs = {'text_embs_list': text_embs_list,
                              'prompt_list': prompt_list,
                              'value_idx_list': value_idx_list,
                              'num_idx': args.num_idx,
                              'result_dict': {},
                              'timestep': [],
                              }

    images = pipe([args.prompt] * args.batch_size, generator=generator,
                  num_inference_steps=50,
                  cross_attention_kwargs=cross_attention_kwargs).images

    result_dict = cross_attention_kwargs['result_dict']

    for timestep in result_dict.keys():
        states_list_16 = {}
        states_abs_np_list = []
        for prompt in prompt_list:
            state = (result_dict[timestep]['shape16'][prompt][0]).mean(dim=-1)
            state = state.reshape(16, 16)
            state = state ** 2
            states_list_16[prompt] = state.clone()

            state_abs = torch.abs(state)

            state_abs = state_abs.unsqueeze(0).unsqueeze(0)
            state_abs = torch.nn.functional.interpolate(
                state_abs.to(dtype=torch.float32),
                size=(128, 128),
                mode='bicubic',
            ).squeeze(0).squeeze(0)


            ## state_abs = (state_abs - torch.min(state_abs)) / (torch.max(state_abs) - torch.min(state_abs))

            state_abs = state_abs.cpu().numpy()
            states_abs_np_list.append(state_abs)

        m1_m1c1_difference_16 = states_list_16['<new1>'] - states_list_16[f'<new1> {c1}']
        m1_m1c2_difference_16 = states_list_16['<new1>'] - states_list_16[f'<new1> {c2}']
        m2_m2c1_difference_16 = states_list_16['<new2>'] - states_list_16[f'<new2> {c1}']
        m2_m2c2_difference_16 = states_list_16['<new2>'] - states_list_16[f'<new2> {c2}']

        original_m1c1_difference_16 = states_list_16[f'<new1> {c1} and <new2> {c2}'] - states_list_16[f'<new1> {c1}']
        original_m2c2_difference_16 = states_list_16[f'<new1> {c1} and <new2> {c2}'] - states_list_16[f'<new2> {c2}']

        prompt_difference_list = ['m1 - m1c1', 'm1 - m1c2', 'm2 - m2c1', 'm2 - m2c2', 'm1c1 and m2c2 - m1c1', 'm1c1 and m2c2 - m2c2']
        states_difference_list_16 = [m1_m1c1_difference_16, m1_m1c2_difference_16,
                                     m2_m2c1_difference_16, m2_m2c2_difference_16,
                                     original_m1c1_difference_16, original_m2c2_difference_16]

        #################################
        states_difference_abs_np_list = []
        for prompt, state in zip(prompt_difference_list, states_difference_list_16):
            state_abs = torch.abs(state)
            state_abs = state_abs.unsqueeze(0).unsqueeze(0)
            state_abs = torch.nn.functional.interpolate(
                state_abs.to(dtype=torch.float32),
                size=(128, 128),
                mode='bicubic',
            ).squeeze(0).squeeze(0)

            ### state_abs = (state_abs - torch.min(state_abs)) / (torch.max(state_abs) - torch.min(state_abs))
            state_abs = state_abs.cpu().numpy()
            states_difference_abs_np_list.append(state_abs)


        # def change_np_to_heatmap(np_image, min=0, max=255):
        #     heatmap = sns.heatmap(np_image, cmap='viridis', vmin=min, vmax=max, cbar=True)
        #     heatmap.set_xticks([])
        #     heatmap.set_yticks([])
        #     # Save the heatmap to a BytesIO buffer
        #     buffer = BytesIO()
        #     plt.savefig(buffer, format='png', bbox_inches='tight')
        #     buffer.seek(0)  # Rewind to the beginning of the buffer
        #     # Convert buffer to PIL Image
        #     pil_image = Image.open(buffer)
        #     plt.close()
        #     # buffer.close()
        #     return pil_image

        def change_np_to_heatmap(np_image, min=0, max=255):
            fig, ax = plt.subplots()
            heatmap = sns.heatmap(np_image, cmap='viridis', vmin=min, vmax=max, cbar=False)
            heatmap.set_xticks([])
            heatmap.set_yticks([])
            # Save the heatmap to a BytesIO buffer
            fig.canvas.draw()
            # Convert the canvas to a NumPy array
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(image_array)
            # Close the plot to free up memory
            plt.close(fig)

            return pil_image

        states_abs_heatmap_list = []
        states_difference_abs_heatmap_list = []
        for state in states_abs_np_list:
            heatmap = change_np_to_heatmap(state, state.min(), state.max())
            states_abs_heatmap_list.append(heatmap)
        for state in states_difference_abs_np_list:
            heatmap = change_np_to_heatmap(state, state.min(), state.max())
            states_difference_abs_heatmap_list.append(heatmap)

        wandb_prompt_list = prompt_list + prompt_difference_list
        wandb_heatmap_list = states_abs_heatmap_list + states_difference_abs_heatmap_list

        if args.is_wandb:
            wandb.log({
                f'difference_timestep_{timestep}': [
                    wandb.Image(image, caption=f"{prompt}") for i, (image, prompt) in
                    enumerate(zip(wandb_heatmap_list, wandb_prompt_list))
                ]
            }
            )

    # assume batch size 1
    image = images[args.num_idx]

    # log to wandb
    if args.is_wandb:
        wandb.log(
            {
                'images': [
                    wandb.Image(image, caption=f'image')
                ]
            }
        )

    '''
    all_images = []
    pipe = pipe.to(torch_dtype=torch.float16)
    images = pipe([args.prompt] * args.batch_size,
                  generator=generator).images
    all_images += images
    images = np.hstack([np.array(x) for x in images])
    images = Image.fromarray(images)
    name = '-'.join(args.prompt[:50].split())
    images.save(f'{args.output_dir}/{name}.png')

    os.makedirs(f'{args.output_dir}/samples', exist_ok=True)
    for i, im in enumerate(all_images):
        im.save(f'{args.output_dir}/samples/{i}.jpg')
    '''

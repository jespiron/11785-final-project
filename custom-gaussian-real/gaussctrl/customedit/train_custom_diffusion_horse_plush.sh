#!/bin/bash

# Set the script to fail if any command fails
set -e

# Train custom diffusion model
accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
  --output_dir=./output/stone_horse \
  --instance_prompt="a photo of a <new1> horse" \
  --class_prompt="a photo of a horse plush" \
  --instance_data_dir=./data/custom_gaussian/horse_plush \
  --class_data_dir=./data/prior_image/horse_plush \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=4 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr \
  --hflip \
  --modifier_token="<new1>" \
  --num_validation_images=8 \
  --validation_prompt="a photo of <new1> horse." \
  --report_to="wandb" \
  --enable_xformers_memory_efficient_attention
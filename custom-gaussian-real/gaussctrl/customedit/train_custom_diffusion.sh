#!/bin/bash

# Set the script to fail if any command fails
set -e

# Train custom diffusion model
accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
  --output_dir=./output/cup \
  --instance_prompt="a photo of a <new1> cup" \
  --class_prompt="a photo of a cup" \
  --instance_data_dir=./data/benchmark_dataset/things_cup3 \
  --class_data_dir=./data/prior_image/cup \
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
  --validation_prompt="a photo of <new1> cup, captured from the front." \
  --report_to="wandb" \
  --enable_xformers_memory_efficient_attention
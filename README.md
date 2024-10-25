# About

Final project for 11785.

**Group name:** Group 22

**Group Members**
* [Wonsik Shin (wonsiks)](https://github.com/ceteris11)
* [Jessica Ruan (jwruan)](https://github.com/jespiron)
* [Brandon Dong (bjdong)](https://github.com/sad-ish-cat)
* [Aradhya Talan (atalan)](https://github.com/aradhyatalan)

**Project Summary:**

Proposal [here](https://drive.google.com/file/d/1CPV3XIylqadEOKOQzVrcmU3B1-otRehx/view?usp=sharing)

# Getting Started

First, set up the virtual environment.

```
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
```

At a high level, our workflow is
1. Edit 2D images using 2D diffusion model
2. Construct 3D object from edited 2D images
3. Evaluate results

As such, this repository is organized into
1. Training the 2D diffusion model under `/latent-diffusion-models`
2. Viewing the 3D object that is constructed via `/splat` viewer
3. Comparing CLIP similarity scores of edited 2D images [TODO]

# Training the 2D Diffusion Model

First, navigate to the `/latent-diffusion-models` directory.

## Generating Images

To generate images using a specified diffusion model and prompt, run `stable_txt2img.py`:
```
PYTHONPATH=$(pwd) python3 ./scripts/stable_txt2img.py
    --ddim_eta 0.0
    --n_samples 1
    --n_iter 1
    --scale 10.0
    --ddim_steps 50
    --ckpt ./models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt
    --prompt "a photo of a dog" 
```
where `--ckpt` specifies the model and `--prompt` specifies the text prompt.

❗ The checkpoint `sd-v1-4-full-ema.ckpt` is too large to upload to Github. Please download it from [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main) and place it under the `models/ldm/stable-diffusion-v1` directory.

List of other pretrained diffusion models is listed in the [Stable Diffusion repo](https://github.com/CompVis/stable-diffusion?tab=readme-ov-file#stable-diffusion-v1).

## Training Personalized Model

The idea behind diffusion personalization is as follows: it introduces the notion of an identifier `[V]`.

Previously, we prompt diffusion models with text such as `photo of a class`, where class is the class name of the subject, such as "dog," "cat," "container," etc. However, we may want more fine-grained control over what kind of dog, cat, or container gets generated.

After introducing this unique identifier `[V]`, we can personalize our diffusion models to support prompts of the form `photo of a [V] class`, such that we can generate exclusively images of labrador dogs, tortoiseshell cats, or trashcan containers.

To train, 
1. Specify the identifier `[V]` by replacing the placeholder `sdjsksksf` in `latent-diffusion-models/ldm/data/personalized.py`. `[V]` does not need to be an English word; it just has to be a rare word in the tokenizer. 
2. Identify the `CLASS_NAME` that we would like to train our identifiers in.
2. Prepare the training and regularization imagesets. The training set is images that are specifically of `[V] CLASS_NAME` objects, while the regularization set is images that are generally of `CLASS_NAME` objects.
3. Give the model a `JOB_NAME`. Checkpoints will be saved under `./logs/${JOB_NAME}/checkpoints`, one at 500 steps and one when training finishes at 800 steps.
4. Train by running
```
PYTHONPATH=$(pwd) python3 main.py
                --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume ./models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt  
                -n JOB_NAME
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word CLASS_NAME
```

# Contributing

Please name branches `your-nickname/name-of-change`. Isn't strict as long as we know who's who

When your PR is approved, please select **[Squash and merge](https://www.lloydatkinson.net/posts/2022/should-you-squash-merge-or-merge-commit/)** from the dropdown. This leads to a much cleaner commit history!

After merging the PR, clean up your branch locally and remotely.
1. Locally: `git branch -D your-nickname/name-of-change`
2. Remotely: deleted automatically since "Automatically delete head branches" is enabled. If it doesn't work, can delete manually by clicking the trash icon next to your branch.

**Set up:**
1. Fork this repository
2. In your fork, add this repo as a remote `git remote add upstream https://github.com/jespiron/11785-final-project.git`
3. Pull changes with `git fetch upstream -p`. The `-p` prunes any branches that were deleted upstream
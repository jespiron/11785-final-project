from diffusers import StableDiffusionPipeline, DDIMScheduler

from editmodule.display_utils import run_and_display
from editmodule.prompt2prompt import make_controller
from editmodule.nullinversion import NullInversion
from editmodule.config import device, config

# Initialize config
# TODO: replace ldm_stable with our personalized diffusion model
config.ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
config.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
try:
    config.ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")

config.tokenizer = config.ldm_stable.tokenizer

#-----
image_path = "./input/gnochi_mirror1.jpeg"
prompt = "a cat sitting next to a mirror"
null_inversion = NullInversion(config.ldm_stable)
(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)

print("Modify or remove offsets according to your image!")

prompts = ["a cat sitting next to a mirror",
           "a silver cat sculpture sitting next to a mirror"
        ]

cross_replace_steps = {'default_': .8, }
self_replace_steps = .6
blend_word = ((('cat',), ("cat",))) # for local edit
eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)
#---

# Created EditSession to modularize the above code
# After we get the above code working, can replace with this
'''
from editmodule.editsession import EditSession

editsession = EditSession(config.ldm_stable)
editsession.init_image(
    image_path="./input/gnochi_mirror.jpeg",
    prompt = "a cat sitting next to a mirror"
)
editsession.init_controller(
    prompts=[
        "a cat sitting next to a mirror",
        "a silver cat sculpture sitting next to a mirror"
    ],
    cross_replace_steps = {'default_': .8, }
    self_replace_steps = .6
    blend_word = ((('cat',), ("cat",))) # for local edit
    eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
)
images, x_t = editsession.run()
'''
from diffusers import DiffusionPipeline

from editmodule.display_utils import run_and_display
from editmodule.prompt2prompt import make_controller
from editmodule.nullinversion import NullInversion

class EditSession:
    def __init__(self, model: DiffusionPipeline):
        self.null_inversion = NullInversion(model)

    def init_image(self, image_path: str, prompt: str):
        (image_gt, image_enc), x_t, uncond_embeddings = \
            self.null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)
        self.image_gt = image_gt
        self.image_enc = image_enc
        self.x_t = x_t
        self.uncond_embeddings = uncond_embeddings

    def init_controller(self, prompts, cross_replace_steps, self_replace_steps, blend_word, eq_params):
        self.prompts = prompts
        self.controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)

    def run(self):
        # Can replace run_and_display with a method that edits images only, that does not display
        images, x_t = run_and_display(self.prompts, self.controller, run_baseline=False, latent=self.x_t, uncond_embeddings=self.uncond_embeddings)
        return images, x_t
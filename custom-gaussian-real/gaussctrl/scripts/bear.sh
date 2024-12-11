ns-train splatfacto --output-dir unedited_models --experiment-name bear --viewer.quit-on-train-completion True nerfstudio-data --data data/bear

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a grizzly bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a golden bear statue in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-12-04_044745/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-12-04_044745/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a fluffy cat in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-12-04_044745/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a <new1> cat in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True --pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"

ns-viewer --load-config outputs/bear/gaussctrl/2024-12-04_080339/config.yml

ns-viewer --load-config outputs/bear/gaussctrl/2024-12-04_082444/config.yml

ns-viewer --load-config outputs/bear/gaussctrl/2024-12-05_060825/config.yml

ns-train splatfacto --output-dir unedited_models --experiment-name stone_horse --viewer.quit-on-train-completion True nerfstudio-data --data data/stone_horse

ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-07-11_173710/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.edit_prompt "a photo of a giraffe in front of the museum" --pipeline.reverse_prompt "a photo of a stone horse in front of the museum" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'stone horse' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-07-11_173710/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.edit_prompt "a photo of a zebra in front of the museum" --pipeline.reverse_prompt "a photo of a stone horse in front of the museum" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'stone horse' --viewer.quit-on-train-completion True 




# originalGS
ns-train splatfacto --output-dir unedited_models --experiment-name stone_horse --viewer.quit-on-train-completion True nerfstudio-data --data data/stone_horse
ns-viewer --load-config unedited_models/stone_horse/splatfacto/2024-12-07_132811/config.yml
ns-render camera-path --load-config unedited_models/stone_horse/splatfacto/2024-12-07_132811/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/stone_horse/camera_paths/2024-12-07-14-09-20.json --output-path renders/stone_horse/originalgs_2024-12-07-14-09-20.mp4

# custom gaussian
ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-12-07_132811/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.edit_prompt "a photo of a <new1> horse in front of the museum" --pipeline.reverse_prompt "a photo of a horse in front of the museum" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-viewer --load-config outputs/stone_horse/gaussctrl/2024-12-07_152249/config.yml
ns-render camera-path --load-config outputs/stone_horse/gaussctrl/2024-12-07_152249/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/stone_horse/camera_paths/2024-12-07-14-09-20.json --output-path renders/stone_horse/customgaussian_2024-12-07-14-09-20.mp4

# gaussctrl
ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-12-07_202048/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.edit_prompt "a photo of a horse plush in front of the museum" --pipeline.reverse_prompt "a photo of a stone horse in front of the museum" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'stone horse' --viewer.quit-on-train-completion True 
ns-viewer --load-config outputs/stone_horse/gaussctrl/2024-12-07_210159/config.yml
ns-render camera-path --load-config outputs/stone_horse/gaussctrl/2024-12-07_210159/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/stone_horse/camera_paths/2024-12-07-14-09-20.json --output-path renders/stone_horse/gaussctrl_2024-12-07-14-09-20.mp4

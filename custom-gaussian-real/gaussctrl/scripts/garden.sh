ns-train splatfacto --output-dir unedited_models --experiment-name garden --viewer.quit-on-train-completion True nerfstudio-data --data data/garden

ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-07-11_173647/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "a photo of a fake plant on a table in the garden in the snow" --pipeline.reverse_prompt "a photo of a fake plant on a table in the garden" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 

ns-viewer --load-config unedited_models/garden/splatfacto/2024-12-05_044750/config.yml
ns-viewer --load-config outputs/garden/gaussctrl/2024-12-05_053537/config.yml

ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-07-11_173647/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "a photo of a blue vase on a table in the garden" --pipeline.reverse_prompt "a photo of a vase on a table in the garden" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-viewer --load-config outputs/garden/gaussctrl/2024-12-05_131700/config.yml

ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-12-05_044750/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "a photo of a <new1> vase on a table in the garden" --pipeline.reverse_prompt "a photo of a vase on a table in the garden" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-12-05_044750/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "A photo of a round wooden table in a garden with a <new1> vase placed on top and a ball underneath the table." --pipeline.reverse_prompt "A photo of a round wooden table in a garden with a vase placed on top and a ball underneath the table." --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-12-05_044750/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "A photo of a round wooden table in a garden with a <new1> vase placed on top." --pipeline.reverse_prompt "A photo of a round wooden table in a garden with a vase placed on top." --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-12-05_044750/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "A photo of a table in a garden with a <new1> vase placed on top." --pipeline.reverse_prompt "A photo of a table in a garden with a vase placed on top." --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 

ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_112216/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/customgaussian_2024-12-05-12-44-52.mp4
ns-render camera-path --load-config unedited_models/garden/splatfacto/2024-12-05_044750/config.yml  --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/original_gs-2024-12-05-12-44-52.mp4

ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_131700/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/gaussctrl_2024-12-05-14-05-55.mp4

# CustomGaussian: A photo of a table in a garden with a <new1> vase placed on top.
ns-viewer --load-config outputs/garden/gaussctrl/2024-12-05_124254/config.yml
ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_124254/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/customgaussian_final_2024-12-05-12-44-52.mp4
ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_124254/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/garden/camera_paths/2024-12-07-03-53-13.json --output-path renders/garden/customgaussian_paper_2024-12-07-03-53-13.mp4

# GaussCtrl: A photo of a table in a garden with a blue vase placed on top.
ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-12-05_044750/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.edit_prompt "A photo of a table in a garden with a blue vase placed on top." --pipeline.reverse_prompt "A photo of a table in a garden with a vase placed on top." --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_231644/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/gaussctrl_final_2024-12-05-12-44-52.mp4
ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_231644/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/garden/camera_paths/2024-12-07-03-53-13.json --output-path renders/garden/gaussctrl_paper_2024-12-07-03-53-13.mp4

# GaussCtrl: a photo of a blue vase on a table in the garden
ns-render camera-path --load-config outputs/garden/gaussctrl/2024-12-05_131700/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/gaussctrl/data/garden/camera_paths/2024-12-05-14-05-55.json --output-path renders/garden/gaussctrl_paper_2024-12-05-14-05-55.mp4

# OriginalGS
ns-render camera-path --load-config unedited_models/garden/splatfacto/2024-12-05_044750/config.yml --camera-path-filename /app/data/wonsik/11785-final-project/custom-gaussian-real/gaussctrl/data/garden/camera_paths/2024-12-07-03-53-13.json --output-path renders/garden/originalgs_paper_2024-12-07-03-53-13.mp4

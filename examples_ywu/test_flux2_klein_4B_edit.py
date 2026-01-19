import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import time

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.to(device)

img = Image.open("data/c442f847-5572-7aa9-806f-fc68467b2118.jpg").convert("RGB")

prompt = "Put the object in better background"
for i in range(2):
    t_start = time.time()
    images = pipe(
        image=img,
        prompt=prompt,
        # height=1024,
        # width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        num_images_per_prompt=1,
        # generator=torch.Generator(device=device).manual_seed(0)
    ).images
    t_end = time.time()
    print(f"inference time: {t_end - t_start} seconds")
for idx, image in enumerate(images):
    image.save(f"results/c442f847-5572-7aa9-806f-fc68467b2118_flux2-klein_4B_edit_{idx}.png")
    # image.save(f"results/0d6e8568-7d8c-8a53-f8b1-6e0fca38a8be_flux2-klein_4B_edit_{idx}.png")
# image.save("results/flux2-klein_edit.png")
# CUDA_VISIBLE_DEVICES=1 python test_flux2_klein_4B.py
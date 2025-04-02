import random

import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from nunchaku import NunchakuFluxTransformer2dModel
from diffusers.utils import load_image
import numpy as np


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = 'A anime style girl with messy beach waves.'
control_image_depth = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/depth.jpg")
control_mode_depth = 2

control_image_canny = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/canny.jpg")
control_mode_canny = 0

width, height = control_image_depth.size

image = pipe(
    prompt,
    control_image=[control_image_depth, control_image_canny],
    control_mode=[control_mode_depth, control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.3, 0.1],
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.manual_seed(SEED),
).images[0]


image.save("reference-controlnet-flux.1-dev.png")

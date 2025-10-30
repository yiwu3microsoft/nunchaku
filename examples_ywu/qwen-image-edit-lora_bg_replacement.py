import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

rank = 128  # you can also use rank=128 model to improve the quality
num_inference_steps = 50
# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
)

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.to("cuda:0")
# if get_gpu_memory() > 18:
#     pipeline.enable_model_cpu_offload()
# else:
#     # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
#     transformer.set_offload(
#         True, use_pin_memory=False, num_blocks_on_gpu=1
#     )  # increase num_blocks_on_gpu if you have more VRAM
#     pipeline._exclude_from_cpu_offload.append("transformer")
#     pipeline.enable_sequential_cpu_offload()



################
# AttributeError: 'NunchakuQwenImageTransformer2DModel' object has no attribute 'update_lora_params'
################

### LoRA Related Code ###
transformer.update_lora_params(
    "/home/yiwu3/projects/data/data_BR1S41-S2-25/outputs/qwen_image_edit_lora_fill_bc3000/checkpoint-3000/pytorch_lora_weights.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(1)  # Your LoRA strength here
### End of LoRA Related Code ###

image = load_image("./data/c442f847-5572-7aa9-806f-fc68467b2118.jpg")

prompt = "Change the background to a house for sale with a yard, creating a welcoming atmosphere. Keep the man's pose, position, and clothing the same."

inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}

output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"results/qwen-image-edit-r{rank}-{num_inference_steps}steps-lora_bg_replace.png")

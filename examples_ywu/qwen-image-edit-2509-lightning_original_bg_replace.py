import math, time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 4  # you can also use the 8-step model to improve the quality
# model_path = f"lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-{num_inference_steps}steps-V1.0-bf16.safetensors"

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", scheduler=scheduler, torch_dtype=torch.bfloat16
)
pipeline.to("cuda:0")

pipeline.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name=f"Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-{num_inference_steps}steps-V1.0-bf16.safetensors"
)

image1 = load_image("./data/c442f847-5572-7aa9-806f-fc68467b2118.jpg")

prompt = "Change the background to a house for sale with a yard, creating a welcoming atmosphere. Keep the man's pose, position, and clothing the same."
inputs = {
    "image": [image1],
    "prompt": prompt,
    "true_cfg_scale": 1.0,
    "num_inference_steps": num_inference_steps,
}
t_start = time.time()
output = pipeline(**inputs)
print("processing time:", time.time()-t_start)
output_image = output.images[0]
output_image.save(f"results/qwen-image-edit-2509-lightning-original-{num_inference_steps}steps_bg_replace.png")

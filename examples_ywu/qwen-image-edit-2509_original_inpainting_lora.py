import os
import torch, time
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to("cuda:0")

pipeline.load_lora_weights(
    "ostris/qwen_image_edit_inpainting"
)

# pipeline.enable_model_cpu_offload()
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("./data/MonaLisa_green_mask.png")
# image2 = Image.open("input2.png")
# prompt = "A man stands in front of a house for sale with a yard, creating a welcoming atmosphere. Keep the man's pose, position, and clothing the same."
prompt = "painting of a woman wearing sunglasses"

inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    t_start = time.time()
    output = pipeline(**inputs)
    print("processing time:", time.time()-t_start)
    output_image = output.images[0]
    output_image.save("results/MonaLisa_inpainting.png")
    # print("image saved at", os.path.abspath("output_image_edit_plus.png"))

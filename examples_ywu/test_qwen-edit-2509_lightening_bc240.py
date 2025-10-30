import os, csv
import torch
from PIL import Image
import math

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

def read_eval_data(fname_tsv, path_image, path_mask):
    # fname_tsv = '/home/yiwu3/projects/data/background_change_furniture_training/validation/validation_rewritten_prompt_map_real_240.csv'
    # path_image = '/home/yiwu3/projects/data/background_change_furniture_training/validation/validation_data_map_real_240'
    # path_mask = '/home/yiwu3/projects/data/background_change_furniture_training/validation/validation_data_map_real_240_controls'

    data_dict = {}

    with open(fname_tsv, mode='r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            if 'ImageID' in row:
                image_id = row['ImageID']
            elif 'RID' in row:
                image_id = row['RID']
            else:
                raise ValueError("Invalid TSV format: Missing ImageID or RID column")
            data_dict[image_id] = {
                # 'KeywordsPrompt': row['KeywordsPrompt'],
                'ProductInfo': row['ProductInfo'],
                'RewrittenPrompt': row['RewrittenPrompt'],
                'fname_original': f"{path_image}/{image_id}.jpg",
                'fname_mask': f"{path_mask}/{image_id}_seg_mask.png",
            }
    return data_dict

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
rank = 128  # you can also use the rank=128 model to improve the quality
model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.to("cuda:0")

data_eval = "map_real_240"

save_root = f"./results/{data_eval}#qwen-edit-2509-lightning-r{rank}-{num_inference_steps}steps"

if not os.path.exists(save_root):
    os.makedirs(save_root)



if data_eval == "map_real_240":
    path_image = '/home/yiwu3/projects/data/data_background_change/map_real_240'
    path_mask = '/home/yiwu3/projects/data/data_background_change/validation_data_map_real_240_controls'

    fname_tsv = '/home/yiwu3/projects/data/data_background_change/validation_rewritten_prompt_map_real_240.tsv'


data_dict = read_eval_data(fname_tsv, path_image, path_mask)
# data_dict = read_tsv(fname_tsv)

count = 0

for image_id, data in data_dict.items():
    count+=1

    print(f"ImageID: {image_id}, {count}/{len(data_dict)}")

    # Load the original image
    img_original = Image.open(data_dict[image_id]['fname_original']).convert("RGB")

    mask_original = Image.open(data_dict[image_id]['fname_mask']).convert("L")

    # create a white background and composite the foreground using the mask (white = foreground)
    white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
    masked_image = Image.composite(img_original, white_bg, mask_original)


    # image1 = Image.open("./22c28f74-1e25-a83b-b562-a2e0bd663c9d.jpg")
    # image2 = Image.open("input2.png")
    # prompt = "A man stands in front of a house for sale with a yard, creating a welcoming atmosphere. Keep the man's pose, position, and clothing the same."
    # prompt = "Change the background to a house for sale with a yard, creating a welcoming atmosphere. Keep the man's pose, position, and clothing the same."
    prompt = data['RewrittenPrompt']
    productInfo = data['ProductInfo']
    print(f"ProductInfo: {productInfo}; Prompt: {prompt}")

    prompt = "The object is " + productInfo + ". " + "Change the background of the object: " + prompt + " Keep the object's pose, position, and details the same."

    # inputs = {
    #     "image": [masked_image],
    #     "prompt": prompt,
    #     "generator": torch.manual_seed(0),
    #     "true_cfg_scale": 4.0,
    #     "negative_prompt": " ",
    #     "num_inference_steps": 40,
    #     "guidance_scale": 1.0,
    #     "num_images_per_prompt": 1,
    # }
    inputs = {
        "image": [masked_image],
        "prompt": prompt,
        "true_cfg_scale": 1.0,
        "num_inference_steps": num_inference_steps,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(f"{save_root}/{image_id}_qwen-edit-2509-lightning-r{rank}-{num_inference_steps}steps.jpg")
        # print("image saved at", os.path.abspath("output_image_edit_plus.png"))

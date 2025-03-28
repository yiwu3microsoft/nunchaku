import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline, FluxFillPipeline, FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

from nunchaku import NunchakuFluxTransformer2dModel


def test_flux_dev_canny():
    transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-canny-dev")
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Canny-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."  # noqa: E501
    control_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
    )

    processor = CannyDetector()
    control_image = processor(
        control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
    )

    image = pipe(
        prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=50, guidance_scale=30.0
    ).images[0]
    image.save("flux.1-canny-dev.png")


def test_flux_dev_depth():
    transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-depth-dev")

    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."  # noqa: E501
    control_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
    )

    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    control_image = processor(control_image)[0].convert("RGB")

    image = pipe(
        prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=30, guidance_scale=10.0
    ).images[0]
    image.save("flux.1-depth-dev.png")


def test_flux_dev_fill():
    image = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/example.png")
    mask = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/mask.png")

    transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-fill-dev")
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")
    image = pipe(
        prompt="A wooden basket of a cat.",
        image=image,
        mask_image=mask,
        height=1024,
        width=1024,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
    ).images[0]
    image.save("flux.1-fill-dev.png")


def test_flux_dev_redux():
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        text_encoder_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
    pipe_prior_output = pipe_prior_redux(image)
    images = pipe(guidance_scale=2.5, num_inference_steps=50, **pipe_prior_output).images
    images[0].save("flux.1-redux-dev.png")

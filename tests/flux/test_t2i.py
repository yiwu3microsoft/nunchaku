import os
import tempfile

import pytest
import torch
from diffusers import FluxPipeline
from peft.tuners import lora
from safetensors.torch import save_file
from tqdm import tqdm

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.lora.flux import comfyui2diffusers, convert_to_nunchaku_flux_lowrank_dict, detect_format, xlab2diffusers
from ..data import get_dataset
from ..utils import already_generate, compute_lpips, hash_str_to_int


def run_pipeline(dataset, pipeline: FluxPipeline, save_dir: str, forward_kwargs: dict = {}):
    os.makedirs(save_dir, exist_ok=True)
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for row in tqdm(dataset):
        filename = row["filename"]
        prompt = row["prompt"]
        seed = hash_str_to_int(filename)
        image = pipeline(prompt, generator=torch.Generator().manual_seed(seed), **forward_kwargs).images[0]
        image.save(os.path.join(save_dir, f"{filename}.png"))


@pytest.mark.parametrize(
    "precision,height,width,num_inference_steps,guidance_scale,use_qencoder,cpu_offload,max_dataset_size,expected_lpips",
    [
        ("int4", 1024, 1024, 4, 0, False, False, 16, 0.258),
        ("int4", 1024, 1024, 4, 0, True, False, 16, 0.41),
        ("int4", 1024, 1024, 4, 0, True, False, 16, 0.41),
        ("int4", 1920, 1080, 4, 0, False, False, 16, 0.258),
        ("int4", 600, 800, 4, 0, False, False, 16, 0.29),
    ],
)
def test_flux_schnell(
    precision: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    use_qencoder: bool,
    cpu_offload: bool,
    max_dataset_size: int,
    expected_lpips: float,
):
    dataset = get_dataset(name="MJHQ", max_dataset_size=max_dataset_size)
    save_root = os.path.join("results", "schnell", f"w{width}h{height}t{num_inference_steps}g{guidance_scale}")

    save_dir_16bit = os.path.join(save_root, "bf16")
    if not already_generate(save_dir_16bit, max_dataset_size):
        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipeline = pipeline.to("cuda")

        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()

    save_dir_4bit = os.path.join(
        save_root, f"{precision}-qencoder" if use_qencoder else f"{precision}" + ("-cpuoffload" if cpu_offload else "")
    )
    if not already_generate(save_dir_4bit, max_dataset_size):
        pipeline_init_kwargs = {}
        if precision == "int4":
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-int4-flux.1-schnell", offload=cpu_offload
            )
        else:
            assert precision == "fp4"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-fp4-flux.1-schnell", precision="fp4", offload=cpu_offload
            )
        pipeline_init_kwargs["transformer"] = transformer
        if use_qencoder:
            text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
            pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        pipeline = pipeline.to("cuda")
        if cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_4bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()
    lpips = compute_lpips(save_dir_16bit, save_dir_4bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.05


LORA_PATH_MAP = {
    "hypersd8": "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors",
    "realism": "XLabs-AI/flux-RealismLora/lora.safetensors",
    "ghibsky": "aleksa-codes/flux-ghibsky-illustration/lora.safetensors",
    "anime": "alvdansen/sonny-anime-fixed/araminta_k_sonnyanime_fluxd_fixed.safetensors",
    "sketch": "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch/FLUX-dev-lora-children-simple-sketch.safetensors",
    "yarn": "linoyts/yarn_art_Flux_LoRA/pytorch_lora_weights.safetensors",
    "haunted_linework": "alvdansen/haunted_linework_flux/hauntedlinework_flux_araminta_k.safetensors",
}


def run_test_flux_dev(
    precision: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    use_qencoder: bool,
    cpu_offload: bool,
    lora_name: str | None,
    lora_scale: float,
    max_dataset_size: int,
    expected_lpips: float,
):
    save_root = os.path.join(
        "results",
        "dev",
        f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"
        + ("-qencoder" if use_qencoder else "")
        + (f"-{lora_name}_{lora_scale:.1f}" if lora_name else ""),
    )
    dataset = get_dataset(
        name="MJHQ" if lora_name in [None, "hypersd8"] else lora_name, max_dataset_size=max_dataset_size
    )

    save_dir_16bit = os.path.join(save_root, "bf16")
    if not already_generate(save_dir_16bit, max_dataset_size):
        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipeline = pipeline.to("cuda")
        if lora_name is not None:
            pipeline.load_lora_weights(
                os.path.dirname(LORA_PATH_MAP[lora_name]),
                weight_name=os.path.basename(LORA_PATH_MAP[lora_name]),
                adapter_name="lora",
            )
            for n, m in pipeline.transformer.named_modules():
                if isinstance(m, lora.LoraLayer):
                    for name in m.scaling.keys():
                        m.scaling[name] = lora_scale

        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()

    save_dir_4bit = os.path.join(save_root, f"{precision}-qencoder" if use_qencoder else f"{precision}")
    if not already_generate(save_dir_4bit, max_dataset_size):
        pipeline_init_kwargs = {}
        if precision == "int4":
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-int4-flux.1-dev", offload=cpu_offload
            )
        else:
            assert precision == "fp4"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-fp4-flux.1-dev", precision="fp4", offload=cpu_offload
            )
        if lora_name is not None:
            lora_path = LORA_PATH_MAP[lora_name]
            lora_format = detect_format(lora_path)
            if lora_format != "svdquant":
                if lora_format == "comfyui":
                    input_lora = comfyui2diffusers(lora_path)
                elif lora_format == "xlab":
                    input_lora = xlab2diffusers(lora_path)
                elif lora_format == "diffusers":
                    input_lora = lora_path
                else:
                    raise ValueError(f"Invalid LoRA format {lora_format}.")
                state_dict = convert_to_nunchaku_flux_lowrank_dict(
                    "mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors", input_lora
                )
                with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp_file:
                    save_file(state_dict, tmp_file.name)
                    transformer.update_lora_params(tmp_file.name)
            else:
                transformer.update_lora_params(lora_path)
            transformer.set_lora_strength(lora_scale)

        pipeline_init_kwargs["transformer"] = transformer
        if use_qencoder:
            text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
            pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        pipeline = pipeline.to("cuda")
        if cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_4bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()
    lpips = compute_lpips(save_dir_16bit, save_dir_4bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.05


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_flux_dev_base(cpu_offload: bool):
    run_test_flux_dev(
        precision="int4",
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=cpu_offload,
        lora_name=None,
        lora_scale=0,
        max_dataset_size=8,
        expected_lpips=0.16,
    )


def test_flux_dev_qencoder_800x600():
    run_test_flux_dev(
        precision="int4",
        height=800,
        width=600,
        num_inference_steps=50,
        guidance_scale=3.5,
        use_qencoder=True,
        cpu_offload=False,
        lora_name=None,
        lora_scale=0,
        max_dataset_size=8,
        expected_lpips=0.36,
    )


def test_flux_dev_hypersd8_1080x1920():
    run_test_flux_dev(
        precision="int4",
        height=1080,
        width=1920,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=False,
        lora_name="hypersd8",
        lora_scale=0.125,
        max_dataset_size=8,
        expected_lpips=0.44,
    )


@pytest.mark.parametrize(
    "num_inference_steps,lora_name,lora_scale,cpu_offload,expected_lpips",
    [
        (25, "realism", 0.9, False, 0.16),
        (25, "ghibsky", 1, False, 0.16),
        (28, "anime", 1, False, 0.27),
        (24, "sketch", 1, False, 0.35),
        (28, "yarn", 1, False, 0.22),
        (25, "haunted_linework", 1, False, 0.34),
    ],
)
def test_flux_dev_loras(num_inference_steps, lora_name, lora_scale, cpu_offload, expected_lpips):
    run_test_flux_dev(
        precision="int4",
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=cpu_offload,
        lora_name=lora_name,
        lora_scale=lora_scale,
        max_dataset_size=8,
        expected_lpips=expected_lpips,
    )


@pytest.mark.parametrize(
    "use_qencoder,cpu_offload,memory_limit",
    [
        (False, False, 17),
        (False, True, 13),
        (True, False, 12),
        (True, True, 6),
    ],
)
def test_flux_schnell_memory(use_qencoder: bool, cpu_offload: bool, memory_limit: float):
    torch.cuda.reset_peak_memory_stats()
    pipeline_init_kwargs = {
        "transformer": NunchakuFluxTransformer2dModel.from_pretrained(
            "mit-han-lab/svdq-int4-flux.1-schnell", offload=cpu_offload
        )
    }
    if use_qencoder:
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
        pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
    ).to("cuda")

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()

    pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=50, guidance_scale=0
    )
    memory = torch.cuda.max_memory_reserved(0) / 1024**3
    assert memory < memory_limit
    del pipeline
    # release the gpu memory
    torch.cuda.empty_cache()

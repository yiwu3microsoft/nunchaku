import os
import warnings
from typing import Any, Optional

import torch
from diffusers import __version__
from huggingface_hub import constants, hf_hub_download
from torch import nn

from nunchaku.utils import ceil_divide


class NunchakuModelLoaderMixin:
    @classmethod
    def _build_model(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> tuple[nn.Module, str, str]:
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            unquantized_part_path = os.path.join(dirname, "unquantized_layers.safetensors")
            transformer_block_path = os.path.join(dirname, "transformer_blocks.safetensors")
        else:
            download_kwargs = {
                "subfolder": subfolder,
                "repo_type": "model",
                "revision": kwargs.get("revision", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "local_dir": kwargs.get("local_dir", None),
                "user_agent": kwargs.get("user_agent", None),
                "force_download": kwargs.get("force_download", False),
                "proxies": kwargs.get("proxies", None),
                "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
                "token": kwargs.get("token", None),
                "local_files_only": kwargs.get("local_files_only", None),
                "headers": kwargs.get("headers", None),
                "endpoint": kwargs.get("endpoint", None),
                "resume_download": kwargs.get("resume_download", None),
                "force_filename": kwargs.get("force_filename", None),
                "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
            }
            unquantized_part_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="unquantized_layers.safetensors", **download_kwargs
            )
            transformer_block_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="transformer_blocks.safetensors", **download_kwargs
            )

        config, _, _ = cls.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=kwargs.get("cache_dir", None),
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies", None),
            local_files_only=kwargs.get("local_files_only", None),
            token=kwargs.get("token", None),
            revision=kwargs.get("revision", None),
            user_agent={"diffusers": __version__, "file_type": "model", "framework": "pytorch"},
            **kwargs,
        )

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        return transformer, unquantized_part_path, transformer_block_path


def pad_tensor(tensor: Optional[torch.Tensor], multiples: int, dim: int, fill: Any = 0) -> torch.Tensor | None:
    if multiples <= 1:
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if shape[dim] % multiples == 0:
        return tensor
    shape[dim] = ceil_divide(shape[dim], multiples) * multiples
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.fill_(fill)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def get_precision(precision: str, device: str | torch.device, pretrained_model_name_or_path: str | None = None) -> str:
    assert precision in ("auto", "int4", "fp4")
    if precision == "auto":
        if isinstance(device, str):
            device = torch.device(device)
        capability = torch.cuda.get_device_capability(0 if device.index is None else device.index)
        sm = f"{capability[0]}{capability[1]}"
        precision = "fp4" if sm == "120" else "int4"
    if pretrained_model_name_or_path is not None:
        if precision == "int4":
            if "fp4" in pretrained_model_name_or_path:
                warnings.warn("The model may be quantized to fp4, but you are loading it with int4 precision.")
        elif precision == "fp4":
            if "int4" in pretrained_model_name_or_path:
                warnings.warn("The model may be quantized to int4, but you are loading it with fp4 precision.")
    return precision

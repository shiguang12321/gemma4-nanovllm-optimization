import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_parameter_or_buffer(model: nn.Module, name: str):
    try:
        return model.get_parameter(name)
    except (AttributeError, ValueError, KeyError):
        pass
    try:
        return model.get_buffer(name)
    except (AttributeError, ValueError, KeyError):
        return None


def translate_weight_name(name: str) -> str | None:
    if name.startswith(("model.vision_tower.", "model.embed_vision.", "model.audio_tower.")):
        return None
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    return name


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for loaded_name in f.keys():
                weight_name = translate_weight_name(loaded_name)
                if weight_name is None:
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = get_parameter_or_buffer(model, param_name)
                        if param is None:
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(loaded_name), shard_id)
                        break
                else:
                    param = get_parameter_or_buffer(model, weight_name)
                    if param is None:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(loaded_name))

import flax.linen as nn

from typing import Callable
from ml_collections import ConfigDict

from DistJax.parallelism.sharding import shard_module_params


def prepare_module(
    layer: Callable[..., nn.Module], layer_name: str, config: ConfigDict
) -> Callable[..., nn.Module]:
    if config.get("fsdp", None) is not None and layer_name in config.fsdp.modules:
        layer = shard_module_params(
            layer,
            axis_name=config.data_axis_name,
            min_weight_size=config.fsdp.min_weight_size,
        )

    if config.get("remat", None) is not None and layer_name in config.remat:
        layer = nn.remat(layer, prevent_cse=False)

    return layer

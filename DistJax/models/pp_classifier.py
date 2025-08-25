import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
from typing import Callable
import functools

from DistJax.parallelism.pipeline_parallel import PipelineModule , ModelParallelWrapper
from DistJax.models.mlp import MLPBlock , MLPLayers



class PPClassifier(nn.Module):
    config: ConfigDict
    pipeline_module_class: Callable[..., nn.Module] = PipelineModule

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:

        x = ModelParallelWrapper(
            module_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size,  # type: ignore
                dtype=self.config.dtype,
            ),
            split_rngs=True,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=0,
            name="input_dense",
        )(x)

        stage_module_fn = functools.partial(
            MLPLayers, config=self.config, train=train, name="mlp_layer"
        )

        pipeline_module_fn = functools.partial(
            self.pipeline_module_class,
            model_axis_name=self.config.model_axis_name,
            num_microbatches=self.config.num_microbatches,
            module_fn=stage_module_fn,
        )

        module = ModelParallelWrapper(
            module_fn=pipeline_module_fn,
            model_axis_name=self.config.model_axis_name,
            name="pipeline",
        )

        x = module(x)

        output_wrapper = functools.partial(
            ModelParallelWrapper,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=self.config.model_axis_size - 1,
        )

        x = output_wrapper(
            module_fn=functools.partial(nn.LayerNorm, dtype=self.config.dtype),
            name="output_norm",
        )(
            x
        )  # type: ignore

        x = output_wrapper(
            module_fn=functools.partial(
                nn.Dense, features=self.config.num_classes, dtype=self.config.dtype  # type: ignore
            ),
            name="output_dense",
        )(x)

        x = x.astype(jnp.float32)
        return x

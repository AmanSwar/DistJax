import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
import functools
from typing import Callable

from DistJax.parallelism.tensor_parallel import TPDense
from DistJax.models.mlp import MLPBlockInput , MLPBlockOutput , TPMLPLayers , TPMLPBlock



class TPClassifier(nn.Module):
    config: ConfigDict
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size // tp_size,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            skip_communication=True,  # Input already gathered.
            name="input_layer",
        )(x)
        # Backbone MLP blocks
        x = TPMLPLayers(
            config=self.config, train=train, name="mlp", block_class=self.block_class
        )(x)
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.num_classes,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            skip_communication=True,  # Manual communication.
            name="output_layer",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
        )(x)
        x = jax.lax.psum(x, axis_name=self.config.model_axis_name)
        x = x.astype(jnp.float32)
        return x

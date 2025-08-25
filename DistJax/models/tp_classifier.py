import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict
import functools
from typing import Callable

from DistJax.parallelism.tensor_parallel import TPDense
from DistJax.models.mlp import MLPBlockInput , MLPBlockOutput


class TPMLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockInput,
                config=self.config,
                features=self.config.hidden_size * self.config.mlp_expansion // tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            name="input",
        )(x)
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockOutput,
                config=self.config,
                features=input_features * tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
            name="output",
        )(x)
        return x


class TPMLPLayers(nn.Module):
    config: ConfigDict
    train: bool
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        module = self.block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry) + carry, None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
            metadata_params={
                "partition_name": None
            },  # We do not need to partition the parameters over the layer axis.
        )(module, x, ())
        return x


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

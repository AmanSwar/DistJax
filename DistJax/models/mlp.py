import jax
import flax.linen as nn

from ml_collections import ConfigDict 
from typing import Callable


class MLPBlock(nn.Module):

    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        input_feat = x.shape[-1]
        residual = x
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)

        x = nn.Dense(
            features=self.config.hidden_size * self.config.mlp_expansion,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)

        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
        x = nn.Dense(features=input_feat, dtype=self.config.dtype, name="output")(x)

        return x + residual


class MLPLayers(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Scan version
        block_class = MLPBlock
        if "MLP" in self.config.remat:
            block_class = nn.remat(block_class, prevent_cse=False)
        block = block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), ()),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
        )(block, x, ())

        return x


class MLPBlockInput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_norm:
            x = nn.RMSNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


class MLPBlockOutput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.silu(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


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

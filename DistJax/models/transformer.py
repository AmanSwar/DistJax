import flax.linen as nn
import jax

from functools import partial
from typing import Callable , Tuple
from ml_collections import ConfigDict

from DistJax.core.attention import dot_product_attention
from DistJax.parallelism.tensor_parallel_async import TPAsyncDense , TPNorm
from DistJax.core.module_utils import prepare_module
from DistJax.models.mlp import MLPBlockInput

class QKVDense(nn.Module):
    config: ConfigDict
    num_heads: int
    head_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:

        q = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="query",
        )(x)

        k = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="key",
        )(x)

        v = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="value",
        )(x)

        if self.config.normalize_qk:
            q = nn.RMSNorm(
                dtype=self.config.dtype,
                name="query_norm",
            )(q)

            k = nn.RMSNorm(
                dtype=self.config.dtype,
                name="key_norm",
            )(k)

        return q, k, v


class AttnOut(nn.Module):

    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.DenseGeneral(
            features=self.features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="out",
        )(x)
        return x


class TPMultiHeadAttn(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads

        x = TPNorm(config=self.config, name="pre_norm")(x)

        q, k, v = TPAsyncDense(
            dense_fn=partial(
                QKVDense,
                config=self.config,
                num_heads=num_heads // tp_size,
                head_dim=head_dim,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="qkv",
        )(x)

        x = dot_product_attention(q, k, v, self.mask)

        x = TPAsyncDense(
            dense_fn=partial(
                AttnOut,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="out",
        )(x)
        return x


class TPTransformerBlock(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        attn_layer = prepare_module(TPMultiHeadAttn, "Attn", self.config)
        attn_out = attn_layer(
            config=self.config,
            train=self.train,
            mask=self.mask,
            name="attn",
        )(x)

        attn_out = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=not self.train
        )(attn_out)

        x = x + attn_out
        mlp_layer = prepare_module(TPAsyncMLPBlock, "MLP", self.config)
        mlp_out = mlp_layer(
            config=self.config,
            train=self.train,
            name="mlp",
        )(x)

        mlp_out = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=not self.train
        )(mlp_out)

        x = x + mlp_out

        return x


class QKVMLPDense(nn.Module):

    config: ConfigDict
    num_heads: int
    head_dim: int
    mlp_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(
        self, x: jax.Array
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:

        h = MLPBlockInput(
            config=self.config,
            features=self.mlp_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            use_norm=False,
            name="mlp",
        )(x)

        q, k, v = QKVDense(
            config=self.config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="qkv",
        )(x)

        return h, (q, k, v)


class AttnMLPOut(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Tuple[jax.Array, jax.Array]) -> jax.Array:
        mlp_h, attn_v = x
        mlp_out = MLPBlockOutput(
            config=self.config,
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="mlp",
        )(mlp_h)
        attn_out = AttnOut(
            config=self.config,
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="attn",
        )(attn_v)
        out = mlp_out + attn_out
        return out


class TPTransformerParallelBlock(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        residual = x

        x = TPNorm(config=self.config, name="pre_norm")(x)

        h, (q, k, v) = TPAsyncDense(
            dense_fn=partial(
                QKVMLPDense,
                config=self.config,
                num_heads=self.config.num_heads // tp_size,
                head_dim=self.config.head_dim,
                mlp_dim=self.config.hidden_size * self.config.mlp_expansion // tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="hqkv",
        )(x)

        v = dot_product_attention(q, k, v, self.mask)

        block_out = TPAsyncDense(
            dense_fn=partial(
                AttnMLPOut,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="out",
        )((h, v))

        block_out = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=not self.train
        )(block_out)

        out = residual + block_out

        return out

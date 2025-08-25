import flax.linen as nn
import jax

from functools import partial
from typing import Callable , Tuple
from ml_collections import ConfigDict


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



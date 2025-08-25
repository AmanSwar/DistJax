import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict


class Classifier(nn.Module):

    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:

        x = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)

        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(
            features=self.config.num_classes,
            dtype=self.config.dtype,
            name="output_dense",
        )(x)

        x = x.astype(jnp.float32)

        return x

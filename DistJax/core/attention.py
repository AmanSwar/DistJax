import flax.linen as nn
import jax.numpy as jnp
import jax


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None,
    softmax_dtype: jnp.dtype = jnp.float32,
):

    num_features = query.shape[-1]

    dtype = query.dtype
    scale = num_features**-0.5

    query = query * scale
    query = query.astype(softmax_dtype)

    key = key.astype(softmax_dtype)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(softmax_dtype).min)

    weights = nn.softmax(weights, axis=-1)
    weights = weights.astype(dtype)

    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)

    return new_vals

import jax

from typing import Callable

def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str):

    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def scale_init(init_fn: Callable, scale_factor: float = 1.0):

    def _init_fn(rng, *args, **kwargs):
        return scale_factor * init_fn(rng, *args, **kwargs)

    return _init_fn


def split_array_over_mesh(x: jax.Array, axis_name: str, split_axis: int) -> jax.Array:
    axis_size = jax.lax.psum(1, axis_name)
    axis_index = jax.lax.axis_index(axis_name)
    slice_size = x.shape[split_axis] // axis_size
    x = jax.lax.dynamic_slice_in_dim(
        x,
        axis_index * slice_size,
        slice_size,
        axis=split_axis,
    )
    return x

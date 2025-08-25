import jax

from typing import Callable

def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str):

    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def scale_init(init_fn: Callable, scale_factor: float = 1.0):

    def _init_fn(rng, *args, **kwargs):
        return scale_factor * init_fn(rng, *args, **kwargs)

    return _init_fn

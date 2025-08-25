import jax


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str):

    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)

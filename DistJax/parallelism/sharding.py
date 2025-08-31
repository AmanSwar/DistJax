import jax
import flax.linen as nn
from jax import lax
import numpy as np

import logging
from typing import Callable
import functools

from DistJax.core.training import PyTree , Parameter , TrainState


@jax.named_scope("shard_params")
def shard_params(
    params: PyTree, axis_name: str, min_weight_size: int = 2**18
) -> PyTree:

    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value = x
            names = (None,) * value.ndim
        if axis_name in names:
            logging.warning(
                f"Parameter {value.shape} with names {names} already sharded on axis {axis_name}."
            )
            return x
        elif value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_weight_size}."
            )
            return x
        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
            for i in idx:
                if shape[i] % axis_size == 0 and names[i] is None:
                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=lax.dynamic_slice_in_dim(  # Shard to keep on present device.
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        names=names[:i] + (axis_name,) + names[i + 1 :],
                    )
                    return p_sharded
            logging.warning(
                f"Could not shard {value.shape} with names {names} on axis {axis_name}, no suitable axis found."
            )
            return x

    return jax.tree_util.tree_map(
        _split,
        params,
        is_leaf=lambda x: isinstance(
            x, nn.Partitioned
        ),  # Consider a nn.Partitioned object as a leaf.
    )


def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)

    # Define a custom gradient for the gather operation.
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            # pmean_scatter
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True)
                / axis_size
            )

        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)


@jax.named_scope("gather_params")
def gather_params(params: PyTree, axis_name: str) -> PyTree:

    def _gather(p: Parameter) -> Parameter:
        if isinstance(p, nn.Partitioned) and axis_name in p.names:
            param_shard = p.names
            shard_axis = param_shard.index(axis_name)
            value = gather_array_with_mean_grads(
                p.value, axis=shard_axis, axis_name=axis_name
            )
            param_shard = (
                param_shard[:shard_axis] + (None,) + param_shard[shard_axis + 1 :]
            )
            if any([name is not None for name in param_shard]):
                return nn.Partitioned(value, param_shard)
            else:
                return value # type: ignore
        else:
            return p

    return jax.tree_util.tree_map(
        _gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def shard_module_params(
    target: nn.Module | Callable, axis_name: str, min_weight_size: int = 2**18
) -> nn.Module | Callable:

    return nn.map_variables(
        target,
        trans_in_fn=functools.partial(gather_params, axis_name=axis_name),
        trans_out_fn=functools.partial(
            shard_params, axis_name=axis_name, min_weight_size=min_weight_size
        ),
        mapped_collections="params",
        mutable=True,
    )


def init_fsdp(rng_key, x, model, optimizer_fn) -> TrainState:
    """Initializes the sharded model state."""
    init_rng, state_rng = jax.random.split(rng_key)
    variables = model.init({"params": init_rng}, x, train=False)
    params = variables.pop("params")
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer_fn, rng=state_rng
    )

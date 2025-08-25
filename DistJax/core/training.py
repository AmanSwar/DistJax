import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple
import numpy as np

import textwrap
from termcolor import colored
import os


PyTree = Any
Parameter = nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]] | None


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


class TrainState(train_state.TrainState):
    rng: jax.Array


def accum_grads_loop(
    batch: Batch,
    state: TrainState,
    key,
    n_minbatch: int,
    loss_fn: Callable,
) -> Tuple[Pytree, Metrics]:

    bs = batch.inputs.shape[0]
    min_batchSize = bs // n_minbatch
    keys = jax.random.split(key, n_minbatch)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    grads = None
    metrics: Metrics = None

    for batch_idx in range(n_minbatch):

        start_idx = batch_idx * min_batchSize
        end_idx = start_idx + min_batchSize
        # slicing Batch class
        miniBatch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], batch)

        (_, step_metric), step_grad = grad_fn(
            state.params, state.apply_fn, miniBatch, keys[batch_idx]
        )

        if grads is None:
            grads = step_grad
            metrics = step_metric

        else:
            grads = jax.tree_util.tree_map(jnp.add, grads, step_grad)
            metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metric)

    grads = jax.tree_util.tree_map(lambda g: g / n_minbatch, grads)
    return grads, metrics


def accum_grads_scan(
    batch: Batch,
    state: TrainState,
    key,
    n_minbatch: int,
    loss_fn: Callable,
) -> Tuple[Pytree, Metrics]:

    bs = batch.inputs.shape[0]
    min_batchSize = bs // n_minbatch
    keys = jax.random.split(key, n_minbatch)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _minbatch_step(batch_idx: jax.Array | int) -> Tuple[Pytree, Metrics]:
        # take out minibatch using dynamic slicing
        minibatch = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x,
                start_index=batch_idx * min_batchSize,
                slice_size=min_batchSize,
                axis=0,
            ),
            batch,
        )
        # apply grad
        (_, step_metric), step_grad = grad_fn(
            state.params, state.apply_fn, minibatch, keys[batch_idx]
        )

        return step_grad, step_metric

    def _scan_step(
        carry: Tuple[Pytree, Metrics],
        batch_idx: jax.Array | int,
    ) -> Tuple[Tuple[Pytree, Metrics], None]:

        step_grads, step_metrics = _minbatch_step(batch_idx)
        carry = jax.tree_util.tree_map(jnp.add, carry, (step_grads, step_metrics))

        return carry, None

    grads_shapes, metrics_shape = jax.eval_shape(_minbatch_step, 0)

    grads = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)

    metrics = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape
    )

    (grads, metrics), _ = jax.lax.scan(
        _scan_step, init=(grads, metrics), xs=jnp.arange(n_minbatch), length=n_minbatch
    )

    grads = jax.tree_util.tree_map(lambda g: g / n_minbatch, grads)

    return grads, metrics


def accum_grads(
    state: TrainState,
    batch: Batch,
    key,
    num_minibatches: int,
    loss_fn: Callable,
    use_scan: bool = True,
) -> Tuple[Pytree, Metrics]:

    if use_scan:

        return accum_grads_scan(
            state=state,
            batch=batch,
            key=key,
            n_minbatch=num_minibatches,
            loss_fn=loss_fn,
        )

    else:

        return accum_grads_loop(
            state=state,
            batch=batch,
            key=key,
            n_minbatch=num_minibatches,
            loss_fn=loss_fn,
        )

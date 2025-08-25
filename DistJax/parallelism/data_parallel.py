import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence , Tuple
from ml_collections import ConfigDict

from DistJax.core.training import PyTree , Parameter , TrainState , Metrics , Batch , accum_grads

def sync_grads(grads: PyTree, axis_names=Sequence[str]):
    def _sync_grads(g: Parameter):

        if isinstance(g, nn.Partitioned):
            replication_axis_name = [
                name
                for name in axis_names
                if name not in jax.tree_util.tree_leaves(g.names)
            ]

            if len(replication_axis_name) == 0:
                return g

            else:
                return g.replace(
                    value=jax.lax.pmean(g.value, axis_name=replication_axis_name)
                )
        else:
            return jax.lax.pmean(g, axis_name=axis_names)

    return jax.tree_util.tree_map(
        _sync_grads, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def train_step_dp(
    state: TrainState, metrics: Metrics, batch: Batch , loss_fn , CONFIG : ConfigDict
) -> Tuple[TrainState, Metrics]:

    rng, step_rng = jax.random.split(state.rng)

    grads, step_metrics = accum_grads(
        state, batch, step_rng, CONFIG.optimizer.num_minibatches, loss_fn=loss_fn
    )

    with jax.named_scope("sync_grads"):
        grads = jax.tree_util.tree_map(
            lambda g: jax.lax.pmean(g, axis_name=CONFIG.data_axis_name), grads
        )

    new_state = state.apply_gradients(grads=grads, rng=rng)

    with jax.named_scope("sync_metrics"):

        step_metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.psum(x, axis_name=CONFIG.data_axis_name), step_metrics
        )

    if metrics is None:
        metics = step_metrics

    else:
        metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)

    return new_state, metrics

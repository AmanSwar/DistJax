import jax
import flax.linen as nn

from typing import Sequence

from DistJax.core.training import PyTree , Parameter


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

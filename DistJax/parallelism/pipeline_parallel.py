import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from typing import Callable, Any , Tuple
import functools

from DistJax.core.training import PyTree , Parameter


def stack_params(
    params: PyTree,
    axis_name: str,
    mask_except: jax.Array | int | None,
    axis: int = 0,
) -> PyTree:
    """
    Stakcs sharded params along a givesn axis

    Args:
        params (PyTree): model parameters
        axis_name (str): name of the axis to stakc along
        mask_except (jax.Array | int | None): only the mask_except-th shard will be non zero
        axis (int, optional): index of the axis to stack along

    Returns:
        PyTree: Parameters 
    """

    def _stack(x):
        """
        Core stack logic - used on each leaf of a PyTree
        Args:
            x (jax.Array): input

        """

        if isinstance(x , nn.Partitioned):
            value , names = x.value , x.names

        else:
            value , names = x , (None ,) * x.ndim

        if mask_except is not None:

            axis_index = jax.lax.axis_index(axis_name)
            value = jnp.expand_dims(value , axis)

        value = jnp.expand_dims(value , axis)
        names = names[:axis] + (axis_name , ) + names[axis + 1 :]

        return nn.Partitioned(value , names=names)

    return tree_map(
        _stack,
        params,
        is_leaf= lambda x : isinstance(x , nn.Partitioned)
    )


def unstack_params(
        params : PyTree,
        axis_name : str
):
    """
    Unstack params along a given axis

    Args:
        params (PyTree) : params
        axis_name (str): axis along which to unstack
    """
    def _unstack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned) and axis_name in x.names:
            value = x.value
            names = x.names
            axis_idx = names.index(axis_name)
            value = value.squeeze(axis_idx)
            names = names[:axis_idx] + names[axis_idx + 1 :]
            if all([n is None for n in names]):
                return value
            else:
                return nn.Partitioned(value, names=names)
        else:
            return x

    return tree_map(
        _unstack,
        params,
        is_leaf= lambda x : isinstance(x , nn.Partitioned)

    )


def execute_pipeline_step(
    module: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    model_axis_name: str,
    **kwargs
) -> Tuple[jax.Array, jax.Array]:
    """
    Single micro batch pipeline step

    Args:
        module (nn.Module): stage to be executed
        state (jax.Array): output of last stage
        input (jax.Array): original input
        model_axis_name (str): name of modle exis in the mesh

    Returns:
        Tuple[jax.Array , jax.Array]: _description_
    """

    # total no. of stage = total axis name
    num_stages = jax.lax.psum(1, model_axis_name)
    # indexify the axis names
    stage_index = jax.lax.axis_index(model_axis_name)
    state = jnp.where(stage_index == 0, input, state)
    state = module(state, *args, **kwargs)

    output = jnp.where(stage_index == num_stages - 1, state, jnp.zeros_like(state))

    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i + 1) % num_stages) for i in range(num_stages)],
    )

    return (state, output)


class ModelParallelWrapper(nn.Module):

    model_axis_name: str
    module_fn: Callable[..., nn.Module]
    mask_except_model_idx: int | None = None
    split_rngs: bool = True
    module_kwargs: FrozenDict[str, Any] = FrozenDict({})

    @nn.compact
    def __call__(self, *args, **kwargs):

        if self.is_initializing() and self.split_rngs:
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(
                    self.scope.rngs["params"].rng, self.model_axis_name
                )
            )

            module = nn.map_variables(
                target=functools.partial(
                    self.module_fn,
                    name="shareded",
                    **self.module_kwargs,
                ),
                trans_in_fn=functools.partial(
                    unstack_params, axis_name=self.model_axis_name
                ),
                trans_out_fn=functools.partial(
                    stack_params,
                    axis_name=self.model_axis_name,
                    mask_except=self.mask_except_model_idx,
                ),
                mapped_collections="params",
                mutable=True,
            )()

            return module(*args, **kwargs)

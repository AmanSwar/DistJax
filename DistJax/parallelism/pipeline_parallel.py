import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from typing import Callable, Any
import functools


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

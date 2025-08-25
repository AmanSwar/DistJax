import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Callable , Any , Literal
from functools import partial

from DistJax.parallelism.pipeline_parallel import ModelParallelWrapper
from DistJax.core.utils import scale_init

class TPDense(nn.Module):

    dense_fn: Any
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    skip_communication: bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"

        dense_fn = partial(
            ModelParallelWrapper,
            model_axis_name=self.model_axis_name,
            module_fn=partial(
                self.dense_fn,
                kernel_init=scale_init(self.kernel_init, self.kernel_init_adjustment),
            ),
            name=self.dense_name,
        )

        if tp_mode == "none":
            x = self.dense_fn(kernel_init=self.kernel_init)(x)

        elif tp_mode == "gather":
            if not self.skip_communication:
                x = jax.lax.all_gather(x, self.model_axis_name, axis=-1, tiled=True)
            x = dense_fn()(x)

        elif tp_mode == "scatter":

            x = dense_fn()(x)

            if not self.skip_communication:
                x = jax.lax.psum_scatter(
                    x,
                    axis_name=self.model_axis_name,
                    scatter_dimension=x.ndim - 1,
                    tiled=True,
                )

        else:
            raise ValueError(f"Unknown Tensor Parallel model : {tp_mode}")

        return x

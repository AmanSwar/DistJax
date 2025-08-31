import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import optax

import functools

from DistJax.configs.default_config import get_default_config
from DistJax.parallelism.sharding import shard_module_params , init_fsdp 
from DistJax.parallelism.data_parallel import train_step_dp
from DistJax.models.simple_classifier import Classifier
from DistJax.core.training import Batch , PyTree 
from DistJax.core.utils import print_metrics , fold_rng_over_axis

from examples.utils import sim_multiCPU_dev

sim_multiCPU_dev()

CONFIG = get_default_config()
devices = np.array(jax.devices())
mesh = Mesh(devices, (CONFIG.data_axis_name,))


ShardedClassifier = shard_module_params(
    Classifier, axis_name=CONFIG.data_axis_name  # Pass the class itself
)

model_fsdp = ShardedClassifier(config=CONFIG.model)
optimizer = optax.adamw(learning_rate=CONFIG.optimizer.learning_rate)

rng = jax.random.PRNGKey(CONFIG.seed)
model_init_rng, data_rng = jax.random.split(rng)

data_inp_rng, data_label_rng = jax.random.split(data_rng)
batch = Batch(
    inputs=jax.random.normal(
        data_inp_rng, (CONFIG.data.batch_size, CONFIG.data.input_size)
    ),
    labels=jax.random.randint(
        data_label_rng, (CONFIG.data.batch_size,), 0, CONFIG.data.num_classes
    ),
)


def loss_fn(
    params: PyTree,
    apply_fn,
    batch: Batch,
    rng,
    config
):
    """Calculates loss and metrics for a single batch."""
    dropout_rng = fold_rng_over_axis(rng, config.data_axis_name)
    logits = apply_fn(
        {"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng}
    )
    loss_vector = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    bs = batch.inputs.shape[0]
    step_metrics = {
        "loss": (loss_vector.sum(), bs),
        "accuracy": (correct_pred.sum(), bs),
    }
    return loss_vector.mean(), step_metrics


init_fsdp_fn = jax.jit(
    shard_map(
        functools.partial(init_fsdp, model=model_fsdp, optimizer_fn=optimizer),
        mesh,
        in_specs=(
            P(),
            P(CONFIG.data_axis_name),
        ),  # we wanna shard the x (input) not the rng_key
        out_specs=P(),
        check_rep=False,
    )
)
train_step_fsdp_fn = jax.jit(
    shard_map(
        functools.partial(train_step_dp, loss_fn=loss_fn,CONFIG=CONFIG),
        mesh,
        in_specs=(P(), P(), P(CONFIG.data_axis_name)),  # we wanna shard batch only
        out_specs=(
            P(),
            P(),
        ),  # state and metric should be replicated across devices
        check_rep=False,
    ),
    donate_argnums=(0,1),
)

state_fsdp = init_fsdp_fn(model_init_rng, batch.inputs)
param_shape = jax.tree_util.tree_leaves(state_fsdp.params)[0].shape
_, metrics_shape = jax.eval_shape(train_step_fsdp_fn, state_fsdp, None, batch)
metrics_fsdp = jax.tree_util.tree_map(
    lambda x: jnp.zeros(x.shape, dtype=x.dtype), metrics_shape
)


print("\n--- Starting Training Loop ---")
for step in range(CONFIG.num_steps):
    # Reset metrics for each step to see per-step performance
    step_metrics = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), metrics_shape
    )
    state_fsdp, step_metrics = train_step_fsdp_fn(state_fsdp, step_metrics, batch)
    print(type(step_metrics))
    print_metrics(step_metrics, step + 1, title="FSDP")

print("\n🎉 Training finished successfully! 🎉")

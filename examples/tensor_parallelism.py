import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map
import flax.linen as nn

import optax

import functools


from utils import sim_multiCPU_dev

from DistJax.configs.tp_config import get_default_tp_classifier_config
from DistJax.models.tp_classifier import TPClassifier
from DistJax.core.training import Batch , TrainState
from DistJax.parallelism.tensor_parallel import init_tp , train_step
from DistJax.core.utils import print_metrics


sim_multiCPU_dev()

num_train_steps = 10
config = get_default_tp_classifier_config()

data_axis_size = len(jax.devices()) // config.model.model_axis_size
device_array = np.array(jax.devices()).reshape(
    data_axis_size, config.model.model_axis_size
)
mesh = Mesh(device_array, (config.data_axis_name, config.model_axis_name))
print(f"Created a {data_axis_size}x{config.model_axis_size} device mesh.")

model_tp = TPClassifier(config=config.model)
optimizer = optax.adamw(learning_rate=config.optimizer.learning_rate)

rng = jax.random.PRNGKey(config.seed)
model_init_rng, data_rng = jax.random.split(rng, 2)
data_inputs_rng, data_labels_rng = jax.random.split(data_rng, 2)
batch = Batch(
    inputs=jax.random.normal(
        data_inputs_rng, (config.data.batch_size, config.data.input_size)
    ),
    labels=jax.random.randint(
        data_labels_rng, (config.data.batch_size,), 0, config.data.num_classes
    ),
)



init_tp_fn_sharded = shard_map(
    functools.partial(init_tp, model=model_tp, optimizer=optimizer),
    mesh,
    in_specs=(P(), P(config.data_axis_name)),  
    out_specs=P(),  
    check_rep=False,
)

state_tp_shapes = jax.eval_shape(init_tp_fn_sharded, model_init_rng, batch.inputs)
state_tp_specs = nn.get_partition_spec(state_tp_shapes)

jit_init_fn = jax.jit(
    shard_map(
        functools.partial(init_tp, model=model_tp, optimizer=optimizer),
        mesh,
        in_specs=(P(), P(config.data_axis_name)),
        out_specs=state_tp_specs,
        check_rep=False,
    )
)
state_tp = jit_init_fn(model_init_rng, batch.inputs)

train_step_fn_sharded = jax.jit(
    shard_map(
        functools.partial(train_step, config=config),
        mesh,
        in_specs=(state_tp_specs, P(), P(config.data_axis_name)),
        out_specs=(state_tp_specs, P()),
        check_rep=False,
    ),
    donate_argnames=("state", "metrics"),
)

_, metric_shapes = jax.eval_shape(train_step_fn_sharded, state_tp, None, batch)
metrics_tp = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)

print("\n--- Starting Training ---")
for step in range(1, num_train_steps + 1):
    state_tp, metrics_tp = train_step_fn_sharded(state_tp, metrics_tp, batch)
    final_metrics = tree_map(lambda x: x[0], metrics_tp)
    print_metrics(final_metrics, step , title="Tensor parallism")
    metrics_tp = tree_map(jnp.zeros_like, metrics_tp)  # Reset metrics

print("\n--- Training Finished ---")
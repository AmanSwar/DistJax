import jax.numpy as jnp

from ml_collections import ConfigDict


def get_default_config():
    # configs
    DATA_CONFIG = ConfigDict(
        dict(
            batch_size=128,
            num_classes=10,
            input_size=784,
        )
    )

    MODEL_CONFIG = ConfigDict(
        dict(
            hidden_size=512,
            dropout_rate=0.1,
            dtype=jnp.bfloat16,
            num_classes=DATA_CONFIG.num_classes,
            data_axis_name="data",
        )
    )

    OPTIMIZER_CONFIG = ConfigDict(
        dict(
            learning_rate=1e-3,
            num_minibatches=4,
        )
    )

    CONFIG = ConfigDict(
        dict(
            model=MODEL_CONFIG,
            optimizer=OPTIMIZER_CONFIG,
            data=DATA_CONFIG,
            data_axis_name=MODEL_CONFIG.data_axis_name,
            seed=69,
            num_steps=10,
        )
    )

    return CONFIG

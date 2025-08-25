import jax.numpy as jnp

from ml_collections import ConfigDict



def get_default_tp_classifier_config():
    data_config = ConfigDict(
        dict(
            batch_size=128,
            num_classes=10,
            input_size=784,
        )
    )
    model_config = ConfigDict(
        dict(
            hidden_size=512,
            dropout_rate=0.1,
            mlp_expansion=1,
            num_layers=3,
            dtype=jnp.bfloat16,
            num_classes=data_config.num_classes,
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,
        )
    )
    optimizer_config = ConfigDict(
        dict(
            learning_rate=1e-3,
            num_minibatches=1,
        )
    )
    config = ConfigDict(
        dict(
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            data_axis_name=model_config.data_axis_name,
            model_axis_name=model_config.model_axis_name,
            model_axis_size=model_config.model_axis_size,
            seed=42,
        )
    )
    return config

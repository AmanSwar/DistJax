# DistJax: A Toolkit for Distributed Training in JAX

**DistJax** is a mini-library and collection of examples designed to simplify the implementation of common distributed training paradigms in JAX and Flax. While JAX provides powerful low-level primitives like `pmap` and `shard_map` for parallelism, orchestrating them into cohesive, large-scale training strategies can be complex. This repository provides high-level, reusable building blocks for data parallelism, tensor parallelism (including asynchronous variants), and pipeline parallelism, allowing researchers and engineers to scale their models with clarity and confidence.

## ✨ Core Features

This library provides modular components and end-to-end examples for the following paradigms, which can be mixed and matched to suit your specific hardware and model architecture.

### Data Parallelism (DP)
The foundational technique of replicating a model's weights across multiple devices and processing different shards of a data batch on each one. It's highly effective for scaling but requires each device to hold a full copy of the model, which can be a memory bottleneck.

### Tensor Parallelism (TP)
A model-parallel strategy that shards individual layers (like the weight matrices in Dense or Attention layers) across devices. This allows for training models that are too large to fit on a single device. This library includes:

- **Standard synchronous communication** using collective operations like `all_gather` and `psum_scatter`, which are easy to reason about but introduce explicit synchronization points.

- **Advanced asynchronous communication primitives** that leverage JAX's `ppermute` to overlap communication with computation. By passing activations between devices in a staggered, ring-like fashion, this approach can hide communication latency and significantly improve GPU/TPU utilization.

### Pipeline Parallelism (PP)
A model-parallel strategy that stages sequential model layers or blocks across different devices. To keep all devices active, the input batch is split into smaller "micro-batches" that are fed into the pipeline in a staggered manner. This minimizes the "pipeline bubble"—the idle time when devices are waiting for data—making it an efficient strategy for very deep models.

### Hybrid Approaches
The components are designed to be composable. For example, you can combine Data Parallelism and Tensor Parallelism: within a group of 8 GPUs, you might use 4-way tensor parallelism to shard a large model, and then replicate this 4-GPU setup twice for 2-way data parallelism. This allows for flexible scaling across both model size and data throughput.

## 📂 Library Structure

The repository is organized to separate reusable logic from specific model implementations and training scripts. This clean separation of concerns makes the library easy to navigate and extend.

```
DistJax/
├── core/               # Generic training utilities (TrainState, attention ops)
├── parallelism/        # The core parallelism primitives and modules
│   ├── data_parallel.py
│   ├── tensor_parallel.py
│   └── pipeline_parallel.py
├── models/             # Example model architectures built with the library
│   ├── simple_classifier.py
│   ├── tp_classifier.py
│   ├── transformer.py
│   └── pp_classifier.py
├── configs/            # Configuration files for the models
├──examples/           # Standalone scripts to run training for each paradigm
    ├── data_parallelism.py
    ├── tensor_parallelism.py
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

Follow these steps to set up the environment and run one of the examples.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DistJax.git
cd DistJax
```

### 2. Install Dependencies

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** For GPU or TPU support, ensure you have installed the appropriate version of JAX by following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### 3. Run an Example
The scripts in the `examples/` directory are designed to be run directly. They will simulate a multi-device environment on your CPU for demonstration purposes.

To run the data-parallel training example:

```bash
python -m examples.data_parallelism
```

This will run a few training steps and print the final metrics.


## 🛠️ Usage and Core Concepts

The library is built around a few key ideas to promote modularity and ease of use.

### Parallelism Primitives

The `DistJax.parallelism` module contains the core building blocks. These are often implemented as `flax.linen.Module` wrappers that inject parallelism logic into standard layers. For example, `tensor_parallel.py` provides `TPDense`, which looks like a normal `Dense` layer but automatically handles the sharding of its weights and the necessary communication (`all_gather` or `psum_scatter`) of its inputs and outputs across a device mesh.

### Model Implementation

The `DistJax.models` directory demonstrates the design philosophy: model architecture should be decoupled from parallelism logic. The models are constructed by composing the parallel primitives from the library. For instance, `transformer.py` builds a fully tensor-parallel Transformer block by using `TPAsyncDense` for its MLP and attention layers, without cluttering the model definition with low-level communication code.

### Orchestration

The `DistJax.examples` scripts tie everything together and provide a blueprint for your own training runs. They handle the essential boilerplate for distributed training:

- **Setting up the JAX Mesh**: A `Mesh` defines the logical topology of your devices (e.g., an 8-device array with a 'data' axis and a 'model' axis). This abstraction is crucial for telling JAX how to distribute data and computations.

- **Loading Model Configurations**: Using `ml_collections.ConfigDict` for clean and hierarchical management of hyperparameters.

- **Initializing the Model State**: Using `shard_map` to correctly initialize parameters across all devices according to the specified parallelism strategy. This ensures each device gets only its designated shard of the model.

- **Defining the Parallel train_step**: The core training function is written once and then parallelized using `shard_map`, with `PartitionSpec` annotations to define how the state, metrics, and data are sharded.

- **Running the Main Training Loop**: The loop executes the JIT-compiled parallel `train_step`, passing sharded data and updating the distributed model state.


## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, new features, or find any bugs, please feel free to open an issue or submit a pull request. Potential areas for future work include:

- Implementation of Fully Sharded Data Parallelism (FSDP)
- More model examples (e.g., Mixture-of-Experts, Vision Transformers)
- Support for more advanced optimizers tailored for distributed settings (e.g., ZeRO)
- Enhanced documentation with more in-depth tutorials and conceptual guides
- Integration with more advanced JAX features as they become available

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
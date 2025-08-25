DistJax: A Toolkit for Distributed Training
DistJax is a mini-library and collection of examples designed to simplify the implementation of common distributed training paradigms in JAX and Flax. This repository provides reusable building blocks for data parallelism, tensor parallelism (including asynchronous variants), and pipeline parallelism, allowing you to scale your models with clarity and confidence.

✨ Core Features
This library provides modular components and end-to-end examples for:

Data Parallelism (DP): The foundational technique of replicating a model and splitting data across multiple devices.

Tensor Parallelism (TP): A model-parallel strategy that shards individual layers (like Dense or Attention) across devices. This includes:
- Standard synchronous communication (all_gather, psum_scatter).

- Advanced asynchronous communication primitives (ppermute) to overlap computation and communication.

Pipeline Parallelism (PP): A model-parallel strategy that stages sequential model layers across devices, processing data in micro-batches to keep all devices active.

Hybrid Approaches: The components are designed to be composable, enabling hybrid strategies like combining Data and Tensor Parallelism (similar to Fully Sharded Data Parallelism).

📂 Library Structure
The repository is organized to separate reusable logic from specific model implementations and training scripts.

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
|examples/           # Standalone scripts to run training for each paradigm
    ├── train_dp.py
    ├── train_tp.py
    ├── train_transformer_tp.py
    └── train_pp.py
├── README.md
└── requirements.txt

🚀 Getting Started
Follow these steps to set up the environment and run one of the examples.

1. Clone the Repository
git clone https://github.com/AmanSwar/DistJax.git
cd DistJax

2. Install Dependencies
It's recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run an Example
Examples adding soon ....


🛠️ Usage and Core Concepts
The library is built around a few key ideas:

Parallelism Primitives
The DistJax.parallelism module contains the core building blocks. For example, tensor_parallel.py provides TPDense and TPAsyncDense, which are flax.linen modules that automatically handle the sharding and communication logic for a dense layer.

Model Implementation
The DistJax.models directory shows how to use these building blocks to create complex, parallel architectures. For example, transformer.py uses TPAsyncDense and other components to build a fully tensor-parallel Transformer block.

Orchestration
The DistJax.examples scripts tie everything together (Suppose to for now ...). They handle:

Setting up the JAX Mesh for device management.

Loading model configurations.

Initializing the model state across all devices using shard_map.

Defining the parallel train_step function and JIT-compiling it.

Running the main training loop.

You can adapt these scripts to build your own custom training runs.


🤝 Contributing
Contributions are welcome! If you have ideas for improvements, new features, or find any bugs, please feel free to open an issue or submit a pull request. Potential areas for future work include:

Implementation of Fully Sharded Data Parallelism (FSDP).

More model examples (e.g., Mixture-of-Experts).

Integration with more advanced JAX features.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
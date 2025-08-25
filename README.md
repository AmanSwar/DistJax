# DistJax: A Toolkit for Distributed Training

DistJax is a mini-library and set of examples that simplify common distributed training paradigms in **JAX** and **Flax**. It provides reusable building blocks for **data parallelism**, **tensor parallelism** (including asynchronous variants), and **pipeline parallelism**, so you can scale models clearly and confidently.

---

## Core Features

- **Data Parallelism (DP):** Replicate a model and split data across devices.
- **Tensor Parallelism (TP):** Shard layers (e.g., Dense, Attention) across devices.
  - Synchronous communication: `all_gather`, `psum_scatter`
  - Asynchronous primitives: `ppermute` to overlap compute and communication
- **Pipeline Parallelism (PP):** Stage sequential layers across devices; use micro-batches to keep all devices utilized.
- **Hybrid Approaches:** Compose components to combine strategies (e.g., DP + TP; similar in spirit to Fully Sharded Data Parallelism).

---

## Library Structure

The repository separates reusable logic from model implementations and training scripts.

```text
DistJax/
├── core/               # Generic training utilities (TrainState, attention ops)
├── parallelism/        # Parallelism primitives and modules
│   ├── data_parallel.py
│   ├── tensor_parallel.py
│   └── pipeline_parallel.py
├── models/             # Example model architectures built with the library
│   ├── simple_classifier.py
│   ├── tp_classifier.py
│   ├── transformer.py
│   └── pp_classifier.py
├── configs/            # Configuration files for the models
├── examples/           # Standalone scripts for each paradigm
│   ├── train_dp.py
│   ├── train_tp.py
│   ├── train_transformer_tp.py
│   └── train_pp.py
├── README.md
└── requirements.txt
```

## Getting Started
1) Clone the repository
```
git clone https://github.com/AmanSwar/DistJax.git
cd DistJax
```
2) Install dependencies

It’s recommended to use a virtual environment.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Run an example
Examples coming soon.

## Usage and Core Concepts
#### Parallelism Primitives

Core building blocks live in parallelism/.
For example, tensor_parallel.py provides TPDense and TPAsyncDense (flax.linen modules) that handle sharding and communication for dense layers.

#### Model Implementation

See models/ for how to assemble components into full architectures.
For example, transformer.py uses TPAsyncDense and related parts to build a tensor-parallel Transformer block.

#### Orchestration
Example scripts in examples/ (work-in-progress) handle:

Setting up a JAX mesh for device management

Loading model configurations

Initializing model state across devices (e.g., via shard_map)

Defining and JIT-compiling a parallel train_step

Running the main training loop

You can adapt these scripts for custom training runs.

## Contributing

Contributions are welcome—issues and pull requests are appreciated.

Potential areas for future work:

Implement Fully Sharded Data Parallelism (FSDP)

Add more model examples (e.g., Mixture-of-Experts)

Integrate additional advanced JAX features
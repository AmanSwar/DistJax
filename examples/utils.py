import os

def sim_multiCPU_dev(num_devices=8):
    """Simulates multiple CPU devices for parallelism demonstrations."""
    os.environ.update(
        {
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={num_devices}",
            "JAX_PLATFORMS": "cpu",
        }
    )
    print(f"Simulated {num_devices} CPU devices.")

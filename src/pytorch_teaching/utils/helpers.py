"""Helper utilities for PyTorch Teaching lessons."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for computation.

    Args:
        prefer_cuda: If True, prefer CUDA over MPS (default: True)

    Returns:
        torch.device: Best available device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_size(model: torch.nn.Module) -> tuple[int, float]:
    """
    Calculate model size in parameters and memory.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, size_in_mb)
    """
    param_size = 0
    param_sum = 0

    for param in model.parameters():
        param_sum += param.nemel()
        param_size += param.nemel() * param.element_size()

    buffer_size = 0
    buffer_sum = 0

    for buffer in model.buffers():
        buffer_sum += buffer.nemel()
        buffer_size += buffer.nemel() * buffer.element_size()

    total_params = param_sum + buffer_sum
    size_mb = (param_size + buffer_size) / 1024**2

    return total_params, size_mb


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
    """
    total_params, size_mb = get_model_size(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nModel Summary:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"{'='*50}\n")


def check_cuda_memory() -> Optional[dict]:
    """
    Check CUDA memory usage.

    Returns:
        Dictionary with memory info if CUDA available, None otherwise
    """
    if not torch.cuda.is_available():
        return None

    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_operation(func, *args, num_runs: int = 100, warmup: int = 10, **kwargs) -> float:
    """
    Benchmark a PyTorch operation.

    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        num_runs: Number of runs for benchmarking
        warmup: Number of warmup runs
        **kwargs: Keyword arguments for func

    Returns:
        Average time in milliseconds
    """
    import time

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    return (end - start) / num_runs * 1000  # Convert to ms


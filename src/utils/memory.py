"""
Memory Management and Profiling Utilities

This module provides utilities for monitoring and optimizing GPU memory usage
on RTX 4070 (8GB VRAM).
"""

import torch
import gc
from typing import Optional, Dict, Any
import psutil
import GPUtil
from contextlib import contextmanager
from functools import wraps
import time


class MemoryMonitor:
    """Monitor GPU and CPU memory usage."""

    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dict with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        gpu_id = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
        max_memory_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9

        return {
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
            "max_allocated_gb": max_memory_allocated,
            "free_gb": 8.0 - memory_reserved,  # RTX 4070 has 8GB
        }

    @staticmethod
    def get_cpu_memory() -> Dict[str, float]:
        """
        Get current CPU memory usage.

        Returns:
            Dict with CPU memory statistics in GB
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_gb": memory_info.rss / 1e9,  # Resident Set Size
            "vms_gb": memory_info.vms / 1e9,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

    @staticmethod
    def print_memory_stats(prefix: str = ""):
        """Print formatted memory statistics."""
        if prefix:
            print(f"\n{'='*60}")
            print(f"Memory Stats: {prefix}")
            print(f"{'='*60}")

        # GPU Memory
        if torch.cuda.is_available():
            gpu_mem = MemoryMonitor.get_gpu_memory()
            print(f"GPU Memory:")
            print(f"  Allocated: {gpu_mem['allocated_gb']:.2f} GB")
            print(f"  Reserved:  {gpu_mem['reserved_gb']:.2f} GB")
            print(f"  Free:      {gpu_mem['free_gb']:.2f} GB")
            print(f"  Max Used:  {gpu_mem['max_allocated_gb']:.2f} GB")

        # CPU Memory
        cpu_mem = MemoryMonitor.get_cpu_memory()
        print(f"CPU Memory:")
        print(f"  RSS:       {cpu_mem['rss_gb']:.2f} GB")
        print(f"  Percent:   {cpu_mem['percent']:.1f}%")

        if prefix:
            print(f"{'='*60}\n")

    @staticmethod
    def reset_peak_stats():
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()


class MemoryOptimizer:
    """Utilities for optimizing memory usage."""

    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def optimize_model_memory(model: torch.nn.Module, enable_checkpointing: bool = True) -> torch.nn.Module:
        """
        Apply memory optimizations to a model.

        Args:
            model: PyTorch model
            enable_checkpointing: Enable gradient checkpointing

        Returns:
            Optimized model
        """
        # Enable gradient checkpointing if available
        if enable_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        # Set model to eval mode to disable dropout (saves memory during inference)
        # (Will be set back to train mode when training)

        return model

    @staticmethod
    def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
        """
        Calculate model memory footprint.

        Args:
            model: PyTorch model

        Returns:
            Dict with memory statistics
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            "parameters_mb": param_size / 1e6,
            "buffers_mb": buffer_size / 1e6,
            "total_mb": total_size / 1e6,
            "total_gb": total_size / 1e9,
        }

    @staticmethod
    def print_model_size(model: torch.nn.Module, model_name: str = "Model"):
        """Print model size information."""
        footprint = MemoryOptimizer.get_model_memory_footprint(model)
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{model_name} Size:")
        print(f"  Total parameters:     {num_params:,}")
        print(f"  Trainable parameters: {num_trainable:,}")
        print(f"  Memory footprint:     {footprint['total_mb']:.2f} MB ({footprint['total_gb']:.3f} GB)")


@contextmanager
def track_memory(operation_name: str = "Operation", verbose: bool = True):
    """
    Context manager to track memory usage of an operation.

    Args:
        operation_name: Name of the operation being tracked
        verbose: Print detailed statistics

    Usage:
        with track_memory("Model Loading"):
            model = load_model()
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Starting: {operation_name}")
        print(f"{'─'*60}")

    # Record initial state
    MemoryMonitor.reset_peak_stats()
    start_time = time.time()

    initial_gpu = MemoryMonitor.get_gpu_memory() if torch.cuda.is_available() else None
    initial_cpu = MemoryMonitor.get_cpu_memory()

    yield

    # Record final state
    end_time = time.time()
    final_gpu = MemoryMonitor.get_gpu_memory() if torch.cuda.is_available() else None
    final_cpu = MemoryMonitor.get_cpu_memory()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Completed: {operation_name}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print(f"{'─'*60}")

        if final_gpu:
            delta_gpu = final_gpu['allocated_gb'] - initial_gpu['allocated_gb']
            print(f"GPU Memory Change: {delta_gpu:+.2f} GB")
            print(f"Peak GPU Memory: {final_gpu['max_allocated_gb']:.2f} GB")

        delta_cpu = final_cpu['rss_gb'] - initial_cpu['rss_gb']
        print(f"CPU Memory Change: {delta_cpu:+.2f} GB")
        print(f"{'─'*60}\n")


def memory_efficient_inference(func):
    """
    Decorator for memory-efficient inference.

    Disables gradient computation and clears cache after execution.

    Usage:
        @memory_efficient_inference
        def predict(model, data):
            return model(data)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)

        # Clear cache after inference
        MemoryOptimizer.clear_cache()

        return result

    return wrapper


def check_memory_requirements(required_gb: float = 6.0) -> bool:
    """
    Check if sufficient GPU memory is available.

    Args:
        required_gb: Required memory in GB

    Returns:
        True if sufficient memory available
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available")
        return False

    gpu_mem = MemoryMonitor.get_gpu_memory()
    available = gpu_mem['free_gb']

    if available < required_gb:
        print(f"⚠️  Insufficient GPU memory!")
        print(f"   Required: {required_gb:.2f} GB")
        print(f"   Available: {available:.2f} GB")
        print(f"   Shortage: {required_gb - available:.2f} GB")
        print(f"\n   Recommendations:")
        print(f"   - Reduce batch size")
        print(f"   - Enable 8-bit quantization")
        print(f"   - Reduce LoRA rank")
        print(f"   - Enable gradient checkpointing")
        return False

    print(f"✓ Sufficient GPU memory available: {available:.2f} GB")
    return True


if __name__ == "__main__":
    # Test memory monitoring
    print("Testing Memory Monitoring Utilities")
    print("="*60)

    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Print memory stats
    MemoryMonitor.print_memory_stats("Initial State")

    # Test memory tracking
    with track_memory("Allocating 1GB tensor"):
        large_tensor = torch.randn(1024, 1024, 256).cuda() if torch.cuda.is_available() else None

    # Check memory requirements
    check_memory_requirements(required_gb=6.0)

    # Clean up
    if torch.cuda.is_available():
        del large_tensor
        MemoryOptimizer.clear_cache()

    MemoryMonitor.print_memory_stats("After Cleanup")

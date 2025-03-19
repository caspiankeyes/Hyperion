# hyperion/utils/hardware.py

"""
Hardware monitoring and optimization utilities for Hyperion.

This module provides tools for determining available resources, optimizing
utilization, and monitoring hardware usage during fine-tuning.
"""

import os
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import math

import torch
import psutil

from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


def detect_available_gpus() -> List[int]:
    """
    Detect available CUDA GPUs.
    
    Returns:
        List of available GPU indices
    """
    if not torch.cuda.is_available():
        return []
        
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        # Parse CUDA_VISIBLE_DEVICES
        try:
            gpu_ids = [int(x.strip()) for x in visible_devices.split(",") if x.strip()]
            return gpu_ids
        except ValueError:
            logger.warning(f"Could not parse CUDA_VISIBLE_DEVICES: {visible_devices}")
    
    # Default: all available GPUs
    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Get memory information for each GPU.
    
    Returns:
        Dictionary mapping GPU index to memory information
    """
    if not torch.cuda.is_available():
        return {}
        
    gpu_info = {}
    
    for gpu_id in range(torch.cuda.device_count()):
        try:
            # Get memory info in bytes
            mem_info = torch.cuda.get_device_properties(gpu_id).total_memory
            mem_allocated = torch.cuda.memory_allocated(gpu_id)
            mem_reserved = torch.cuda.memory_reserved(gpu_id)
            mem_free = mem_info - mem_allocated
            
            # Convert to GB
            gb = 1024 ** 3
            gpu_info[gpu_id] = {
                "total_memory_gb": mem_info / gb,
                "allocated_memory_gb": mem_allocated / gb,
                "reserved_memory_gb": mem_reserved / gb,
                "free_memory_gb": mem_free / gb
            }
        except Exception as e:
            logger.warning(f"Error getting memory info for GPU {gpu_id}: {e}")
            
    return gpu_info


def get_system_info() -> Dict[str, Any]:
    """
    Get system hardware information.
    
    Returns:
        Dictionary with system information
    """
    # Basic system info
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
    }
    
    # Memory info
    mem = psutil.virtual_memory()
    system_info["memory"] = {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_percent": mem.percent
    }
    
    # Disk info
    disk = psutil.disk_usage("/")
    system_info["disk"] = {
        "total_gb": disk.total / (1024 ** 3),
        "used_gb": disk.used / (1024 ** 3),
        "free_gb": disk.free / (1024 ** 3),
        "used_percent": disk.percent
    }
    
    # GPU info
    if torch.cuda.is_available():
        system_info["gpu_count"] = torch.cuda.device_count()
        system_info["cuda_version"] = torch.version.cuda
        system_info["gpu_info"] = {}
        
        for gpu_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_id)
            system_info["gpu_info"][gpu_id] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024 ** 3)
            }
    else:
        system_info["gpu_count"] = 0
        
    return system_info


def estimate_optimal_batch_size(
    model_size_params: int, 
    gpu_vram_gb: float,
    seq_length: int, 
    mixed_precision: str = "bf16", 
    activation_factor: float = 1.2
) -> int:
    """
    Estimate optimal batch size for training based on VRAM and model size.
    
    Args:
        model_size_params: Number of parameters in the model
        gpu_vram_gb: Available GPU VRAM in GB
        seq_length: Maximum sequence length
        mixed_precision: Precision mode (no, fp16, bf16)
        activation_factor: Factor for activation memory overhead
        
    Returns:
        Estimated optimal batch size
    """
    # Calculate parameter memory in bytes
    bytes_per_param = 2 if mixed_precision in ["fp16", "bf16"] else 4
    model_memory_bytes = model_size_params * bytes_per_param
    
    # Calculate per-token memory requirements
    token_memory_factor = 2.5  # Higher for transformers due to KV cache
    token_memory_bytes = model_size_params * token_memory_factor / 1000  # Per token
    
    # Calculate batch memory requirements
    activation_memory = token_memory_bytes * seq_length * activation_factor
    
    # Calculate optimizer memory
    optimizer_memory_bytes = model_size_params * 8  # Adam uses 8 bytes per parameter
    
    # Total memory per batch sample
    memory_per_sample = activation_memory
    
    # Available memory for batch
    gpu_vram_bytes = gpu_vram_gb * (1024 ** 3)
    available_memory = gpu_vram_bytes - model_memory_bytes - optimizer_memory_bytes
    
    # Safety factor
    safety_factor = 0.9
    available_memory *= safety_factor
    
    # Calculate batch size
    batch_size = max(1, int(available_memory / memory_per_sample))
    
    logger.info(f"Estimated optimal batch size: {batch_size} (with safety factor {safety_factor})")
    logger.info(f"Model memory: {model_memory_bytes / (1024**3):.2f} GB, "
                f"Optimizer: {optimizer_memory_bytes / (1024**3):.2f} GB")
    
    return batch_size


# hyperion/utils/hardware.py (continued)

def get_optimal_training_parameters(
    model_size_params: int,
    seq_length: int,
    total_samples: int,
    gpu_info: Optional[Dict[int, Dict[str, float]]] = None,
    target_epochs: float = 3.0,
    mixed_precision: str = "bf16",
    min_batch_size: int = 1
) -> Dict[str, Any]:
    """
    Calculate optimal training parameters based on hardware constraints.
    
    Args:
        model_size_params: Number of parameters in the model
        seq_length: Maximum sequence length
        total_samples: Total number of training samples
        gpu_info: GPU memory information (if None, will be detected)
        target_epochs: Target number of epochs
        mixed_precision: Precision mode (no, fp16, bf16)
        min_batch_size: Minimum acceptable batch size
        
    Returns:
        Dictionary with optimal training parameters
    """
    # Get GPU info if not provided
    if gpu_info is None:
        gpu_info = get_gpu_memory_info()
        
    if not gpu_info:
        logger.warning("No GPU information available, using conservative estimates")
        gpu_vram_gb = 8.0  # Conservative estimate
    else:
        # Use the GPU with most available memory
        gpu_id = max(gpu_info.keys(), key=lambda k: gpu_info[k].get("free_memory_gb", 0))
        gpu_vram_gb = gpu_info[gpu_id].get("free_memory_gb", 8.0)
        
    # Estimate batch size
    batch_size = estimate_optimal_batch_size(
        model_size_params=model_size_params,
        gpu_vram_gb=gpu_vram_gb,
        seq_length=seq_length,
        mixed_precision=mixed_precision
    )
    
    # Ensure minimum batch size
    batch_size = max(batch_size, min_batch_size)
    
    # Calculate gradient accumulation steps to achieve effective batch size
    target_effective_batch_size = 16  # Common baseline
    gradient_accumulation_steps = max(1, round(target_effective_batch_size / batch_size))
    
    # Calculate number of training steps
    steps_per_epoch = math.ceil(total_samples / (batch_size * gradient_accumulation_steps))
    total_steps = int(steps_per_epoch * target_epochs)
    
    # Calculate learning rate schedule
    warmup_ratio = 0.03
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Return optimized parameters
    return {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "steps_per_epoch": steps_per_epoch,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": True if model_size_params > 1e9 else False
    }


def set_optimal_torch_settings():
    """Configure PyTorch with optimal settings for training."""
    # Enable TF32 for faster computation on Ampere GPUs
    if torch.cuda.is_available():
        # Check if running on Ampere or newer
        for i in range(torch.cuda.device_count()):
            device_cap = torch.cuda.get_device_capability(i)
            if device_cap[0] >= 8:  # Ampere or newer
                logger.info("Enabling TF32 precision for faster computation")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    # Set default dtype
    torch.set_default_dtype(torch.float32)
    
    # Enable cuDNN benchmark mode for optimized performance
    if torch.backends.cudnn.is_available():
        logger.info("Enabling cuDNN benchmark mode")
        torch.backends.cudnn.benchmark = True
        

def optimize_cpu_settings(num_workers: Optional[int] = None):
    """
    Optimize CPU settings for data loading and processing.
    
    Args:
        num_workers: Number of data loader workers (if None, will be auto-detected)
    """
    if num_workers is None:
        # Use number of physical cores as default
        num_workers = max(1, psutil.cpu_count(logical=False) - 1)
        
    # Set environment variables for better performance
    os.environ["OMP_NUM_THREADS"] = str(num_workers)
    os.environ["MKL_NUM_THREADS"] = str(num_workers)
    
    # Set torch settings
    torch.set_num_threads(num_workers)
    
    logger.info(f"CPU optimization: {num_workers} workers configured")
    return num_workers


class ResourceMonitor:
    """
    Monitor and record resource usage during training.
    
    Tracks GPU memory, CPU usage, and other resource metrics.
    """
    
    def __init__(self, log_interval_seconds: int = 60):
        """
        Initialize resource monitor.
        
        Args:
            log_interval_seconds: Interval between monitoring logs in seconds
        """
        self.log_interval_seconds = log_interval_seconds
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.metrics_history = {
            "timestamp": [],
            "gpu_memory_used": [],
            "cpu_percent": [],
            "memory_percent": []
        }
        
    def start_monitoring(self):
        """Start resource monitoring in a background thread."""
        if self.monitoring_thread is not None:
            logger.warning("Monitoring already active")
            return
            
        import threading
        import time
        
        def monitoring_loop():
            while not self.stop_monitoring:
                try:
                    # Record current timestamp
                    from datetime import datetime
                    self.metrics_history["timestamp"].append(datetime.now())
                    
                    # Record GPU memory
                    if torch.cuda.is_available():
                        gpu_mem = 0
                        for i in range(torch.cuda.device_count()):
                            gpu_mem += torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                        self.metrics_history["gpu_memory_used"].append(gpu_mem)
                    else:
                        self.metrics_history["gpu_memory_used"].append(0)
                    
                    # Record CPU usage
                    self.metrics_history["cpu_percent"].append(psutil.cpu_percent())
                    
                    # Record memory usage
                    self.metrics_history["memory_percent"].append(psutil.virtual_memory().percent)
                    
                    # Log current status
                    if len(self.metrics_history["timestamp"]) % 10 == 0:
                        self._log_current_status()
                        
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    
                # Sleep for the specified interval
                time.sleep(self.log_interval_seconds)
                
        # Start monitoring thread
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Resource monitoring started (interval: {self.log_interval_seconds}s)")
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.monitoring_thread is None:
            return
            
        self.stop_monitoring = True
        self.monitoring_thread.join(timeout=5)
        self.monitoring_thread = None
        logger.info("Resource monitoring stopped")
        
    def _log_current_status(self):
        """Log current resource status."""
        if not self.metrics_history["timestamp"]:
            return
            
        idx = -1  # Last recorded metric
        timestamp = self.metrics_history["timestamp"][idx]
        gpu_mem = self.metrics_history["gpu_memory_used"][idx]
        cpu_percent = self.metrics_history["cpu_percent"][idx]
        memory_percent = self.metrics_history["memory_percent"][idx]
        
        logger.info(f"Resources: GPU memory {gpu_mem:.1f} GB, "
                    f"CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        
    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak resource usage during monitoring.
        
        Returns:
            Dictionary with peak resource metrics
        """
        if not self.metrics_history["timestamp"]:
            return {
                "peak_gpu_memory_gb": 0,
                "peak_cpu_percent": 0,
                "peak_memory_percent": 0
            }
            
        return {
            "peak_gpu_memory_gb": max(self.metrics_history["gpu_memory_used"]),
            "peak_cpu_percent": max(self.metrics_history["cpu_percent"]),
            "peak_memory_percent": max(self.metrics_history["memory_percent"])
        }
        
    def get_metrics_history(self) -> Dict[str, List]:
        """
        Get complete metrics history.
        
        Returns:
            Dictionary with metrics history
        """
        return self.metrics_history
        
    def plot_resource_usage(self, output_path: Optional[str] = None):
        """
        Generate a plot of resource usage over time.
        
        Args:
            output_path: Path to save the plot image (if None, will display)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
        except ImportError:
            logger.error("Matplotlib not installed. Run `pip install matplotlib` to enable plotting.")
            return
            
        if not self.metrics_history["timestamp"]:
            logger.warning("No metrics recorded yet")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot GPU memory
        timestamps = self.metrics_history["timestamp"]
        ax1.plot(timestamps, self.metrics_history["gpu_memory_used"], 'r-', label='GPU Memory')
        ax1.set_ylabel('GPU Memory (GB)')
        ax1.set_title('Resource Usage Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot CPU and system memory
        ax2.plot(timestamps, self.metrics_history["cpu_percent"], 'b-', label='CPU Usage')
        ax2.plot(timestamps, self.metrics_history["memory_percent"], 'g-', label='System Memory')
        ax2.set_ylabel('Usage %')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Resource usage plot saved to {output_path}")
        else:
            plt.show()
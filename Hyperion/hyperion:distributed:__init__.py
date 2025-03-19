# hyperion/distributed/__init__.py

"""
Distributed training orchestration for multi-node setups.

This module provides utilities for efficient multi-node, multi-GPU training with
optimized communication patterns and memory management.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

import torch
import torch.distributed as dist
from transformers import Trainer

from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    world_size: int
    """Total number of GPUs across all nodes."""
    
    nodes: int = 1
    """Number of physical nodes/machines."""
    
    backend: str = "nccl"
    """Communication backend (nccl, gloo, etc.)."""
    
    mixed_precision: str = "bf16"
    """Mixed precision mode (no, fp16, bf16)."""
    
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing."""
    
    zero_stage: int = 3
    """DeepSpeed ZeRO optimization stage (1, 2, or 3)."""
    
    offload_optimizer: bool = False
    """Whether to offload optimizer states to CPU."""
    
    offload_param: bool = False
    """Whether to offload parameters to CPU."""
    
    memory_efficient_attention: bool = True
    """Whether to use memory-efficient attention implementation."""
    
    gradient_accumulation_steps: int = 1
    """Number of steps for gradient accumulation."""
    
    communication_buffer_size: int = 0  # 0 means auto-detect
    """Custom buffer size for communication (in MB)."""
    
    optimizer_overlap: bool = True
    """Whether to overlap communication and computation."""
    
    custom_cuda_kernel: bool = True
    """Whether to use optimized custom CUDA kernels."""
    
    checkpoint_interval_min: int = 30
    """Frequency of checkpointing in minutes."""


class DistributedManager:
    """
    Manager for distributed training orchestration.
    
    Handles initialization, optimization, and communication for distributed training
    across multiple nodes and GPUs.
    """
    
    def __init__(self, world_size: int = 1, nodes: int = 1, **kwargs):
        """
        Initialize distributed training manager.
        
        Args:
            world_size: Total number of GPUs
            nodes: Number of physical machines
            **kwargs: Additional distributed configuration parameters
        """
        self.config = DistributedConfig(
            world_size=world_size,
            nodes=nodes,
            **{k: v for k, v in kwargs.items() if k in DistributedConfig.__dataclass_fields__}
        )
        
        # Initialize internal state
        self._initialized = False
        self._rank = 0
        self._local_rank = 0
        self._node_rank = 0
        
        # Auto-detect environment variables if in a distributed environment
        if os.environ.get("WORLD_SIZE") and os.environ.get("RANK"):
            self._rank = int(os.environ.get("RANK", "0"))
            self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self._node_rank = self._rank // (self.config.world_size // self.config.nodes)
            
            # Override world_size if environment variable is set
            if os.environ.get("WORLD_SIZE"):
                self.config.world_size = int(os.environ.get("WORLD_SIZE"))
    
    def initialize(self):
        """Initialize the distributed process group."""
        if self._initialized:
            return
        
        if self.config.world_size <= 1:
            logger.info("Running in non-distributed mode (world_size=1)")
            self._initialized = True
            return
        
        # Initialize the process group
        logger.info(f"Initializing distributed training with {self.config.world_size} GPUs across {self.config.nodes} nodes")
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.config.backend,
                init_method=os.environ.get("MASTER_ADDR", "env://"),
                world_size=self.config.world_size,
                rank=self._rank
            )
            
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            
        # Log initialization status
        logger.info(f"Distributed initialization complete: rank={self._rank}, local_rank={self._local_rank}, node={self._node_rank}")
        
        self._initialized = True
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """
        Generate DeepSpeed configuration based on distributed settings.
        
        Returns:
            DeepSpeed configuration dictionary
        """
        # Base configuration for ZeRO
        config = {
            "zero_optimization": {
                "stage": self.config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.config.offload_optimizer else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.config.offload_param else "none"
                },
                "overlap_comm": self.config.optimizer_overlap,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "sub_group_size": 1e9,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False
        }
        
        # Add FP16/BF16 settings
        if self.config.mixed_precision == "fp16":
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        elif self.config.mixed_precision == "bf16":
            config["bf16"] = {
                "enabled": True
            }
            
        # Add communication buffer size if specified
        if self.config.communication_buffer_size > 0:
            config["communication_data_type"] = "bfloat16" if self.config.mixed_precision == "bf16" else "fp16"
            config["prescale_gradients"] = True
            config["communication_buffer_size"] = self.config.communication_buffer_size * 1024 * 1024  # Convert MB to bytes
            
        # Enable memory efficient attention if requested
        if self.config.memory_efficient_attention:
            config["memory_efficient_attention"] = {
                "enabled": True,
                "algorithm": "flash"
            }
            
        # Add custom CUDA kernels if enabled
        if self.config.custom_cuda_kernel:
            config["custom_cuda_kernel"] = True
            
        # Add checkpoint config
        if self.config.checkpoint_interval_min > 0:
            config["checkpoint"] = {
                "tag_validation": "Hyperion",
                "checkpoint_interval": self.config.checkpoint_interval_min * 60,  # Convert to seconds
                "keep_n_latest_checkpoints": 5
            }
            
        return config
    
    def wrap_trainer(self, trainer: Trainer) -> Trainer:
        """
        Wrap a Trainer instance with distributed capabilities.
        
        Args:
            trainer: Standard HuggingFace Trainer instance
            
        Returns:
            Trainer with distributed capabilities
        """
        # Initialize distributed environment if not already done
        self.initialize()
        
        # Get deepspeed config
        ds_config = self.get_deepspeed_config()
        
        # Add deepspeed configuration to trainer args
        if not hasattr(trainer, "args") or trainer.args is None:
            raise ValueError("Trainer must have args configured before wrapping for distributed training")
            
        # Convert DeepSpeed config to JSON string
        ds_config_json = json.dumps(ds_config)
        
        # Create temporary file for deepspeed config
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(ds_config_json)
            ds_config_path = f.name
            
        # Update trainer args with distributed configuration
        trainer.args.deepspeed = ds_config_path
        trainer.args.local_rank = self._local_rank
        
        # Configure DDP settings
        if self.config.world_size > 1:
            trainer.args.ddp_find_unused_parameters = False
            trainer.args.ddp_bucket_cap_mb = 25
            
        # Set gradient checkpointing
        if self.config.gradient_checkpointing and hasattr(trainer.model, "gradient_checkpointing_enable"):
            trainer.model.gradient_checkpointing_enable()
            
        # Return the updated trainer
        return trainer
        
    def barrier(self):
        """Synchronize all processes."""
        if self._initialized and self.config.world_size > 1:
            torch.distributed.barrier()
            
    def cleanup(self):
        """Clean up distributed environment."""
        if self._initialized and self.config.world_size > 1:
            torch.distributed.destroy_process_group()
            self._initialized = False
            
    @property
    def rank(self) -> int:
        """Global rank of current process."""
        return self._rank
        
    @property
    def local_rank(self) -> int:
        """Local rank within the current node."""
        return self._local_rank
        
    @property
    def node_rank(self) -> int:
        """Rank of the current node."""
        return self._node_rank
        
    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return self.config.world_size
        
    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0)."""
        return self._rank == 0


# Export strategies
from hyperion.distributed.tensor_parallel import enable_tensor_parallelism
from hyperion.distributed.pipeline_parallel import enable_pipeline_parallelism

__all__ = [
    "DistributedManager",
    "DistributedConfig",
    "enable_tensor_parallelism",
    "enable_pipeline_parallelism"
]
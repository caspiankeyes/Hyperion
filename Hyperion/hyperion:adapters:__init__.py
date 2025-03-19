# hyperion/adapters/__init__.py

"""
Parameter-efficient fine-tuning adapters.

This module provides adapter configurations and utilities for parameter-efficient
fine-tuning methods like LoRA, QLoRA, and IA³.
"""

from typing import Dict, Union
from peft import LoraConfig as PeftLoraConfig
from peft import AdaLoraConfig, IA3Config, PrefixTuningConfig
from hyperion.config import LoraConfig


def get_adapter_config(config: Union[LoraConfig, Dict]):
    """
    Convert Hyperion adapter config to PEFT library config.
    
    Args:
        config: Hyperion adapter configuration
        
    Returns:
        PEFT library configuration object
    """
    if isinstance(config, dict):
        config_type = config.pop("type", "lora")
        if config_type == "lora":
            config = LoraConfig(**config)
        else:
            raise ValueError(f"Unsupported adapter type: {config_type}")
    
    if isinstance(config, LoraConfig):
        return PeftLoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type,
            fan_in_fan_out=config.fan_in_fan_out
        )
    
    raise ValueError(f"Unsupported adapter configuration type: {type(config)}")


# Expose specific adapter implementations
from hyperion.adapters.lora import LoraAdapter
from hyperion.adapters.ia3 import IA3Adapter

__all__ = ["get_adapter_config", "LoraAdapter", "IA3Adapter"]
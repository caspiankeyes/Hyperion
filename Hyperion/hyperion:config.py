# hyperion/config.py

"""
Configuration classes for Hyperion fine-tuning.

These classes provide a structured interface for defining fine-tuning parameters,
adapter configurations, and quantization settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os
import json


@dataclass
class LoraConfig:
    """
    Configuration for Low-Rank Adaptation (LoRA) parameters.
    
    LoRA reduces training parameters by factorizing weight updates into low-rank matrices.
    """
    
    r: int = 8
    """Rank of the update matrices, determines the compression ratio."""
    
    lora_alpha: int = 16
    """Scaling factor for the update matrices, affects learning dynamics."""
    
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    """The modules to apply LoRA to, typically attention components."""
    
    lora_dropout: float = 0.05
    """Dropout rate applied to LoRA layers for regularization."""
    
    bias: str = "none"
    """Bias configuration; options are 'none', 'all', or 'lora_only'."""
    
    task_type: str = "CAUSAL_LM"
    """Task type for adapter configuration; typically CAUSAL_LM or SEQ_CLS."""
    
    fan_in_fan_out: bool = False
    """Whether target modules have fan-in/fan-out architecture (affects transposition)."""
    
    use_rslora: bool = False
    """Whether to use rank-stabilized LoRA (RSLoRA) for improved training stability."""
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "LoraConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization parameters.
    
    Quantization reduces memory usage by decreasing the bit-precision of model weights.
    """
    
    bits: int = 4
    """Bit precision to quantize weights to, typically 2, 3, 4, or 8."""
    
    group_size: int = 128
    """Size of quantization groups, affects granularity and precision."""
    
    double_quant: bool = True
    """Whether to use double quantization to further compress quantization statistics."""
    
    use_triton: bool = False
    """Whether to use Triton kernels for quantization (when available)."""
    
    quant_method: str = "awq"
    """Quantization method to use; options include 'awq', 'gptq', 'squeezellm'."""
    
    quant_storage: str = "float16"
    """Data type for storing quantized weights, affects memory usage and precision."""
    
    dynamic_quant: bool = False
    """Whether to use dynamic quantization based on activation patterns."""
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "QuantizationConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class FinetuneConfig:
    """
    Top-level configuration for fine-tuning process.
    
    Combines adapter, quantization, and training parameters into a single configuration.
    """
    
    base_model: str
    """HuggingFace model ID or path to local model weights."""
    
    adapter: Optional[LoraConfig] = None
    """LoRA adapter configuration, if using parameter-efficient fine-tuning."""
    
    quantization: Optional[QuantizationConfig] = None
    """Quantization configuration, if using quantized weights."""
    
    training: Dict = field(default_factory=lambda: {
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
        "lr_scheduler": "cosine"
    })
    """Training hyperparameters."""
    
    max_seq_length: int = 2048
    """Maximum sequence length for training."""
    
    mixed_precision: str = "bf16"
    """Mixed precision training mode; options are 'no', 'fp16', or 'bf16'."""
    
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage."""
    
    model_revision: Optional[str] = None
    """Revision of the base model to use, if applicable."""
    
    trust_remote_code: bool = False
    """Whether to trust remote code when loading model."""
    
    flash_attention: bool = True
    """Whether to use flash attention when available."""
    
    sliding_window: Optional[int] = None
    """Size of sliding window for attention, if using sliding window attention."""
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        config_dict = {k: v for k, v in self.__dict__.items()}
        if self.adapter:
            config_dict["adapter"] = self.adapter.to_dict()
        if self.quantization:
            config_dict["quantization"] = self.quantization.to_dict()
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FinetuneConfig":
        """Create configuration from dictionary."""
        adapter_dict = config_dict.pop("adapter", None)
        quantization_dict = config_dict.pop("quantization", None)
        
        config = cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
        
        if adapter_dict:
            config.adapter = LoraConfig.from_dict(adapter_dict)
        if quantization_dict:
            config.quantization = QuantizationConfig.from_dict(quantization_dict)
            
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FinetuneConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
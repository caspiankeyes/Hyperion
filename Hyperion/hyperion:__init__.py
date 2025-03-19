# hyperion/__init__.py

"""
Hyperion: Distributed LLM Fine-Tuning Framework
===============================================

A comprehensive framework for parameter-efficient fine-tuning of large language models
with emphasis on computational efficiency, distributed training orchestration, and
production-grade deployment pipelines.

Core components:
- Parameter-efficient adapters (LoRA, QLoRA, IA³)
- Memory-hierarchical quantization
- Distributed sharded training
- Hardware-aware compilation
- Continuous evaluation
- One-command deployment

See https://hyperion-labs.github.io/docs for full documentation.
"""

__version__ = "1.5.3"

from hyperion.config import FinetuneConfig, LoraConfig, QuantizationConfig
from hyperion.trainer import Trainer
from hyperion.deployment import PackageManager

# Make core components available at the top level
__all__ = [
    "FinetuneConfig",
    "LoraConfig", 
    "QuantizationConfig",
    "Trainer",
    "PackageManager"
]
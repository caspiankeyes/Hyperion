# hyperion/quantization/__init__.py

"""
Quantization utilities for memory-efficient model training and inference.

This module provides tools for reducing the precision of model weights and activations
while maintaining model quality.
"""

from typing import Dict, Union, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM

from hyperion.config import QuantizationConfig
from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


def quantize_model(
    model_name_or_path: str,
    quantization_config: QuantizationConfig,
    **model_kwargs
) -> PreTrainedModel:
    """
    Load and quantize a model to the specified precision.
    
    Args:
        model_name_or_path: HuggingFace model ID or path to local model
        quantization_config: Quantization parameters
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Quantized model
    """
    #
    # Select quantization method based on configuration
    method = quantization_config.quant_method.lower()
    
    if method == "awq":
        logger.info(f"Using AWQ {quantization_config.bits}-bit quantization")
        return _apply_awq_quantization(model_name_or_path, quantization_config, **model_kwargs)
    elif method == "gptq":
        logger.info(f"Using GPTQ {quantization_config.bits}-bit quantization")
        return _apply_gptq_quantization(model_name_or_path, quantization_config, **model_kwargs)
    elif method == "squeezellm":
        logger.info(f"Using SqueezeLLM {quantization_config.bits}-bit quantization")
        return _apply_squeezellm_quantization(model_name_or_path, quantization_config, **model_kwargs)
    else:
        logger.warning(f"Unknown quantization method: {method}, falling back to BitsAndBytes")
        return _apply_bnb_quantization(model_name_or_path, quantization_config, **model_kwargs)


def _apply_bnb_quantization(
    model_name_or_path: str,
    quantization_config: QuantizationConfig,
    **model_kwargs
) -> PreTrainedModel:
    """Apply quantization using BitsAndBytes library."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "BitsAndBytes is not installed. "
            "Please install it with `pip install bitsandbytes`."
        )
    
    # Configure BitsAndBytes parameters
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config.bits == 4,
        load_in_8bit=quantization_config.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=quantization_config.double_quant,
        bnb_4bit_quant_type="nf4",  # Use normalized float 4
    )
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        **model_kwargs
    )
    
    return model


def _apply_awq_quantization(
    model_name_or_path: str,
    quantization_config: QuantizationConfig,
    **model_kwargs
) -> PreTrainedModel:
    """Apply quantization using AWQ (Activation-aware Weight Quantization)."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "AWQ is not installed. "
            "Please install it with `pip install autoawq`."
        )
    
    # Load model with AWQ quantization
    model = AutoAWQForCausalLM.from_quantized(
        model_name_or_path,
        **model_kwargs
    )
    
    return model


def _apply_gptq_quantization(
    model_name_or_path: str,
    quantization_config: QuantizationConfig,
    **model_kwargs
) -> PreTrainedModel:
    """Apply quantization using GPTQ."""
    try:
        from transformers import GPTQConfig
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise ImportError(
            "GPTQ is not installed. "
            "Please install it with `pip install auto-gptq`."
        )
    
    # Configure GPTQ parameters
    gptq_config = GPTQConfig(
        bits=quantization_config.bits,
        group_size=quantization_config.group_size,
        desc_act=True,
    )
    
    # Load model with GPTQ quantization
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        quantization_config=gptq_config,
        **model_kwargs
    )
    
    return model


def _apply_squeezellm_quantization(
    model_name_or_path: str,
    quantization_config: QuantizationConfig,
    **model_kwargs
) -> PreTrainedModel:
    """Apply quantization using SqueezeLLM."""
    logger.warning("SqueezeLLM quantization not yet implemented, falling back to BitsAndBytes")
    return _apply_bnb_quantization(model_name_or_path, quantization_config, **model_kwargs)


# Export additional quantization utilities
from hyperion.quantization.calibration import calibrate_model
from hyperion.quantization.memory import estimate_memory_usage

__all__ = [
    "quantize_model", 
    "calibrate_model",
    "estimate_memory_usage"
]
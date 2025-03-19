# hyperion/trainer.py

"""
Trainer implementation for Hyperion.

Orchestrates the fine-tuning process, including model loading, adapter application,
training loop management, and evaluation.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Callable

import torch
import transformers
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from hyperion.config import FinetuneConfig
from hyperion.utils.logging import get_logger
from hyperion.adapters import get_adapter_config
from hyperion.quantization import quantize_model
from hyperion.evaluation import EvaluationCallback


logger = get_logger(__name__)


class Trainer:
    """
    Hyperion's primary training orchestrator.
    
    Manages the end-to-end process of fine-tuning language models with
    parameter-efficient techniques, quantization, and distributed optimization.
    """
    
    def __init__(
        self, 
        config: FinetuneConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
        callbacks: List[transformers.TrainerCallback] = None,
        optimizers: tuple = (None, None),
    ):
        """
        Initialize the trainer with configuration and optional components.
        
        Args:
            config: Fine-tuning configuration
            tokenizer: Optional pre-initialized tokenizer
            model: Optional pre-initialized model
            callbacks: Additional callbacks for training process
            optimizers: Custom optimizer and scheduler
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []
        self.optimizers = optimizers
        
        # Track initialization state
        self._initialized = False
        self._model_prepared = False
        self._training_args = None
        
    def _initialize_components(self):
        """Initialize model and tokenizer if not provided during initialization."""
        if self.model is None:
            logger.info(f"Loading base model: {self.config.base_model}")
            
            # Configure loading parameters
            load_kwargs = {
                "revision": self.config.model_revision,
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
            }
            
            # Apply quantization if configured
            if self.config.quantization:
                logger.info(f"Applying {self.config.quantization.bits}-bit quantization")
                self.model = quantize_model(
                    self.config.base_model,
                    self.config.quantization,
                    **load_kwargs
                )
            else:
                # Load standard model
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    **load_kwargs
                )
            
            # Apply additional model optimizations
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                
            if self.config.flash_attention and hasattr(self.model, "enable_flash_attention"):
                logger.info("Enabling flash attention")
                self.model.enable_flash_attention()
        
        # Load tokenizer if not provided
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer for: {self.config.base_model}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.base_model,
                use_fast=True,
                revision=self.config.model_revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Set padding token if not present
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._initialized = True
        
    def _prepare_model_for_training(self):
        """Apply adapter configuration and prepare model for training."""
        if not self._initialized:
            self._initialize_components()
            
        if self._model_prepared:
            return
            
        # Apply quantization preparation if needed
        if self.config.quantization:
            logger.info("Preparing model for quantized training")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
            
        # Apply adapter configuration if specified
        if self.config.adapter:
            logger.info(f"Applying adapter configuration")
            peft_config = get_adapter_config(self.config.adapter)
            self.model = get_peft_model(self.model, peft_config)
            
            # Log trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
            
        self._model_prepared = True
            
    def _create_training_args(self, **kwargs):
        """Create HuggingFace TrainingArguments from Hyperion configuration."""
        # Convert Hyperion training config to HF training arguments
        args_dict = {
            # Required arguments
            "output_dir": kwargs.pop("output_dir"),
            
            # Transfer training parameters from config
            "learning_rate": self.config.training.get("learning_rate", 2e-4),
            "warmup_ratio": self.config.training.get("warmup_ratio", 0.03),
            "weight_decay": self.config.training.get("weight_decay", 0.01),
            "gradient_accumulation_steps": self.config.training.get("gradient_accumulation_steps", 4),
            "lr_scheduler_type": self.config.training.get("lr_scheduler", "cosine"),
            
            # Set from config
            "max_seq_length": self.config.max_seq_length,
            "fp16": self.config.mixed_precision == "fp16",
            "bf16": self.config.mixed_precision == "bf16",
            "gradient_checkpointing": self.config.gradient_checkpointing,
            
            # Default values with override from kwargs
            "num_train_epochs": kwargs.pop("num_train_epochs", 3),
            "per_device_train_batch_size": kwargs.pop("per_device_train_batch_size", 4),
            "per_device_eval_batch_size": kwargs.pop("per_device_eval_batch_size", 4),
            "logging_steps": kwargs.pop("logging_steps", 50),
            "eval_steps": kwargs.pop("eval_steps", 500),
            "save_steps": kwargs.pop("save_steps", 1000),
            "save_total_limit": kwargs.pop("save_total_limit", 3),
            "report_to": kwargs.pop("report_to", "tensorboard"),
            "push_to_hub": kwargs.pop("push_to_hub", False),
            "hub_model_id": kwargs.pop("hub_model_id", None),
            "hub_strategy": kwargs.pop("hub_strategy", "every_save"),
            "seed": kwargs.pop("seed", 42),
        }
        
        # Add remaining kwargs
        args_dict.update(kwargs)
        
        return TrainingArguments(**args_dict)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **kwargs
    ):
        """
        Execute the fine-tuning process.
        
        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
            **kwargs: Additional arguments passed to TrainingArguments
        
        Returns:
            TrainOutput containing training results
        """
        # Prepare model and components
        self._prepare_model_for_training()
        
        # Create training arguments
        training_args = self._create_training_args(**kwargs)
        self._training_args = training_args
        
        # Setup evaluation callback if evaluation dataset is provided
        if eval_dataset and "evaluation_strategy" not in kwargs:
            training_args.evaluation_strategy = "steps"
        
        # Add evaluation callback if needed
        callbacks = list(self.callbacks)
        if eval_dataset and any(isinstance(cb, EvaluationCallback) for cb in callbacks):
            callbacks.append(EvaluationCallback(
                model=self.model, 
                tokenizer=self.tokenizer,
                benchmark_datasets=kwargs.get("benchmark_datasets", ["mmlu", "hellaswag"])
            ))
            
        # Create and configure HuggingFace trainer
        hf_trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            optimizers=self.optimizers,
            data_collator=kwargs.get("data_collator", None),
        )
        
        # Execute training
        logger.info("Starting training process")
        result = hf_trainer.train(
            resume_from_checkpoint=kwargs.get("resume_from_checkpoint", None)
        )
        
        # Save model and tokenizer
        if training_args.output_dir:
            logger.info(f"Saving model to {training_args.output_dir}")
            hf_trainer.save_model(training_args.output_dir)
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            # Save configuration
            self.config.save(os.path.join(training_args.output_dir, "hyperion_config.json"))
            
        return result
        
    def export_for_inference(self, output_dir: str, **kwargs):
        """
        Prepare and export the model for inference deployment.
        
        Args:
            output_dir: Directory to save the exported model
            **kwargs: Additional export options
        """
        if not self._model_prepared:
            self._prepare_model_for_training()
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Merge LoRA weights if applicable
        if self.config.adapter and hasattr(self.model, "merge_and_unload"):
            logger.info("Merging adapter weights with base model")
            self.model = self.model.merge_and_unload()
            
        # Apply additional optimizations
        inference_config = kwargs.get("inference_config", {})
        if inference_config.get("optimize_for_inference", True):
            logger.info("Optimizing model for inference")
            # Apply specific optimizations based on deployment target
            target = inference_config.get("target", "general")
            if target == "onnx":
                self._export_to_onnx(output_dir, inference_config)
            elif target == "tensorrt":
                self._export_to_tensorrt(output_dir, inference_config)
            else:
                # Standard export for PyTorch
                logger.info("Exporting standard PyTorch model")
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                # Save inference config
                with open(os.path.join(output_dir, "hyperion_inference_config.json"), "w") as f:
                    json.dump(inference_config, f, indent=2)
        
        logger.info(f"Model exported successfully to {output_dir}")
        return output_dir
        
    def _export_to_onnx(self, output_dir: str, config: Dict):
        """Export model to ONNX format."""
        # Implementation for ONNX export
        logger.info("ONNX export not yet implemented")
        
    def _export_to_tensorrt(self, output_dir: str, config: Dict):
        """Export model to TensorRT format."""
        # Implementation for TensorRT export
        logger.info("TensorRT export not yet implemented")
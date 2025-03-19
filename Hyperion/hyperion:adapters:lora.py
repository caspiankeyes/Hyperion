# hyperion/adapters/lora.py

"""
Low-Rank Adaptation (LoRA) implementation.

LoRA reduces parameter count by factorizing weight updates into low-rank matrices.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from hyperion.config import LoraConfig


class LoraLayer(nn.Module):
    """
    Implementation of a LoRA (Low-Rank Adaptation) layer.
    
    This layer applies a low-rank update to the original weights:
    W + ΔW, where ΔW = BA and B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        scaling: float = 1.0,
        dropout_p: float = 0.0,
    ):
        """
        Initialize a LoRA adapter layer.
        
        Args:
            base_layer: Original layer to adapt
            rank: Rank of the update matrices
            scaling: Scaling factor for the update
            dropout_p: Dropout probability for regularization
        """
        super().__init__()
        
        # Store original layer and dimensions
        self.base_layer = base_layer
        
        # Determine input and output dimensions based on weight shape
        if hasattr(base_layer, "weight"):
            weight = base_layer.weight
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]
        else:
            raise ValueError("Base layer must have a 'weight' attribute")
        
        # Store hyperparameters
        self.rank = rank
        self.scaling = scaling / rank
        
        # Initialize low-rank matrices
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        # Initialize with random weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Initially enabled
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with LoRA adaptation applied
        """
        # Get base layer output
        base_output = self.base_layer(x)
        
        # Skip LoRA computation if disabled
        if not self.enabled:
            return base_output
        
        # Compute LoRA contribution
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        
        # Apply scaling factor
        lora_output = lora_output * self.scaling
        
        # Combine base and LoRA outputs
        return base_output + lora_output
        
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the base layer weights.
        
        After merging, the LoRA layers can be disabled or removed for efficient inference.
        """
        if not self.enabled:
            return
            
        # Compute merged weights
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        
        # Update base weights
        with torch.no_grad():
            self.base_layer.weight.data += delta_w
            
        # Disable LoRA after merging
        self.enabled = False
        
    def unmerge_weights(self) -> None:
        """
        Revert merged weights, separating LoRA contribution from base weights.
        """
        if self.enabled:
            return
            
        # Compute LoRA contribution
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        
        # Remove LoRA contribution
        with torch.no_grad():
            self.base_layer.weight.data -= delta_w
            
        # Re-enable LoRA
        self.enabled = True


class LoraAdapter:
    """
    Applies LoRA adaptation to a model.
    
    Utility class for adding and managing LoRA layers throughout a model.
    """
    
    def __init__(self, config: LoraConfig):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: LoRA configuration parameters
        """
        self.config = config
        self.lora_layers = {}
        
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA adaptation to a model.
        
        Args:
            model: The model to adapt
            
        Returns:
            Adapted model with LoRA layers
        """
        # Recursively search for target modules and apply LoRA
        return self._apply_to_submodules(model, "")
    
    def _apply_to_submodules(self, module: nn.Module, path: str) -> nn.Module:
        """
        Recursively apply LoRA to matching submodules.
        
        Args:
            module: Current module to process
            path: Dot-separated path to current module
            
        Returns:
            Module with LoRA applied to matching submodules
        """
        # Process child modules first
        for name, child in list(module.named_children()):
            child_path = f"{path}.{name}" if path else name
            
            # Recursively apply to children
            setattr(module, name, self._apply_to_submodules(child, child_path))
        
        # Check if current module should have LoRA applied
        if self._should_apply_lora(module, path):
            return self._create_lora_layer(module)
        
        return module
        
    def _should_apply_lora(self, module: nn.Module, path: str) -> bool:
        """
        Determine if LoRA should be applied to this module.
        
        Args:
            module: Module to check
            path: Path to module
            
        Returns:
            Whether to apply LoRA to this module
        """
        # Check if module has required attributes for LoRA
        if not hasattr(module, "weight"):
            return False
            
        # Check against target modules list
        target_modules = self.config.target_modules
        
        # Check by full path match
        if path in target_modules:
            return True
            
        # Check by module type name
        module_type = module.__class__.__name__
        if module_type in target_modules:
            return True
            
        # Check if module name contains any target pattern
        for target in target_modules:
            if target in path.split(".")[-1]:
                return True
                
        return False
    
    def _create_lora_layer(self, module: nn.Module) -> nn.Module:
        """
        Create a LoRA adapter layer for the given module.
        
        Args:
            module: Base module to adapt
            
        Returns:
            LoRA-adapted module
        """
        return LoraLayer(
            base_layer=module,
            rank=self.config.r,
            scaling=self.config.lora_alpha,
            dropout_p=self.config.lora_dropout
        )
        
    def merge_weights(self, model: nn.Module) -> nn.Module:
        """
        Merge LoRA weights into the base model weights.
        
        Args:
            model: Model with LoRA layers
            
        Returns:
            Model with LoRA weights merged for efficient inference
        """
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.merge_weights()
        return model
        
    def unmerge_weights(self, model: nn.Module) -> nn.Module:
        """
        Unmerge LoRA weights from the base model weights.
        
        Args:
            model: Model with merged LoRA weights
            
        Returns:
            Model with separated LoRA layers
        """
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge_weights()
        return model
# hyperion/evaluation/benchmarks/hellaswag.py

"""
HellaSwag benchmark implementation for Hyperion.

HellaSwag is a commonsense natural language inference benchmark that tests whether a model can
complete a sentence or paragraph in a way that displays common sense.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from hyperion.evaluation.benchmarks.base import BenchmarkEvaluator
from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


class HellaSwagEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the HellaSwag benchmark.
    
    Implements evaluation methods for testing commonsense reasoning in language models.
    """
    
    def __init__(self):
        """Initialize the HellaSwag evaluator."""
        super().__init__()
        self.name = "hellaswag"
        self.primary_metric = "accuracy"
        self.dataset = None
        
    def _load_dataset(self):
        """Load the HellaSwag dataset."""
        if self.dataset is not None:
            return
            
        try:
            from datasets import load_dataset
            self.dataset = load_dataset("hellaswag", split="validation")
            logger.info(f"Loaded HellaSwag dataset with {len(self.dataset)} examples")
        except Exception as e:
            raise RuntimeError(f"Failed to load HellaSwag dataset: {e}")
            
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_samples: Optional[int] = None,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate model on HellaSwag benchmark.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            num_samples: Number of samples to evaluate on (if None, use all)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset
        self._load_dataset()
        
        # Limit to num_samples if specified
        if num_samples is not None and num_samples < len(self.dataset):
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            indices = indices[:num_samples]
            dataset = self.dataset.select(indices)
        else:
            dataset = self.dataset
            
        # Prepare evaluation loop
        total_correct = 0
        total_examples = 0
        
        logger.info(f"Evaluating on {len(dataset)} HellaSwag examples")
        
        # Set model to evaluation mode
        model.eval()
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            correct = self._evaluate_batch(model, tokenizer, batch)
            
            total_correct += correct
            total_examples += len(batch)
            
        # Calculate metrics
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct": total_correct
        }
        
    def _evaluate_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch: Dict[str, List[Any]]
    ) -> int:
        """
        Evaluate a batch of examples.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            batch: Batch of examples
            
        Returns:
            Number of correct predictions
        """
        contexts = batch["ctx"]
        endings = batch["endings"]
        labels = batch["label"]
        
        correct = 0
        
        with torch.no_grad():
            for ctx, ends, label in zip(contexts, endings, labels):
                # Prepare choices
                choices = [ctx + " " + end for end in ends]
                
                # Tokenize all choices
                encodings = [tokenizer(choice, return_tensors="pt").to(model.device) for choice in choices]
                
                # Get log probabilities for each choice
                choice_logprobs = []
                
                for encoding in encodings:
                    # Forward pass
                    outputs = model(**encoding)
                    logits = outputs.logits
                    
                    # Calculate sequence log probability
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_ids = encoding["input_ids"][..., 1:].contiguous()
                    
                    # Get token log probs
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    token_log_probs = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
                    
                    # Sum log probs for sequence score
                    seq_log_prob = token_log_probs.sum().item()
                    choice_logprobs.append(seq_log_prob)
                
                # Select most likely ending
                predicted_label = np.argmax(choice_logprobs)
                
                # Check if prediction is correct
                if predicted_label == label:
                    correct += 1
                    
        return correct
        
    def get_primary_metric(self) -> str:
        """Get the primary evaluation metric."""
        return self.primary_metric
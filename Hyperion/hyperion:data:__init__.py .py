# hyperion/data/__init__.py

"""
Dataset processing utilities for fine-tuning.

This module provides tools for preparing, processing, and augmenting datasets
for language model fine-tuning.
"""

from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import os
import logging
import random

from datasets import Dataset, load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


class DatasetProcessor:
    """
    Processor for preparing datasets for fine-tuning.
    
    Handles dataset loading, filtering, formatting, and tokenization for
    efficient training preparation.
    """
    
    def __init__(
        self,
        dataset: Union[str, Dataset, Dict],
        instruction_template: Optional[str] = None,
        input_template: Optional[str] = None,
        output_template: Optional[str] = None,
        sample_limit: Optional[int] = None,
        validation_split: float = 0.1,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        seed: int = 42,
    ):
        """
        Initialize the dataset processor.
        
        Args:
            dataset: Dataset identifier or loaded dataset
            instruction_template: Template string for formatting instructions
            input_template: Template string for formatting inputs
            output_template: Template string for formatting outputs
            sample_limit: Maximum number of samples to use
            validation_split: Proportion of data to use for validation
            tokenizer: Optional tokenizer for tokenization
            seed: Random seed for reproducibility
        """
        self.dataset_src = dataset
        self.instruction_template = instruction_template
        self.input_template = input_template
        self.output_template = output_template
        self.sample_limit = sample_limit
        self.validation_split = validation_split
        self.tokenizer = tokenizer
        self.seed = seed
        
        # Will be set during preparation
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset_format = self._detect_format()
        
    def _detect_format(self) -> str:
        """
        Detect the format of the dataset based on source and templates.
        
        Returns:
            Detected format ('alpaca', 'sharegpt', 'general', etc.)
        """
        # Check if explicit format is provided
        if isinstance(self.dataset_src, dict) and 'format' in self.dataset_src:
            return self.dataset_src['format']
            
        # Try to detect from the dataset source if it's a string
        if isinstance(self.dataset_src, str):
            src_lower = self.dataset_src.lower()
            if 'alpaca' in src_lower:
                return 'alpaca'
            elif 'sharegpt' in src_lower:
                return 'sharegpt'
            elif 'wizardlm' in src_lower:
                return 'wizardlm'
            elif 'dolly' in src_lower:
                return 'dolly'
                
        # Default to general format
        return 'general'
        
    def _load_dataset(self) -> Dataset:
        """
        Load the dataset from the provided source.
        
        Returns:
            Loaded dataset
        """
        if isinstance(self.dataset_src, Dataset):
            return self.dataset_src
            
        if isinstance(self.dataset_src, dict):
            if 'path' in self.dataset_src:
                # Load from path with optional arguments
                load_args = {k: v for k, v in self.dataset_src.items() if k != 'path'}
                return load_dataset(self.dataset_src['path'], **load_args)
            elif 'train' in self.dataset_src:
                # Pre-split dataset dict
                return DatasetDict(self.dataset_src)
        
        if isinstance(self.dataset_src, str):
            # Simple dataset loading by name
            logger.info(f"Loading dataset from HuggingFace Hub: {self.dataset_src}")
            try:
                return load_dataset(self.dataset_src)
            except Exception as e:
                # Try loading as a local path
                logger.info(f"Failed to load from Hub, trying as local path: {str(e)}")
                if os.path.exists(self.dataset_src):
                    # Guess format based on extension
                    extension = os.path.splitext(self.dataset_src)[1].lower()
                    if extension == '.json':
                        return load_dataset('json', data_files=self.dataset_src)
                    elif extension in ['.csv', '.tsv']:
                        return load_dataset('csv', data_files=self.dataset_src)
                    elif extension in ['.parquet', '.pq']:
                        return load_dataset('parquet', data_files=self.dataset_src)
                    else:
                        raise ValueError(f"Unsupported file extension: {extension}")
                else:
                    raise ValueError(f"Dataset source not found: {self.dataset_src}")
                    
        raise ValueError(f"Unsupported dataset source: {self.dataset_src}")
        
    def _get_formatting_function(self) -> Callable:
        """
        Get the appropriate formatting function based on dataset format.
        
        Returns:
            Function for formatting dataset examples
        """
        dataset_format = self.dataset_format
        
        if dataset_format == 'alpaca':
            return self._format_alpaca
        elif dataset_format == 'sharegpt':
            return self._format_sharegpt
        elif dataset_format == 'wizardlm':
            return self._format_wizardlm
        elif dataset_format == 'dolly':
            return self._format_dolly
        else:
            return self._format_general
            
    def _format_alpaca(self, example: Dict) -> Dict:
        """Format example in Alpaca dataset format."""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # Apply templates
        if self.instruction_template and instruction:
            instruction = self.instruction_template.format(instruction=instruction)
            
        if self.input_template and input_text:
            input_text = self.input_template.format(input=input_text)
            
        if self.output_template and output:
            output = self.output_template.format(output=output)
            
        # Combine components
        if input_text:
            text = f"{instruction}\n{input_text}\n{output}"
        else:
            text = f"{instruction}\n{output}"
            
        return {"text": text}
        
    def _format_sharegpt(self, example: Dict) -> Dict:
        """Format example in ShareGPT dataset format."""
        conversations = example.get('conversations', [])
        
        formatted_text = ""
        for i, msg in enumerate(conversations):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            if i > 0:
                formatted_text += "\n\n"
                
            if role.lower() == 'human':
                formatted_text += f"Human: {content}"
            elif role.lower() in ['assistant', 'gpt']:
                formatted_text += f"Assistant: {content}"
            else:
                formatted_text += f"{role}: {content}"
                
        return {"text": formatted_text}
        
    def _format_wizardlm(self, example: Dict) -> Dict:
        """Format example in WizardLM dataset format."""
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        text = f"Instruction: {instruction}\n\nResponse: {output}"
        return {"text": text}
        
    def _format_dolly(self, example: Dict) -> Dict:
        """Format example in Dolly dataset format."""
        instruction = example.get('instruction', '')
        context = example.get('context', '')
        response = example.get('response', '')
        
        if context:
            text = f"Instruction: {instruction}\n\nContext: {context}\n\nResponse: {response}"
        else:
            text = f"Instruction: {instruction}\n\nResponse: {response}"
            
        return {"text": text}
        
    def _format_general(self, example: Dict) -> Dict:
        """General formatting when format is unknown."""
        # Try to detect components or use the entire example as text
        text = example.get('text', '')
        
        if not text:
            # Try to construct from other common fields
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            response = example.get('response', '')
            context = example.get('context', '')
            
            components = []
            
            if instruction:
                components.append(f"Instruction: {instruction}")
                
            if context:
                components.append(f"Context: {context}")
                
            if input_text:
                components.append(f"Input: {input_text}")
                
            if output:
                components.append(f"Output: {output}")
            elif response:
                components.append(f"Response: {response}")
                
            if components:
                text = "\n\n".join(components)
            else:
                # Last resort: serialize the entire example
                import json
                text = json.dumps(example)
                
        return {"text": text}
        
    def _tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize text examples."""
        if self.tokenizer is None:
            return examples
            
        tokenized = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
        
    def prepare(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare datasets for training and evaluation.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Load the dataset
        dataset = self._load_dataset()
        
        # Determine which split to use if dataset has multiple splits
        if isinstance(dataset, DatasetDict):
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                # Use the first split
                dataset = dataset[list(dataset.keys())[0]]
                
        # Apply sample limit if specified
        if self.sample_limit and len(dataset) > self.sample_limit:
            indices = random.Random(self.seed).sample(range(len(dataset)), self.sample_limit)
            dataset = dataset.select(indices)
            
        # Format the dataset
        format_func = self._get_formatting_function()
        formatted_dataset = dataset.map(
            format_func,
            remove_columns=dataset.column_names
        )
        
        # Split into train and validation sets if needed
        if self.validation_split > 0:
            splits = formatted_dataset.train_test_split(
                test_size=self.validation_split,
                seed=self.seed
            )
            train_dataset = splits['train']
            eval_dataset = splits['test']
        else:
            train_dataset = formatted_dataset
            eval_dataset = None
            
        # Apply tokenization if tokenizer is provided
        if self.tokenizer:
            logger.info("Tokenizing datasets")
            
            train_dataset = train_dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=["text"]
                )
                
        # Save references
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        logger.info(f"Dataset preparation complete: {len(train_dataset)} training samples")
        if eval_dataset:
            logger.info(f"Validation set has {len(eval_dataset)} samples")
            
        return train_dataset, eval_dataset


# Export additional dataset utilities
from hyperion.data.augmentation import DataAugmenter
from hyperion.data.filters import FilteringPipeline

__all__ = [
    "DatasetProcessor",
    "DataAugmenter",
    "FilteringPipeline"
]
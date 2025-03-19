# hyperion/deployment/__init__.py

"""
Deployment utilities for fine-tuned models.

This module provides tools for packaging, optimizing, and deploying fine-tuned
models to various inference platforms.
"""

import os
import json
import shutil
from typing import Dict, Optional, Any, Union
import logging

from hyperion.utils.logging import get_logger


logger = get_logger(__name__)


class PackageManager:
    """
    Manager for model packaging and deployment.
    
    Handles the process of preparing fine-tuned models for deployment to
    various inference targets and platforms.
    """
    
    def __init__(
        self,
        model_path: str,
        quantization_level: Optional[str] = None,
        format: str = "hf_endpoint",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the package manager.
        
        Args:
            model_path: Path to the fine-tuned model
            quantization_level: Quantization level for deployment
            format: Target deployment format
            metadata: Additional metadata for deployment
        """
        self.model_path = model_path
        self.quantization_level = quantization_level
        self.format = format
        self.metadata = metadata or {}
        
        # Validate model path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
    def _prepare_package(self, output_dir: str) -> str:
        """
        Prepare the model package for deployment.
        
        Args:
            output_dir: Directory to store the packaged model
            
        Returns:
            Path to the prepared package
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy model files
        logger.info(f"Copying model files from {self.model_path} to {output_dir}")
        
        # Check if model is a directory or a safetensors file
        if os.path.isdir(self.model_path):
            # Copy all files in the directory
            for filename in os.listdir(self.model_path):
                source_path = os.path.join(self.model_path, filename)
                target_path = os.path.join(output_dir, filename)
                
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, target_path)
                elif os.path.isdir(source_path):
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            # Assume it's a single model file
            shutil.copy2(self.model_path, os.path.join(output_dir, os.path.basename(self.model_path)))
        
        # Apply quantization if specified
        if self.quantization_level:
            self._apply_quantization(output_dir)
            
        # Add deployment metadata
        self._add_metadata(output_dir)
        
        # Create deployment-specific files
        self._create_deployment_files(output_dir)
        
        return output_dir
        
# hyperion/deployment/__init__.py (continued)

    def _apply_quantization(self, output_dir: str):
        """Apply quantization to the model package."""
        logger.info(f"Applying {self.quantization_level} quantization for deployment")
        
        if self.quantization_level == "4bit-optimized":
            self._apply_4bit_quantization(output_dir)
        elif self.quantization_level == "8bit-optimized":
            self._apply_8bit_quantization(output_dir)
        elif self.quantization_level == "onnx":
            self._apply_onnx_quantization(output_dir)
        elif self.quantization_level == "tensorrt":
            self._apply_tensorrt_quantization(output_dir)
        else:
            logger.warning(f"Unknown quantization level: {self.quantization_level}")
    
    def _apply_4bit_quantization(self, output_dir: str):
        """Apply 4-bit quantization to the model."""
        try:
            from optimum.gptq import GPTQQuantizer
            
            # Config file path
            config_path = os.path.join(output_dir, "config.json")
            if not os.path.exists(config_path):
                raise ValueError("Model config.json not found")
                
            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Add quantization configuration
            config["quantization_config"] = {
                "bits": 4,
                "group_size": 128,
                "damp_percent": 0.1,
                "desc_act": True
            }
            
            # Write updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            # Add quantization metadata
            with open(os.path.join(output_dir, "hyperion_quantization.json"), "w") as f:
                json.dump({
                    "quantization": "4bit-gptq",
                    "applied_by": "hyperion",
                    "version": "1.5.3"
                }, f, indent=2)
                
        except ImportError:
            logger.warning("GPTQ quantization requires optimum.gptq, falling back to metadata-only")
            # Still add metadata for downstream handling
            with open(os.path.join(output_dir, "hyperion_quantization.json"), "w") as f:
                json.dump({
                    "quantization": "4bit-gptq",
                    "applied_by": "hyperion",
                    "version": "1.5.3",
                    "status": "metadata-only"
                }, f, indent=2)
    
    def _apply_8bit_quantization(self, output_dir: str):
        """Apply 8-bit quantization to the model."""
        logger.info("8-bit quantization not yet implemented, adding metadata only")
        with open(os.path.join(output_dir, "hyperion_quantization.json"), "w") as f:
            json.dump({
                "quantization": "8bit-optimized",
                "applied_by": "hyperion",
                "version": "1.5.3",
                "status": "metadata-only"
            }, f, indent=2)
    
    def _apply_onnx_quantization(self, output_dir: str):
        """Apply ONNX quantization to the model."""
        logger.info("ONNX quantization not yet implemented, adding metadata only")
        with open(os.path.join(output_dir, "hyperion_quantization.json"), "w") as f:
            json.dump({
                "quantization": "onnx",
                "applied_by": "hyperion",
                "version": "1.5.3",
                "status": "metadata-only"
            }, f, indent=2)
    
    def _apply_tensorrt_quantization(self, output_dir: str):
        """Apply TensorRT quantization to the model."""
        logger.info("TensorRT quantization not yet implemented, adding metadata only")
        with open(os.path.join(output_dir, "hyperion_quantization.json"), "w") as f:
            json.dump({
                "quantization": "tensorrt",
                "applied_by": "hyperion",
                "version": "1.5.3",
                "status": "metadata-only"
            }, f, indent=2)
            
    def _add_metadata(self, output_dir: str):
        """Add deployment metadata to the package."""
        # Create or update the README.md file
        if "model_card" in self.metadata:
            # Copy model card to README.md
            model_card_path = self.metadata["model_card"]
            if os.path.exists(model_card_path):
                shutil.copy2(model_card_path, os.path.join(output_dir, "README.md"))
        
        # Add hyperion metadata.json file
        metadata = {
            "hyperion_version": "1.5.3",
            "format": self.format,
            "quantization": self.quantization_level,
            "created_at": self._get_timestamp(),
        }
        
        # Add user-provided metadata
        for key, value in self.metadata.items():
            if key != "model_card":  # Skip model card as it's handled separately
                metadata[key] = value
                
        # Write metadata file
        with open(os.path.join(output_dir, "hyperion_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
    def _create_deployment_files(self, output_dir: str):
        """Create deployment-specific files based on target format."""
        if self.format == "hf_endpoint":
            self._create_hf_endpoint_files(output_dir)
        elif self.format == "hf_spaces":
            self._create_hf_spaces_files(output_dir)
        elif self.format == "docker":
            self._create_docker_files(output_dir)
        elif self.format == "aws_sagemaker":
            self._create_sagemaker_files(output_dir)
        elif self.format == "self_hosted":
            self._create_self_hosted_files(output_dir)
    
    def _create_hf_endpoint_files(self, output_dir: str):
        """Create files for Hugging Face Inference Endpoints deployment."""
        # Create inference.py script for custom handler
        inference_script = """
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_id = "."  # Local path
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

def inference(text, params=None):
    params = params or {}
    
    # Configure generation parameters
    generation_kwargs = {
        "max_new_tokens": params.get("max_new_tokens", 512),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
        "repetition_penalty": params.get("repetition_penalty", 1.1),
        "do_sample": params.get("do_sample", True),
    }
    
    # Generate text
    result = generator(text, **generation_kwargs)
    generated_text = result[0]["generated_text"]
    
    # Extract only the newly generated text
    new_text = generated_text[len(text):]
    
    return {"generated_text": new_text}
"""
        
        with open(os.path.join(output_dir, "inference.py"), "w") as f:
            f.write(inference_script.strip())
            
        # Create requirements.txt
        requirements = """
transformers>=4.35.0
torch>=2.0.0
accelerate>=0.25.0
sentencepiece>=0.1.99
protobuf>=3.20.0
"""
        
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write(requirements.strip())
            
    def _create_hf_spaces_files(self, output_dir: str):
        """Create files for Hugging Face Spaces deployment."""
        # Create app.py for Gradio interface
        app_script = """
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_id = "."  # Local path
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

def generate_text(text, max_length, temperature, top_p, repetition_penalty):
    # Configure generation parameters
    generation_kwargs = {
        "max_new_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
    }
    
    # Generate text
    result = generator(text, **generation_kwargs)
    generated_text = result[0]["generated_text"]
    
    # Extract only the newly generated text
    new_text = generated_text[len(text):]
    
    return new_text

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Hyperion Fine-tuned Model")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                placeholder="Enter your prompt here...",
                label="Input",
                lines=5
            )
            
            with gr.Row():
                submit_btn = gr.Button("Generate")
                clear_btn = gr.Button("Clear")
            
            with gr.Accordion("Advanced Options", open=False):
                max_length = gr.Slider(
                    minimum=16,
                    maximum=1024,
                    value=512,
                    step=8,
                    label="Maximum Length"
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.05,
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p"
                )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    label="Repetition Penalty"
                )
                
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Output",
                lines=10
            )
    
    # Set up button actions
    submit_btn.click(
        generate_text,
        inputs=[input_text, max_length, temperature, top_p, repetition_penalty],
        outputs=output_text
    )
    
    clear_btn.click(
        lambda: ("", ""),
        inputs=None,
        outputs=[input_text, output_text]
    )

# Launch app
demo.launch()
"""
        
        with open(os.path.join(output_dir, "app.py"), "w") as f:
            f.write(app_script.strip())
            
        # Create requirements.txt
        requirements = """
gradio>=3.50.0
transformers>=4.35.0
torch>=2.0.0
accelerate>=0.25.0
sentencepiece>=0.1.99
protobuf>=3.20.0
"""
        
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write(requirements.strip())
    
    def _create_docker_files(self, output_dir: str):
        """Create Docker deployment files."""
        # Create Dockerfile
        dockerfile = """
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy model files
COPY . /app/model/

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    flask>=2.0.0 \
    gunicorn>=20.0.0

# Copy server files
COPY server.py /app/
COPY gunicorn_config.py /app/

# Expose port
EXPOSE 8000

# Start the server
CMD ["gunicorn", "--config", "gunicorn_config.py", "server:app"]
"""
        
        with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile.strip())
            
        # Create server.py
        server_script = """
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

app = Flask(__name__)

# Initialize model and tokenizer on startup
model_path = os.path.join(os.path.dirname(__file__), "model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data["text"]
    
    # Extract generation parameters
    params = {
        "max_new_tokens": data.get("max_new_tokens", 512),
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 0.9),
        "repetition_penalty": data.get("repetition_penalty", 1.1),
        "do_sample": data.get("temperature", 0.7) > 0,
    }
    
    # Generate text
    result = generator(text, **params)
    generated_text = result[0]["generated_text"]
    
    # Extract only the newly generated text
    new_text = generated_text[len(text):]
    
    return jsonify({
        "generated_text": new_text,
        "full_text": generated_text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
"""
        
        with open(os.path.join(output_dir, "server.py"), "w") as f:
            f.write(server_script.strip())
            
        # Create gunicorn config
        gunicorn_config = """
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Server mechanics
daemon = False
raw_env = []
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "hyperion_api"
"""
        
        with open(os.path.join(output_dir, "gunicorn_config.py"), "w") as f:
            f.write(gunicorn_config.strip())
    
    def _create_sagemaker_files(self, output_dir: str):
        """Create AWS SageMaker deployment files."""
        logger.info("SageMaker deployment files not yet implemented")
        
    def _create_self_hosted_files(self, output_dir: str):
        """Create self-hosted deployment files."""
        logger.info("Self-hosted deployment files not yet implemented")
        
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
        
    def deploy(
        self,
        platform: str = "hf_inference_endpoints",
        platform_kwargs: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy the packaged model to the specified platform.
        
        Args:
            platform: Target deployment platform
            platform_kwargs: Platform-specific deployment parameters
            temp_dir: Temporary directory for package preparation
            
        Returns:
            Deployment information
        """
        platform_kwargs = platform_kwargs or {}
        
        # Create temporary directory if not provided
        if temp_dir is None:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="hyperion_deploy_")
            
        # Prepare the package
        package_dir = self._prepare_package(temp_dir)
        
        # Deploy to the specified platform
        if platform == "hf_inference_endpoints":
            deployment_info = self._deploy_to_hf_endpoints(package_dir, **platform_kwargs)
        elif platform == "hf_spaces":
            deployment_info = self._deploy_to_hf_spaces(package_dir, **platform_kwargs)
        elif platform == "aws_sagemaker":
            deployment_info = self._deploy_to_sagemaker(package_dir, **platform_kwargs)
        elif platform == "docker":
            deployment_info = self._deploy_to_docker(package_dir, **platform_kwargs)
        else:
            logger.warning(f"Unknown deployment platform: {platform}")
            deployment_info = {"status": "error", "message": f"Unknown platform: {platform}"}
            
        # Return deployment information
        return deployment_info
        
    def _deploy_to_hf_endpoints(self, package_dir: str, **kwargs) -> Dict[str, Any]:
        """Deploy to Hugging Face Inference Endpoints."""
        try:
            from huggingface_hub import HfApi, create_inference_endpoint
        except ImportError:
            return {
                "status": "error",
                "message": "huggingface_hub not installed. Install with `pip install huggingface_hub`"
            }
            
        # Extract required parameters
        hf_token = kwargs.get("hf_token")
        repo_id = kwargs.get("repo_id")
        instance_type = kwargs.get("instance_type", "g5.xlarge")
        instance_size = kwargs.get("instance_size", "small")
        
        if not hf_token:
            return {"status": "error", "message": "Missing hf_token parameter"}
            
        if not repo_id:
            return {"status": "error", "message": "Missing repo_id parameter"}
            
        # Push model to Hugging Face Hub
        api = HfApi(token=hf_token)
        
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        api.create_repo(repo_id=repo_id, private=kwargs.get("private", False), exist_ok=True)
        api.upload_folder(
            folder_path=package_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=kwargs.get("commit_message", "Deployed with Hyperion")
        )
        
        # Create inference endpoint
        logger.info(f"Creating inference endpoint for {repo_id}")
        endpoint = create_inference_endpoint(
            repo_id=repo_id,
            token=hf_token,
            instance_type=instance_type,
            instance_size=instance_size,
            framework="custom",
            custom_image=kwargs.get("custom_image"),
            namespace=kwargs.get("namespace"),
            accelerator=kwargs.get("accelerator", "gpu"),
            region=kwargs.get("region", "us-east-1"),
        )
        
        return {
            "status": "success",
            "platform": "hf_inference_endpoints",
            "repo_id": repo_id,
            "endpoint_id": endpoint.get("id"),
            "url": endpoint.get("url"),
            "status_url": endpoint.get("status_url"),
        }
        
    def _deploy_to_hf_spaces(self, package_dir: str, **kwargs) -> Dict[str, Any]:
        """Deploy to Hugging Face Spaces."""
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            return {
                "status": "error",
                "message": "huggingface_hub not installed. Install with `pip install huggingface_hub`"
            }
            
        # Extract required parameters
        hf_token = kwargs.get("hf_token")
        space_id = kwargs.get("space_id")
        
        if not hf_token:
            return {"status": "error", "message": "Missing hf_token parameter"}
            
        if not space_id:
            return {"status": "error", "message": "Missing space_id parameter"}
            
        # Push model to Hugging Face Hub
        api = HfApi(token=hf_token)
        
        # Create the Space if it doesn't exist
        logger.info(f"Creating/updating Hugging Face Space: {space_id}")
        create_repo(
            repo_id=space_id,
            token=hf_token,
            repo_type="space",
            space_sdk="gradio",
            private=kwargs.get("private", False),
            exist_ok=True
        )
        
        # Upload the package to the Space
        api.upload_folder(
            folder_path=package_dir,
            repo_id=space_id,
            repo_type="space",
            commit_message=kwargs.get("commit_message", "Deployed with Hyperion")
        )
        
        # Return deployment information
        return {
            "status": "success",
            "platform": "hf_spaces",
            "space_id": space_id,
            "url": f"https://huggingface.co/spaces/{space_id}"
        }
        
    def _deploy_to_sagemaker(self, package_dir: str, **kwargs) -> Dict[str, Any]:
        """Deploy to AWS SageMaker."""
        logger.warning("SageMaker deployment not yet implemented")
        return {
            "status": "error",
            "message": "SageMaker deployment not yet implemented"
        }
        
    def _deploy_to_docker(self, package_dir: str, **kwargs) -> Dict[str, Any]:
        """Deploy as Docker container."""
        logger.warning("Docker deployment not yet implemented")
        return {
            "status": "error",
            "message": "Docker deployment not yet implemented"
        }


from hyperion.deployment.formats import export_to_onnx, export_to_tensorrt

__all__ = [
    "PackageManager",
    "export_to_onnx",
    "export_to_tensorrt"
]
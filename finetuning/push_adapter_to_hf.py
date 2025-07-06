#!/usr/bin/env python3
"""
Push the fine-tuned LoRA adapter to Hugging Face Hub.

This script uploads the Meditron3-8B LoRA adapter trained for medical MCQ 
JSON output generation to Hugging Face Hub.
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
from huggingface_hub.utils import HfHubHTTPError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Push LoRA adapter to Hugging Face Hub")
    parser.add_argument("--adapter_path", type=str, 
                        default="final_lora_adapter",
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name on Hugging Face (e.g., 'username/model-name')")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token (will use HF_TOKEN env var if not provided)")
    parser.add_argument("--commit_message", type=str, 
                        default="Upload Meditron3-8B LoRA adapter for medical MCQ JSON generation",
                        help="Commit message for the upload")
    return parser.parse_args()

def validate_adapter_directory(adapter_path):
    """Validate that the adapter directory contains required files."""
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logging.error(f"Adapter directory {adapter_path} does not exist!")
        return False
    
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]
    
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "README.md"
    ]
    
    # Check required files
    for file_name in required_files:
        file_path = adapter_dir / file_name
        if not file_path.exists():
            logging.error(f"Required file {file_name} not found in {adapter_path}")
            return False
        logging.info(f"✓ Found required file: {file_name}")
    
    # Check optional files
    for file_name in optional_files:
        file_path = adapter_dir / file_name
        if file_path.exists():
            logging.info(f"✓ Found optional file: {file_name}")
        else:
            logging.info(f"○ Optional file not found: {file_name}")
    
    return True

def read_adapter_config(adapter_path):
    """Read and return the adapter configuration."""
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error reading adapter config: {e}")
        return None

def create_model_card(adapter_config, training_info):
    """Create a comprehensive model card for the adapter."""
    
    base_model = adapter_config.get("base_model_name_or_path", "OpenMeditron/Meditron3-8B")
    lora_r = adapter_config.get("r", 64)
    lora_alpha = adapter_config.get("lora_alpha", 128)
    target_modules = adapter_config.get("target_modules", [])
    
    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- medical
- mcq
- lora
- peft
- json-generation
- french
- meditron
language:
- fr
- en
pipeline_tag: text-generation
library_name: peft
---

# Meditron3-8B LoRA Adapter for Medical MCQ JSON Generation

This is a LoRA (Low-Rank Adaptation) adapter for the **{base_model}** model, fine-tuned for medical multiple-choice question answering with structured JSON output generation.

## Model Details

### Base Model
- **Model**: {base_model}
- **Architecture**: Llama-based medical language model
- **Parameters**: 8B parameters
- **Precision**: BFloat16

### LoRA Configuration
- **Rank (r)**: {lora_r}
- **Alpha**: {lora_alpha}
- **Dropout**: {adapter_config.get("lora_dropout", 0.1)}
- **Target Modules**: {', '.join(target_modules)}
- **Task Type**: Causal Language Modeling

## Training Details

### Dataset
- **Source**: asanchez75/medical_textbooks_mcq
- **Domain**: Medical multiple-choice questions
- **Language**: Primarily French medical content
- **Format**: JSON-structured input/output pairs
- **Size**: 1,481 examples (1,184 train, 148 validation, 149 test)

### Training Configuration
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 steps
- **Effective Batch Size**: 16
- **Sequence Length**: 2048 tokens
- **Hardware**: NVIDIA A100 SXM4 40GB

### Performance
- **Final Test Loss**: 0.7995
- **Training Time**: ~18.5 minutes (1,107 seconds)
- **Memory Usage**: 23.2GB peak (A100 40GB)
- **LoRA Memory Usage**: 7.59GB additional for training

## Usage

### Loading the Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "YOUR_HF_USERNAME/REPO_NAME")
```

### Inference Example

```python
import json

# Input format (medical context text)
input_text = "L'hypertension artérielle essentielle est une maladie chronique caractérisée par une pression artérielle élevée. Le traitement de première intention comprend les modifications du mode de vie et les médicaments antihypertenseurs."

# Format prompt using the same structure as training
prompt_prefix = "<|user|>\nContext:\n"
prompt_suffix = "\n\nGenerate ONE valid multiple-choice question based strictly on the context above. Output ONLY the valid JSON object representing the question.\nMCQ JSON:<|end|>\n<|assistant|>\n"

# Generate response
formatted_prompt = prompt_prefix + input_text + prompt_suffix
inputs = tokenizer(formatted_prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Expected Output Format

```json
{{
    "question": "Quel est le traitement de première intention de l'hypertension artérielle essentielle?",
    "options": {{
        "A": "Inhibiteurs de l'ECA",
        "B": "Bêta-bloquants",
        "C": "Diurétiques thiazidiques",
        "D": "Antagonistes calciques"
    }},
    "correct_answer": "A",
    "explanation": "Les inhibiteurs de l'ECA sont recommandés en première intention pour le traitement de l'hypertension artérielle essentielle selon les guidelines internationales."
}}
```

## Model Architecture

This adapter targets the following modules in the Meditron3-8B model:
- Query projection (q_proj)
- Key projection (k_proj) 
- Value projection (v_proj)
- Output projection (o_proj)
- Gate projection (gate_proj)
- Up projection (up_proj)
- Down projection (down_proj)

## Limitations and Biases

- **Domain Specific**: Optimized for French medical content
- **MCQ Format**: Designed for structured multiple-choice questions
- **Medical Focus**: Performance may vary on non-medical content
- **Language**: Primarily trained on French medical terminology

## Citation

If you use this model, please cite:

```bibtex
@misc{{meditron3-8b-mcq-lora,
    title={{Meditron3-8B LoRA Adapter for Medical MCQ JSON Generation}},
    author={{Your Name}},
    year={{2025}},
    publisher={{Hugging Face}},
    url={{https://huggingface.co/YOUR_USERNAME/REPO_NAME}}
}}
```

## License

This adapter is released under the Apache-2.0 license, consistent with the base Meditron3-8B model.
"""
    
    return model_card

def upload_adapter(adapter_path, repo_name, token, private, commit_message):
    """Upload the adapter to Hugging Face Hub."""
    
    try:
        # Initialize HF API
        api = HfApi(token=token)
        
        # Create repository
        logging.info(f"Creating repository: {repo_name}")
        try:
            create_repo(
                repo_id=repo_name,
                private=private,
                token=token,
                repo_type="model"
            )
            logging.info(f"✓ Repository {repo_name} created successfully")
        except HfHubHTTPError as e:
            if "already exists" in str(e):
                logging.info(f"Repository {repo_name} already exists, continuing...")
            else:
                raise e
        
        # Read adapter config for model card
        adapter_config = read_adapter_config(adapter_path)
        if not adapter_config:
            logging.error("Failed to read adapter configuration")
            return False
        
        # Training info from actual training results
        training_info = {
            "test_loss": 0.7995,
            "epochs": 3,
            "learning_rate": 2e-5,
            "training_time": "18.5 minutes",
            "peak_memory": "23.2 GB",
            "dataset_size": 1481
        }
        
        # Create model card
        model_card_content = create_model_card(adapter_config, training_info)
        
        # Write model card to adapter directory
        readme_path = Path(adapter_path) / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card_content)
        logging.info("✓ Model card created")
        
        # Upload all files in the adapter directory
        logging.info(f"Uploading adapter files from {adapter_path}")
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=repo_name,
            repo_type="model",
            token=token,
            commit_message=commit_message
        )
        
        logging.info(f"✅ Successfully uploaded adapter to https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logging.error(f"Error uploading adapter: {e}")
        return False

def main():
    args = parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        logging.error("No Hugging Face token provided. Use --token or set HF_TOKEN environment variable.")
        return
    
    # Validate adapter directory
    if not validate_adapter_directory(args.adapter_path):
        logging.error("Adapter validation failed")
        return
    
    logging.info(f"Uploading adapter from: {args.adapter_path}")
    logging.info(f"Target repository: {args.repo_name}")
    logging.info(f"Private repository: {args.private}")
    
    # Upload adapter
    success = upload_adapter(
        adapter_path=args.adapter_path,
        repo_name=args.repo_name,
        token=token,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        logging.info("🎉 Upload completed successfully!")
        logging.info(f"🔗 View your model at: https://huggingface.co/{args.repo_name}")
    else:
        logging.error("❌ Upload failed!")

if __name__ == "__main__":
    main()

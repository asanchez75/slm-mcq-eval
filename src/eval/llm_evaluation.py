from tqdm import tqdm
import os
import json
import re
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download, HfFolder

tqdm.pandas()

from pydantic import BaseModel

# Global variable to cache LoRA model and tokenizer
_lora_model_cache = {}

def download_lora_adapter(repo_id: str, local_path: str, hf_token: Optional[str] = None) -> bool:
    """Download a LoRA adapter from Hugging Face Hub."""
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    
    # Use token if provided or from environment
    token = hf_token or HfFolder.get_token() or os.getenv("HF_TOKEN")
    
    try:
        # Create local path if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=token,
            repo_type="model"
        )
        print(f"✓ LoRA adapter downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading LoRA adapter: {e}")
        return False

def validate_adapter_files(adapter_path: str) -> bool:
    """Validate that the downloaded adapter has required files."""
    adapter_dir = Path(adapter_path)
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    
    for file_name in required_files:
        file_path = adapter_dir / file_name
        if not file_path.exists():
            print(f"❌ Required file {file_name} not found in {adapter_path}")
            return False
    
    return True

def initialize_lora_model(
    base_model: str = "OpenMeditron/Meditron3-8B",
    lora_repo: str = None,
    local_adapter_cache: str = "./hf_adapter_cache",
    hf_token: Optional[str] = None,
    gpu_memory_utilization: float = 0.8,
    force_reload: bool = False
) -> Dict[str, Any]:
    """Initialize vLLM engine with LoRA support."""
    
    # Create cache key
    cache_key = f"{base_model}_{lora_repo}"
    
    # Return cached model if available and not forcing reload
    if cache_key in _lora_model_cache and not force_reload:
        print(f"✓ Using cached LoRA model: {cache_key}")
        return _lora_model_cache[cache_key]
    
    if not lora_repo:
        raise ValueError("lora_repo must be provided")
    
    print(f"🚀 Initializing vLLM engine for {base_model} with LoRA: {lora_repo}")
    
    # Setup adapter cache path
    adapter_cache_path = os.path.join(local_adapter_cache, lora_repo.replace('/', '_'))
    
    # Download LoRA adapter if not cached
    if not os.path.exists(adapter_cache_path) or not validate_adapter_files(adapter_cache_path):
        if not download_lora_adapter(lora_repo, adapter_cache_path, hf_token):
            raise RuntimeError(f"Failed to download LoRA adapter: {lora_repo}")
        
        if not validate_adapter_files(adapter_cache_path):
            raise RuntimeError(f"Invalid LoRA adapter files in: {adapter_cache_path}")
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("⚠️  WARNING: CUDA not available, running on CPU")
        effective_gpu_util = 0
    else:
        effective_gpu_util = gpu_memory_utilization
        print(f"✓ CUDA available. GPU memory utilization: {effective_gpu_util}")
    
    try:
        # Initialize vLLM engine
        llm = LLM(
            model=base_model,
            max_model_len=2048,
            dtype="bfloat16",
            gpu_memory_utilization=effective_gpu_util,
            trust_remote_code=True,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
            disable_log_stats=True,
            enforce_eager=True,
            disable_custom_all_reduce=True
        )
        
        # Get tokenizer
        tokenizer = llm.get_tokenizer()
        
        # Create LoRA request
        lora_request = LoRARequest(
            lora_name="medical_mcq_adapter",
            lora_int_id=1,
            lora_local_path=adapter_cache_path
        )
        
        # Cache the model components
        model_components = {
            "llm": llm,
            "tokenizer": tokenizer,
            "lora_request": lora_request,
            "adapter_path": adapter_cache_path
        }
        
        _lora_model_cache[cache_key] = model_components
        print("✅ vLLM engine with LoRA initialized successfully")
        
        return model_components
        
    except Exception as e:
        print(f"❌ Error initializing vLLM engine: {e}")
        raise

def parse_lora_output(output_text: str) -> Optional[str]:
    """Parse the output from LoRA model, extracting relevant content."""
    try:
        # Try to find JSON object in the response
        json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            result = json.loads(json_text)
            # Return specific field or whole result based on use case
            return result.get("answer", str(result))
        else:
            # If no JSON, return cleaned text
            return output_text.strip()
    except json.JSONDecodeError:
        return output_text.strip()
    except Exception as e:
        print(f"⚠️  Error parsing LoRA output: {e}")
        return output_text.strip()

def call_openai_api(client, system_prompt, user_prompt, temp=0.5, max_completion_tokens = 1):
    """
    Unified API call function that handles both OpenAI and LoRA models.
    
    Args:
        client: Either OpenAI client or LoRA components dict
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        temp: Temperature for sampling
        max_completion_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated response or None if error
    """
    try:
        # Check if client is LoRA components (dict) or OpenAI client
        if isinstance(client, dict) and "llm" in client:
            # Handle LoRA model
            llm = client["llm"]
            tokenizer = client["tokenizer"]
            lora_request = client["lora_request"]
            
            # Create chat format matching fine-tuning
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Define sampling parameters
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_completion_tokens,
                stop=["</s>", "<|im_end|>", "<|end|>"]
            )
            
            # Generate response
            outputs = llm.generate(
                [formatted_prompt],
                sampling_params,
                lora_request=lora_request
            )
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            # Parse and return response
            parsed_response = parse_lora_output(generated_text)
            return parsed_response
            
        else:
            # Handle OpenAI client (original behavior)
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=temp,
                max_completion_tokens=max_completion_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def generate_prompt_for_question(row,
                                 question_col='question',
                                 option_a_col = 'option_a',
                                 option_b_col = 'option_b',
                                 option_c_col = 'option_c',
                                 option_d_col = 'option_d',
                                 correct_option = 'correct_option',
                                 include_options=True,
                                 include_correct_option = True,
                                 context_col=None):
    question_text = row[question_col]
    options = f"a) {row[option_a_col]}\nb) {row[option_b_col]}\nc) {row[option_c_col]}\nd) {row[option_d_col]}"
    correct_option = row[correct_option]
    
    user_prompt_delimiter = "-----\n"
    user_prompt_question = f"Question:\n{question_text}\n"
    user_prompt_options = f"Options:\n{options}\n"

    user_prompt = user_prompt_delimiter + user_prompt_question
    if include_options:
        user_prompt += user_prompt_options
    if include_correct_option:
        correct_option = f"Correct option: {correct_option}\n"
        user_prompt += correct_option

    user_prompt += user_prompt_delimiter
    
    if context_col is not None:
        mcq_context = f"""Context:\n-----\n{row[context_col]}\n-----\n"""
        user_prompt = mcq_context + user_prompt

    return user_prompt


def get_lora_config_from_env():
    """Get LoRA configuration from environment variables."""
    return {
        "base_model": os.getenv("LORA_BASE_MODEL", "OpenMeditron/Meditron3-8B"),
        "lora_repo": os.getenv("LORA_REPO", None),
        "hf_token": os.getenv("HF_TOKEN", None),
        "local_adapter_cache": os.getenv("LORA_ADAPTER_CACHE", "./hf_adapter_cache"),
        "gpu_memory_utilization": float(os.getenv("LORA_GPU_MEMORY", "0.8")),
        "temperature": float(os.getenv("LORA_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("LORA_MAX_TOKENS", "1024"))
    }

def process_dataframe(model_name, df):
    try:
        # df['rank'] = df.progress_apply(generate_prompt_for_question, axis=1)
        df.to_csv(f'/kaggle/working/results_of_{model_name}.csv', index=False)
    except Exception as e:
        print(f"Error occurred for model {model_name}: {e}")
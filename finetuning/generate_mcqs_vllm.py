# generate_mcqs_vllm.py
import json
import argparse
import sys
import time
import os
import re
from typing import List, Dict, Optional, Tuple

# Import vLLM classes
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Import huggingface_hub for downloading adapter
from huggingface_hub import snapshot_download, HfFolder

# Import utility functions for loading data (assuming utils.py exists)
try:
    from utils import load_test_set, get_all_txt_contents_from_folders
except ImportError:
    sys.exit("ERROR: Could not import 'load_test_set' and 'get_all_txt_contents_from_folders' from 'utils.py'. Ensure the file exists and is in the Python path.")

# Import configuration (for DEFAULT_PROMPT)
try:
    from config import DEFAULT_PROMPT
except ImportError:
    print("WARNING: Could not import DEFAULT_PROMPT from config.py. Using a fallback prompt.")
    DEFAULT_PROMPT = """Based on the following educational content, generate a multiple-choice question with four answer options where only one is correct. The question and its options must adhere to the following rule: The incorrect options (distractors) should be plausible and logically related to the question."""

# Import Pydantic model for structure definition and validation helper
from pydantic import BaseModel, Field, ValidationError

import torch # For checking cuda availability
from tqdm import tqdm

# --- Pydantic Model for MCQ Structure ---
class MCQQuestion(BaseModel):
    """Defines the expected structure of the generated MCQ"""
    question: str = Field(description="The multiple-choice question")
    option_a: str = Field(description="The first answer option labeled 'A'")
    option_b: str = Field(description="The second answer option labeled 'B'")
    option_c: str = Field(description="The third answer option labeled 'C'")
    option_d: str = Field(description="The fourth answer option labeled 'D'")
    correct_option: str = Field(description="This consists only a letter (A, B, C, or D) of the correct option")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate MCQs from text content using vLLM offline LoRA inference.")

# Data Paths
parser.add_argument('--lisa_path', type=str, default="../../../../data/lisa_sheets/", help="Path to the parent directory containing content folders (e.g., lisa_sheets).")
parser.add_argument('--test_folders_path', type=str, default="../../../../data/train_test_split/test_folders.json", help="Path to the JSON file listing folders to include in the test set.")

# Model and LoRA Configuration
parser.add_argument('--lora_repo_id', type=str, default="asanchez75/meditron3-8b-mcq-generation-lora", help="Hugging Face Repo ID for the LoRA adapter (e.g., 'your-username/your-lora-adapter').")
parser.add_argument('--base_model_id', type=str, default="OpenMeditron/Meditron3-8B", help="Hugging Face Repo ID for the base model.")
parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face token if needed for private models/adapters.")

# vLLM Engine Configuration
parser.add_argument('--quantization', type=str, default=None, help="Quantization method (e.g., bitsandbytes, awq, gptq, None).")
parser.add_argument('--max_model_len', type=int, default=8192, help="Max model length (prompt + output). Adjust based on expected context + MCQ length.")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.85, help="GPU memory utilization fraction for vLLM.")
parser.add_argument('--max_lora_rank', type=int, default=64, help="Maximum LoRA rank anticipated.")

# Generation Parameters
parser.add_argument('-t', "--temperature", type=float, default=0.5, help="Sampling temperature for the LLM.")
parser.add_argument('--max_tokens', type=int, default=300, help="Maximum number of tokens to generate for the MCQ.") # Increased for MCQ generation

# Output Configuration
parser.add_argument('-o', "--output_dir", type=str, default='./generated_mcqs_vllm', help="Directory to save output JSON file and logs.")
parser.add_argument('--limit', type=int, default=None, help="Limit the number of text items to process (for testing).")

args = parser.parse_args()

# --- Configuration ---
path_lisa = args.lisa_path
path_test_folders = args.test_folders_path
base_model_identifier = args.base_model_id
lora_repo_id = args.lora_repo_id
quantization_method = args.quantization
max_model_len = args.max_model_len
gpu_memory_utilization = args.gpu_memory_utilization
dtype_setting = "bfloat16" # Use bf16 instead of auto
max_lora_rank = args.max_lora_rank

# Construct output filenames
output_lora_name = lora_repo_id.split('/')[-1] # Get adapter name for filename
output_base_filename = f'generated_mcqs_vllm_{output_lora_name}_temp{args.temperature}'
exceptions_base_filename = f'exceptions_vllm_{output_lora_name}_temp{args.temperature}'

output_file = os.path.join(args.output_dir, output_base_filename + '.json')
exceptions_file = os.path.join(args.output_dir, exceptions_base_filename + '.json')

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# --- Function Definition: Robust JSON MCQ Extraction ---
def extract_mcq_json(response_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extracts a JSON object representing an MCQ from the response text.

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        A tuple containing:
        - dict: The parsed and validated MCQ data if successful.
        - None: If parsing or validation fails.
        And:
        - None: If successful.
        - str: An error message describing the failure reason.
    """
    if not response_text:
        return None, "ERROR_EMPTY_RESPONSE"

    response_text = response_text.strip()

    # Try to find JSON object (match ```json ... ``` code blocks first, then bare {})
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL) # Fallback to bare {}

    if json_match:
        json_part = json_match.group(1)
        try:
            data = json.loads(json_part)
            # Validate against Pydantic model
            mcq = MCQQuestion(**data)
             # Additional validation: Ensure correct_option is a single valid letter
            correct_opt_upper = mcq.correct_option.strip().upper()
            if correct_opt_upper not in ['A', 'B', 'C', 'D']:
                 return None, f"PARSE_ERROR_INVALID_CORRECT_OPTION_VALUE: '{mcq.correct_option}'"
            # Return validated data as dict, ensure correct option is uppercase
            mcq_dict = mcq.dict()
            mcq_dict['correct_option'] = correct_opt_upper
            return mcq_dict, None
        except json.JSONDecodeError as e:
            error_type = "PARSE_ERROR_INVALID_JSON"
            error_detail = f"{e} in snippet: {json_part[:100]}..."
        except ValidationError as e:
            error_type = "PARSE_ERROR_VALIDATION_FAILED"
            # Format Pydantic error for better readability
            error_detail = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        except Exception as e:
            error_type = f"PARSE_ERROR_UNEXPECTED_{type(e).__name__}"
            error_detail = str(e)[:100]
        # Fall through if parsing/validation failed
    else:
         error_type = "PARSE_ERROR_NO_JSON_FOUND"
         error_detail = response_text[:100] # Snippet of text

    # If we reached here, parsing failed
    # print(f"DEBUG: Failed to parse: {error_type} - {error_detail}") # Uncomment for debugging
    return None, f"{error_type}: {error_detail}"


# --- Download/Locate LoRA Adapter ---
local_lora_path = os.path.join(".", "lora_adapters", output_lora_name) # Use local relative path
print(f"Ensuring LoRA adapter '{lora_repo_id}' is available at '{local_lora_path}'...")
try:
    # Use token if provided via argument or environment variable HUGGING_FACE_HUB_TOKEN
    token = args.hf_token or HfFolder.get_token()
    snapshot_download(
        repo_id=lora_repo_id,
        local_dir=local_lora_path,
        local_dir_use_symlinks=False, # Avoid symlinks for broader compatibility
        token=token
    )
    print("LoRA adapter download/check complete.")
except Exception as e:
    sys.exit(f"ERROR: Failed to download/access LoRA adapter '{lora_repo_id}'. Error: {e}")

# --- Initialize vLLM Engine ---
print(f"Initializing vLLM engine for base model: {base_model_identifier} with LoRA enabled")
print(f"Quantization: {quantization_method}, Max Length: {max_model_len}, GPU Util: {gpu_memory_utilization}")
try:
    # Check if CUDA is available before setting gpu_memory_utilization
    effective_gpu_memory_utilization = gpu_memory_utilization if torch.cuda.is_available() else 0
    if effective_gpu_memory_utilization == 0 and torch.cuda.is_available():
         print("WARNING: CUDA is available, but gpu_memory_utilization is set to 0. Running on CPU?")
    elif not torch.cuda.is_available():
         print("INFO: CUDA not available, vLLM will run on CPU (if supported) or fail.")
         effective_gpu_memory_utilization = 0 # Ensure it's 0 if no CUDA

    llm = LLM(
        model=base_model_identifier,
        quantization=quantization_method if quantization_method and quantization_method.lower() != 'none' else None,
        max_model_len=max_model_len,
        dtype=dtype_setting,
        gpu_memory_utilization=effective_gpu_memory_utilization,
        trust_remote_code=True, # Often needed for models like Phi-3 or custom code
        enable_lora=True,       # Enable LoRA support
        max_loras=1,            # Max concurrent LoRAs needed for this script
        max_lora_rank=max_lora_rank # Max rank expected
    )
    tokenizer = llm.get_tokenizer()
    print("vLLM Engine Initialized with LoRA support.")
except Exception as e:
    sys.exit(f"ERROR: Failed to initialize vLLM engine: {type(e).__name__} - {e}")


# --- Load Source Data ---
print("Loading source text data...")
try:
    all_txt_contents = get_all_txt_contents_from_folders(path_lisa)
    print(f"Found {len(all_txt_contents)} text items.")
    dataset = load_test_set(all_txt_contents, path_folders=path_test_folders)
    print(f"Loaded {len(dataset)} items for the test set based on {path_test_folders}")
except FileNotFoundError as e:
    sys.exit(f"ERROR: Data file or folder not found: {e}")
except Exception as e:
    sys.exit(f"ERROR: Failed to load data: {e}")

if not dataset:
    sys.exit("ERROR: No data loaded into the test set. Check paths and filter file.")

# Apply limit if specified
if args.limit is not None and args.limit > 0:
    dataset = dataset[:args.limit]
    print(f"Processing limited to the first {len(dataset)} items.")


# --- Prepare Prompts ---
# Use the default prompt from config and add instruction for JSON output based on Pydantic model
# Get format instructions based on the Pydantic model to guide the LLM
try:
    # Use JsonOutputParser to generate format instructions based on the Pydantic model
    from langchain_core.output_parsers import JsonOutputParser
    mcq_parser = JsonOutputParser(pydantic_object=MCQQuestion)
    format_instructions = mcq_parser.get_format_instructions()
except ImportError:
    print("WARNING: Langchain not found. Using basic format instructions.")
    format_instructions = """Return ONLY a valid JSON object matching this structure:
{
    "question": "The question text",
    "option_a": "Option A text",
    "option_b": "Option B text",
    "option_c": "Option C text",
    "option_d": "Option D text",
    "correct_option": "A"
}
Replace the example values with the generated content. Ensure the 'correct_option' is only the letter A, B, C, or D."""

prompt_template_string = """Context:
{context}

Generate ONE valid multiple-choice question based strictly on the context above. Output ONLY the valid JSON object representing the question.
MCQ JSON:"""

prompts_list = []
source_data_map = [] # Store original data to map results back

print("Preparing prompts...")
for idx, item in enumerate(dataset):
    context = item.get('content', '')
    folder = item.get('folder', 'unknown_folder')

    if not context.strip():
        print(f"WARNING: Skipping item {idx} from folder '{folder}' due to empty content.")
        continue

    # Use the updated prompt template matching the fine-tuning format
    prompt_content = prompt_template_string.format(
        context=context
    )

    # Apply Meditron3-8B (Llama 3.1) chat template to match fine-tuning format
    messages = [
        {"role": "user", "content": prompt_content}
    ]

    try:
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Ensures the prompt ends correctly for generation
        )
        prompts_list.append(final_prompt)
        # ***MODIFICATION START***: Store full original content and folder
        source_data_map.append({
            "id": idx,
            "folder": folder,
            "content": context # Store the full content
        })
        # ***MODIFICATION END***
    except Exception as e:
        print(f"WARNING: Failed to apply chat template for item {idx} from folder '{folder}'. Skipping. Error: {e}")

# --- Define Sampling Parameters ---
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens, # Use arg for max tokens
    # top_p=0.9, # Optionally add other params
    stop=["```"] # Optionally add stop sequences if the model tends to add extra ```
)

# --- Create LoRA Request ---
lora_request = LoRARequest(
    lora_name="mcq_generator_adapter", # Give your request a name
    lora_int_id=1,                     # Give it a unique integer ID for this run
    lora_local_path=local_lora_path    # Path where adapter was downloaded
)

# --- Perform Batch Inference ---
results = []
exceptions_log = []
total_inference_time = 0

if not prompts_list:
    sys.exit("ERROR: No valid prompts were prepared. Exiting.")

print(f"\nStarting batch inference for {len(prompts_list)} prompts using LoRA: {lora_request.lora_name} from {lora_request.lora_local_path}")
try:
    start_time = time.time()
    # Generate responses using vLLM
    outputs = llm.generate(
        prompts_list,
        sampling_params,
        lora_request=lora_request # Pass the LoRA request
    )
    total_inference_time = time.time() - start_time
    print(f"Batch inference completed in {total_inference_time:.2f} seconds.")

    # --- Process Results ---
    print("Processing results...")
    if len(outputs) != len(source_data_map):
          print(f"WARNING: Mismatch between number of outputs ({len(outputs)}) and source map ({len(source_data_map)}). Results might be misaligned.")

    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
        # ***MODIFICATION START***: Use full original data
        original_data = source_data_map[i] if i < len(source_data_map) else {'id': f'UNKNOWN_ID_{i}', 'folder': 'unknown', 'content': 'N/A'}
        # ***MODIFICATION END***
        llm_response_text = output.outputs[0].text

        # Parse the response using the robust JSON extractor
        parsed_mcq, error_message = extract_mcq_json(llm_response_text)

        if parsed_mcq:
            # ***MODIFICATION START***: Append result in the desired format
            results.append({
                "folder": original_data['folder'],
                "content": original_data['content'],
                "question": parsed_mcq # parsed_mcq is already the dict for the question structure
            })
            # ***MODIFICATION END***
        else:
            # Log parsing errors/failures
            # ***MODIFICATION START***: Ensure error log has consistent keys if needed later
            exceptions_log.append({
                'source_id': original_data['id'],
                'source_folder': original_data['folder'],
                'error_type': 'Parsing Error',
                'error_message': error_message,
                'llm_raw_output': llm_response_text.strip(),
                'original_content': original_data['content'] # Log original content with error
            })
            # ***MODIFICATION END***

except Exception as e:
    error_message = f"INFERENCE_ERROR: {type(e).__name__} - {str(e)}"
    print(f"\n{error_message}")
    # Log the main inference error
    exceptions_log.append({
        'id': 'BATCH_INFERENCE_FAILURE',
        'error': str(e),
        'type': type(e).__name__
    })
    if results:
        print("Attempting to save partial results...")
    else:
        print("No results generated before the error.")


# --- Save Results ---
print(f"\nSaving {len(results)} successfully generated MCQs to {output_file}...")
try:
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False) # Use indent=2 to match example
except IOError as e:
    print(f"ERROR: Could not write results to {output_file}: {e}")

# --- Save Exceptions Log ---
if exceptions_log:
    print(f"Saving {len(exceptions_log)} exceptions/parsing errors to {exceptions_file}...")
    try:
        with open(exceptions_file, 'w', encoding='utf-8') as file:
            json.dump(exceptions_log, file, indent=2, ensure_ascii=False) # Use indent=2
    except IOError as e:
        print(f"ERROR: Could not write exceptions log to {exceptions_file}: {e}")

# --- Calculate and Display Summary ---
total_processed_prompts = len(prompts_list)
successful_generations = len(results)
failed_parses = len(exceptions_log) - (1 if any(e.get('id') == 'BATCH_INFERENCE_FAILURE' for e in exceptions_log) else 0) # Count only parsing errors

print("-" * 50)
print("MCQ Generation Summary")
print("-" * 50)
print(f"Processed {total_processed_prompts} text items (after filtering/limiting).")
print(f"Total Inference Time: {total_inference_time:.2f} seconds")
if total_processed_prompts > 0:
      print(f"Average Time per Item: {total_inference_time / total_processed_prompts:.4f} seconds")
print(f"Successfully generated and parsed MCQs: {successful_generations}")
print(f"Failed to parse LLM output (or empty): {failed_parses}")
if any(e.get('id') == 'BATCH_INFERENCE_FAILURE' for e in exceptions_log):
     print("WARNING: A batch inference error occurred. Some items may not have been processed.")

success_rate = (successful_generations / total_processed_prompts * 100) if total_processed_prompts > 0 else 0
print(f"Success Rate (Parsed / Processed): {success_rate:.2f}%")
print("-" * 50)
print(f"Results saved to: {output_file}")
if exceptions_log:
    print(f"Errors/Exceptions saved to: {exceptions_file}")
print("Script finished.")

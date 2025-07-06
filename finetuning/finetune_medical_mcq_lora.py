# -*- coding: utf-8 -*-
"""
Simplified fine-tuning script 
for Medical MCQ JSON Generation.

Hardcoded parameters, splits data (80/10/10), trains LoRA, saves adapters locally.
Evaluates on validation set during training and on test set after training.
Logs saved to file.
"""

import os
import torch
from datasets import load_dataset, Dataset # Import Dataset for type hints if needed
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
from pathlib import Path
import json # Added for saving test results

# Set environment variables for stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single A100 GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Configuration Constants ---
MODEL_NAME = "OpenMeditron/Meditron3-8B"
DATASET_NAME = "asanchez75/medical_textbooks_mcq"
MAX_SEQ_LENGTH = 2048  # Increased for A100 40GB
LOAD_IN_4BIT = False  # Full precision with A100 40GB
# LoRA configurations
LORA_R = 64  # Increased for better capacity with A100
LORA_ALPHA = 128  # Scaled alpha for higher rank
# Training configurations
NUM_TRAIN_EPOCHS = 3  # Increased for proper convergence
MAX_STEPS = -1 # Set > 0 to override epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Increased for A100 40GB
GRADIENT_ACCUMULATION_STEPS = 4  # Reduced while maintaining effective batch size
LEARNING_RATE = 2e-5  # Optimized LR for medical domain fine-tuning
# Evaluation/Logging/Saving during training
EVAL_STEPS = 50  # Evaluate every N steps
LOGGING_STEPS = 10 # Log loss every N steps
SAVE_STEPS = 100   # Save checkpoint every N steps (can align with EVAL_STEPS)
# Dataset split ratios
TEST_VALID_SIZE = 0.2 # Combined size for validation + test (e.g., 20%)
TEST_SIZE_FROM_VALID = 0.5 # Proportion of the TEST_VALID_SIZE to use for test (e.g., 0.5 means 10% test, 10% validation)
RANDOM_SEED = 42 # For reproducible splits
# Output Directories
OUTPUT_BASE_DIR = Path(f"/workspace")
CHECKPOINT_DIR = OUTPUT_BASE_DIR / "checkpoints"
FINAL_ADAPTER_DIR = OUTPUT_BASE_DIR / "final_lora_adapter"
LOG_DIR = OUTPUT_BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "finetune.log"

# --- Create Output Directories ---
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("datasets").setLevel(logging.INFO)
logging.getLogger("trl").setLevel(logging.INFO)

# --- Data Loading Function ---
def load_huggingface_dataset(dataset_name):
    """Loads data from HuggingFace dataset hub."""
    logging.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
    
    try:
        # Load the dataset from HuggingFace Hub
        dataset = load_dataset(dataset_name, split="train")
        logging.info(f"Successfully loaded dataset '{dataset_name}' with {len(dataset)} examples.")
        logging.info(f"Dataset features: {dataset.features}")
        
        # Rename 'mcq_question' to 'output_json_str' and 'content' to 'input_json_str' for consistency
        if "mcq_question" in dataset.column_names:
            dataset = dataset.rename_column("mcq_question", "output_json_str")
            logging.info("Renamed column 'mcq_question' to 'output_json_str'.")
        
        if "content" in dataset.column_names:
            dataset = dataset.rename_column("content", "input_json_str")
            logging.info("Renamed column 'content' to 'input_json_str'.")
        
        logging.info(f"Dataset features after renaming: {dataset.features}")
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
        
        return dataset
        
    except Exception as e:
        logging.error(f"Failed to load dataset from HuggingFace Hub: {e}")
        raise e

# --- Main Fine-tuning Logic ---

def run_finetuning():
    """Performs the fine-tuning process with data splitting and evaluation."""

    logging.info("Starting fine-tuning script with evaluation...")
    # Log configuration constants (can be expanded)
    logging.info(f"Base Model: {MODEL_NAME}, Dataset: {DATASET_NAME}")
    logging.info(f"Output Dir: {OUTPUT_BASE_DIR}")

    # --- Load Model and Tokenizer ---
    dtype = torch.bfloat16
    logging.info(f"Loading base model: {MODEL_NAME}")
    
    # Load model in full precision for A100 40GB
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Add LoRA Adapters ---
    logging.info("Adding LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    logging.info("LoRA adapters added.")

    # --- Data Preparation ---
    logging.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_huggingface_dataset(DATASET_NAME)
    logging.info(f"Initial dataset size: {len(dataset)}")

    # Split dataset into Train, Validation, Test
    logging.info(f"Splitting dataset: Train={1-TEST_VALID_SIZE:.0%}, Validation/Test={TEST_VALID_SIZE:.0%}")
    try:
        train_val_split = dataset.train_test_split(test_size=TEST_VALID_SIZE, seed=RANDOM_SEED)
        train_dataset = train_val_split['train']
        temp_dataset = train_val_split['test']

        logging.info(f"Splitting Validation/Test set: Validation={1-TEST_SIZE_FROM_VALID:.0%}, Test={TEST_SIZE_FROM_VALID:.0%}")
        val_test_split = temp_dataset.train_test_split(test_size=TEST_SIZE_FROM_VALID, seed=RANDOM_SEED)
        validation_dataset = val_test_split['train']
        test_dataset = val_test_split['test']

        logging.info(f"Final split sizes: Train={len(train_dataset)}, Validation={len(validation_dataset)}, Test={len(test_dataset)}")
    except Exception as e:
        logging.error(f"Failed to split dataset: {e}. Ensure dataset has enough examples.", exc_info=True)
        return

    # Apply formatting to all splits - process one example at a time
    logging.info("Formatting datasets...")
    
    # Use proper Meditron3-8B (Llama 3.1) chat template
    prompt_prefix = "<|start_header_id|>user<|end_header_id|>\n\nContext:\n"
    prompt_suffix = "\n\nGenerate ONE valid multiple-choice question based strictly on the context above. Output ONLY the valid JSON object representing the question.\nMCQ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    end_token = tokenizer.eos_token
    
    def format_single_example(example):
        input_json = example["input_json_str"]
        output_json = example["output_json_str"]
        
        # Handle both string and dict formats for output_json
        if isinstance(output_json, dict):
            output_json = json.dumps(output_json, ensure_ascii=False, indent=None)
        
        if isinstance(input_json, str) and len(input_json) > 0 and \
           isinstance(output_json, str) and len(output_json) > 0:
            # Use the same prompt format as train_phi3_json_lora_4bit.py
            formatted_text = prompt_prefix + input_json + prompt_suffix + output_json + end_token
            return {"text": formatted_text}
        else:
            return {"text": ""}  # Empty text for invalid examples
    
    # Process datasets one example at a time
    train_dataset = train_dataset.map(format_single_example, remove_columns=list(train_dataset.features))
    validation_dataset = validation_dataset.map(format_single_example, remove_columns=list(validation_dataset.features))
    test_dataset = test_dataset.map(format_single_example, remove_columns=list(test_dataset.features))
    
    # Filter out empty examples
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0)
    validation_dataset = validation_dataset.filter(lambda x: len(x["text"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) > 0)

    logging.info("Dataset mapping complete.")
    logging.info(f"Formatted dataset sizes: Train={len(train_dataset)}, Validation={len(validation_dataset)}, Test={len(test_dataset)}")
    if len(train_dataset) == 0:
        logging.error("Training dataset is empty after formatting! Cannot proceed.")
        return


    # --- Train the Model ---
    training_args = TrainingArguments(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,  # Same as training batch size
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=0.1,            # 10% warmup for better stability
            num_train_epochs=NUM_TRAIN_EPOCHS if MAX_STEPS <= 0 else 0,
            max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
            learning_rate=LEARNING_RATE,
            fp16=False,  # Don't use FP16 with A100
            bf16=True,   # Use BF16 for A100 - excellent support
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS,
            eval_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",       # Save every epoch to match eval strategy
            load_best_model_at_end=True, # Load best model for final evaluation
            metric_for_best_model="eval_loss",
            greater_is_better=False,     # Lower loss is better
            optim="adamw_torch",  # Use standard AdamW for large models
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=RANDOM_SEED,
            output_dir=str(CHECKPOINT_DIR),
            report_to="none", # Change for WandB/Tensorboard
            ddp_find_unused_parameters=False,  # Optimize for multi-GPU
            dataloader_pin_memory=False,  # Disable pin memory to reduce CUDA errors
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            dataloader_num_workers=0,  # Reduce workers to avoid CUDA context issues
            remove_unused_columns=False,  # Keep all columns to avoid data issues
        )

    # Create a custom data collator for proper tokenization
    from transformers import DataCollatorForLanguageModeling
    
    def tokenize_function(example):
        # Tokenize single example (not batched)
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize the datasets (single examples, not batched)
    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    
    # Custom data collator for causal LM that handles labels properly
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    
    @dataclass
    class DataCollatorForCausalLM:
        tokenizer: Any
        padding: Union[bool, str] = True
        max_length: int = None
        pad_to_multiple_of: int = None
        return_tensors: str = "pt"
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Extract input_ids and labels separately
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]
            
            # Pad input_ids
            batch = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            
            # Pad labels manually to match input_ids length
            max_length = batch["input_ids"].shape[1]
            padded_labels = []
            for label_seq in labels:
                if len(label_seq) < max_length:
                    # Pad with -100 (ignore token for loss calculation)
                    padded_labels.append(label_seq + [-100] * (max_length - len(label_seq)))
                else:
                    padded_labels.append(label_seq[:max_length])
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            return batch
    
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset, # Pass validation set here
        args=training_args,
        data_collator=data_collator,
    )

    # Verify memory usage
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logging.info(f"{start_gpu_memory} GB of memory reserved before training.")

    logging.info("Starting training...")
    trainer_stats = trainer.train()
    logging.info("Training finished.")

    # Log final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    train_runtime = trainer_stats.metrics.get('train_runtime', 0)
    logging.info(f"{train_runtime:.2f} seconds used for training.")
    logging.info(f"Peak reserved memory = {used_memory} GB.")
    logging.info(f"Peak reserved memory for training (LoRA) = {used_memory_for_lora} GB.")

    # --- Evaluate on Test Set ---
    # The trainer automatically loads the best model checkpoint if load_best_model_at_end=True
    if test_dataset and len(test_dataset) > 0:
        logging.info("Evaluating the best model on the test set...")
        try:
            test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test") # Add prefix
            logging.info(f"Test Set Evaluation Results: {test_results}")
            # Save test results
            test_results_file = LOG_DIR / "test_results.json"
            with open(test_results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            logging.info(f"Test results saved to {test_results_file}")
        except Exception as e:
            logging.error(f"Failed to evaluate on test set: {e}", exc_info=True)
    else:
        logging.warning("Test dataset is empty or not available, skipping final test evaluation.")

    # --- Saving the Final LoRA Model ---
    logging.info(f"Saving final LoRA adapters locally to {FINAL_ADAPTER_DIR}")
    try:
        # Save using PEFT's save_pretrained method
        model.save_pretrained(str(FINAL_ADAPTER_DIR))
        tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
        logging.info(f"Successfully saved final LoRA adapters to {FINAL_ADAPTER_DIR}")
    except Exception as e:
        logging.error(f"Failed to save final LoRA adapters: {e}", exc_info=True)

    logging.info("Script finished.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This script requires a GPU.")
    else:
        run_finetuning()



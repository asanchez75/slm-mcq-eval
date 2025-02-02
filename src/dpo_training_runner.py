import os
import gc
import torch
import argparse
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


class DPOTrainingRunner:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def create_pair(self, chosen_row, rejected_row):
        chosen_text = (
            f"Question: {chosen_row['question']}\n"
            f"a) {chosen_row['option_a']}\n"
            f"b) {chosen_row['option_b']}\n"
            f"c) {chosen_row['option_c']}\n"
            f"d) {chosen_row['option_d']}"
        )
        rejected_text = (
            f"Question: {rejected_row['question']}\n"
            f"a) {rejected_row['option_a']}\n"
            f"b) {rejected_row['option_b']}\n"
            f"c) {rejected_row['option_c']}\n"
            f"d) {rejected_row['option_d']}"
        )
        return {"chosen": chosen_text, "rejected": rejected_text}
    
    def load_datasets(self):
        chosen_df = pd.read_csv(self.args.chosen_csv)
        rejected_df = pd.read_csv(self.args.rejected_csv)
        
        eval_chosen_indices = chosen_df.sample(n=self.args.eval_size, random_state=self.seed).index
        eval_rejected_indices = rejected_df.sample(n=self.args.eval_size, random_state=self.seed).index
        
        eval_chosen_df = chosen_df.loc[eval_chosen_indices].reset_index(drop=True)
        eval_rejected_df = rejected_df.loc[eval_rejected_indices].reset_index(drop=True)
        
        train_chosen_df = chosen_df.drop(eval_chosen_indices).reset_index(drop=True)
        train_rejected_df = rejected_df.drop(eval_rejected_indices).reset_index(drop=True)
        
        eval_examples = []
        for i in range(len(eval_chosen_df)):
            chosen_row = eval_chosen_df.iloc[i]
            rejected_row = eval_rejected_df.iloc[i]
            eval_examples.append(self.create_pair(chosen_row, rejected_row))
        eval_df = pd.DataFrame(eval_examples)
        
        if len(train_chosen_df) < len(train_rejected_df):
            upsampled_train_chosen_df = train_chosen_df.sample(
                n=len(train_rejected_df), replace=True, random_state=self.seed
            ).reset_index(drop=True)
        else:
            upsampled_train_chosen_df = train_chosen_df.copy()
        
        train_rejected_df = train_rejected_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        upsampled_train_chosen_df = upsampled_train_chosen_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        training_examples = []
        for i in range(len(upsampled_train_chosen_df)):
            chosen_row = upsampled_train_chosen_df.iloc[i]
            rejected_row = train_rejected_df.iloc[i % len(train_rejected_df)]
            training_examples.append(self.create_pair(chosen_row, rejected_row))
        train_df = pd.DataFrame(training_examples)
        
        eval_dataset = Dataset.from_pandas(eval_df)
        train_dataset = Dataset.from_pandas(train_df)
        return train_dataset, eval_dataset
    
    def setup_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(self.args.model_name, quantization_config=bnb_config)
        model.config.use_cache = False
        return model, tokenizer
    
    def train(self):
        train_dataset, eval_dataset = self.load_datasets()
        model, tokenizer = self.setup_model()
        
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias=self.args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=self.args.lora_target_modules
        )
        
        training_args = DPOConfig(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=self.args.gradient_checkpointing,
            remove_unused_columns=self.args.remove_unused_columns,
            learning_rate=self.args.learning_rate,
            evaluation_strategy=self.args.evaluation_strategy,
            logging_strategy=self.args.logging_strategy,
            lr_scheduler_type=self.args.lr_scheduler_type,
            num_train_epochs=self.args.num_train_epochs,
            save_strategy=self.args.save_strategy,
            logging_steps=self.args.logging_steps,
            output_dir=self.args.output_dir,
            optim=self.args.optim,
            warmup_steps=self.args.warmup_steps,
            bf16=self.args.bf16,
            report_to=self.args.report_to,
        )
        
        dpo_trainer = DPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
            beta=self.args.beta,
            max_prompt_length=self.args.max_prompt_length,
            max_length=self.args.max_length,
        )
        
        dpo_trainer.train()
        dpo_trainer.save_model(self.args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a DPO model with command-line parameters.")
    # Data paths and sizes
    parser.add_argument("--chosen_csv", type=str, default="ar-dataset-ft.csv")
    parser.add_argument("--rejected_csv", type=str, default="ar-dataset.csv")
    parser.add_argument("--eval_size", type=int, default=150)
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_dir", type=str, default="model")
    # General training hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--remove_unused_columns", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5.0e-06)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=2048)
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    # DPO parameter
    parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()
    runner = DPOTrainingRunner(args)
    runner.train()

if __name__ == "__main__":
    main()
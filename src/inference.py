import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os


SYSTEM_INSTRUCTION = """
You are Phi3.5, a highly capable and fine-tuned AI assistant specialized in generating high-quality multiple-choice questions (MCQs). 
Your main role is to assist educators, trainers, and learners by creating MCQs.
"""

def load_model(checkpoint_path):
    """
    Loads the tokenizer and model from the specified checkpoint.
    """
    print(f"Loading model from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()
    model.to("cuda")

    print("Model loaded successfully!")
    return tokenizer, model

def generate_response(model, tokenizer, text, system_instruction=SYSTEM_INSTRUCTION):
    """
    Generates a response from the fine-tuned Phi-3.5 model and returns only the generated text.
    """
    try:
        full_prompt = f"<|system|>{system_instruction}<|end|>\n\n<|user|>{text}<|end|>\n\n<|assistant|>"

        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=4096, temperature=0.7, do_sample=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract text after "<|assistant|>"
        assistant_tag = "<|assistant|>"
        if assistant_tag in generated_text:
            generated_text = generated_text.split(assistant_tag)[-1].strip()

        return generated_text
    except Exception as error:
        print("An error occurred:", error)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned Phi-3.5 model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint directory.")
    parser.add_argument("text", type=str, help="Input text for the model.")
    
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)

    response = generate_response(model, tokenizer, args.text)
    
    if response:
        print("\nGenerated Response:\n")
        print(response)
    else:
        print("Failed to generate a response.")

if __name__ == "__main__":
    main()

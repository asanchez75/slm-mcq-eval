import ollama
from ollama import chat
import argparse


def run_inference(model_name: str, text: str) -> str:

    response = chat(model=model_name, messages=[{'role': 'user', 'content': text}])
    
    return response['message']['content']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with an Ollama model.")
    parser.add_argument("model_name", type=str, help="The name of the model to use for inference.")
    parser.add_argument("text", type=str, help="The input text prompt for the model.")
    
    args = parser.parse_args()

    output = run_inference(args.model_name, args.text)
    print(output)
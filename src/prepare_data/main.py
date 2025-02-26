import json
from mcq_model import create_prompt_chain
from parallel_generation import run_in_parallel
from utils import load_test_set

def main():
    with open("all_txt_contents.json", "r") as f:
        all_txt_contents = json.load(f)
    
    test_set = load_test_set(all_txt_contents, test_folders_file="test_folders.json")
    
    chain = create_prompt_chain()
    
    generated_questions = run_in_parallel(test_set, chain, max_workers=10)
    
    with open("generated_questions.json", "w") as f:
        json.dump(generated_questions, f, indent=2)
    
    print(f"Generated {len(generated_questions)} questions. Results saved to generated_questions.json.")

if __name__ == "__main__":
    main()
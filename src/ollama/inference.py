import ollama
from ollama import chat
from ollama import ChatResponse

import pandas as pd
from pydantic import BaseModel
from ollama import generate

from pydantic import ValidationError

df = pd.read_csv("lisa_sheets.csv")

class MCQQuestion(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str


def generate_mcq(content, model_name= "phi3.5:3.8b-mini-instruct-q8_0"):
    prompt = f"""
    Based on the following educational content, generate a multiple-choice question with four answer 
    options where only one is correct. The question should assess understanding of the main ideas, 
    and the options should be clear, informative, and relevant. Ensure that the distractors (incorrect options) 
    follow a logical but incorrect interpretation, based on common misconceptions or misunderstandings of the topic.
    Answer options must be as short as possible.

    **Educational Content**
    {content}
    """
    
    generate_params = {
        'model': MODEL_NAME,
        'options': {'temperature': 0.5, 'num_ctx': 8192, 'top_p': 1}, 
        'prompt': prompt,
        'format': MCQQuestion.model_json_schema()
    }
    
    response = generate(**generate_params)
    
    return response['response']


phi_generated = df_small["content_gpt"].apply(generate_mcq)


def validate_mcq(mcq_json):
    try:
        return MCQQuestion.model_validate_json(mcq_json)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None
        
phi16_test = phi_generated.apply(validate_mcq)

phi16_test.isna().sum()
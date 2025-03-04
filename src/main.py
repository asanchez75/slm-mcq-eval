import os
import json
import pandas as pd
from src.eval.eval_dataframe import eval_dataframe
from dotenv import load_dotenv

load_dotenv('../.env')
OPENAI_KEY = os.environ.get("OPENAI_KEY")

with open('./eval/prompts.json', 'r') as file:
    # Load the JSON data from the file
    system_prompts = json.load(file)

df_mcq = pd.read_csv('../data/gpt4_mcq.csv')[:10]
df_lisa_sheets = pd.read_csv('../data/lisa_sheets.csv')

df_eval = eval_dataframe(df_mcqs=df_mcq,
                         df_lisa_sheets=df_lisa_sheets,
                         openai_key=OPENAI_KEY,
                         answerability_system_prompt=system_prompts['answerability_prompt'],
                         compute_answerability=False,
                         compute_originality=False,
                         compute_ambiguity=False)
df_eval.to_csv('test.csv', index=False)
import os
import json
import pandas as pd
from src.eval.eval_dataframe import eval_dataframe, eval_dataframe_parallel
from dotenv import load_dotenv

def main():
    load_dotenv('../.env')
    OPENAI_KEY = os.environ.get("OPENAI_KEY")

    with open('./eval/prompts.json', 'r') as file:
        # Load the JSON data from the file
        system_prompts = json.load(file)

    df_mcq = pd.read_csv('../data/gpt4_mcq.csv')[:12]
    df_lisa_sheets = pd.read_csv('../data/lisa_sheets.csv')[:12]

    df_eval = eval_dataframe_parallel(df_mcqs=df_mcq,
                                      df_lisa_sheets=df_lisa_sheets,
                                      openai_key=OPENAI_KEY,
                                      num_workers=12,
                                      answerability_system_prompt=system_prompts['answerability_prompt'])
    df_eval.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()

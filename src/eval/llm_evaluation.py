
from tqdm import tqdm
import numpy as np

tqdm.pandas()

def call_openai_api(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def generate_prompt_for_question(row):
    question_text = row['question']
    options = f"a) {row['option_a']}\nb) {row['option_b']}\nc) {row['option_c']}\nd) {row['option_d']}"
    user_prompt = f"""Question:\n{question_text}\nOptions:\n{options}"""
    try:
        return call_openai_api(system_prompt, user_prompt)
    except Exception as e:
        print(f"Error processing question at index {row.id}: {e}")
        return None


def process_dataframe(model_name, df):
    try:
        df['rank'] = df.progress_apply(generate_prompt_for_question, axis=1)
        df.to_csv(f'/kaggle/working/results_of_{model_name}.csv', index=False)
    except Exception as e:
        print(f"Error occurred for model {model_name}: {e}")
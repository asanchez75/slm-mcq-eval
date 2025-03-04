import json

from openai import OpenAI

from src.eval.llm_evaluation import generate_prompt_for_question, call_openai_api


def compute_answerability_for_df(df,
                                 api_key,
                                 question_col,
                                 option_a_col,
                                 option_b_col,
                                 option_c_col,
                                 option_d_col,
                                 context_col,
                                 model_answer_col,
                                 system_prompt,
                                 temp,
                                 max_completion_tokens):

    client = OpenAI(api_key = api_key)
            
    def answerability_applicable(row):
        user_prompt = generate_prompt_for_question(row,
                                                   question_col=question_col,
                                                   option_a_col = option_a_col,
                                                   option_b_col = option_b_col,
                                                   option_c_col = option_c_col,
                                                   option_d_col = option_d_col,
                                                   context_col=context_col)
        
        return call_openai_api(client, user_prompt, system_prompt, temp=temp, max_completion_tokens=max_completion_tokens)

    df[model_answer_col] = df.apply(answerability_applicable, axis=1)
    return df
    
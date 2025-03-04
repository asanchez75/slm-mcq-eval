import pandas as pd

from src.eval.ambiguity import calculate_ambiguity_for_df
from src.eval.answerability import compute_answerability
from src.eval.negation import starts_with_negation
from src.eval.originality import calculate_originality_for_df
from src.eval.question_check import is_question
from src.eval.readability import calculate_readability_for_df
from src.eval.relevance import calculate_relevance_for_df

from dotenv import load_dotenv
import os

load_dotenv('../../.env')
OPENAI_KEY = os.environ.get("OPENAI_KEY")


def eval_dataframe(df_mcqs: pd.DataFrame,
                   df_lisa_sheets: pd.DataFrame,
                   # openai params
                   openai_key: str,
                   temp=0.5,
                   max_completion_tokens=4096,
                   output_file_path='./mcqs_eval.csv',
                   # answerability params
                   compute_answerability=True,
                   answerability_col='gpt_answer',
                   answerability_system_prompt=None,
                   # originality params
                   compute_originality=True,
                   originality_col='originality',
                   # readability params
                   compute_readability=True,
                   readability_col='readability',
                   # negation params
                   compute_negation=True,
                   negation_col='starts_with_negation',
                   # is_question params
                   compute_is_question=True,
                   is_question_col='is_question',
                   # relevance_params
                   compute_relevance=True,
                   relevance_col='relevance',
                   # ambiguity_params
                   compute_ambiguity=True,
                   ambiguity_col='ambiguity',
                   # general df params
                   question_col='question',
                   option_a_col='option_a',
                   option_b_col='option_b',
                   option_c_col='option_c',
                   option_d_col='option_d',
                   correct_option_col='correct_option',
                   lisa_sheet_id_col='id',
                   lisa_sheet_col='content_gpt',):
    # Merge the MCQs and LISA sheets on the specified ID column
    df_merged = pd.merge(df_mcqs,
                         df_lisa_sheets[[lisa_sheet_id_col, lisa_sheet_col]],
                         on=lisa_sheet_id_col, how='left')

    if compute_originality:
        df_merged = calculate_originality_for_df(df_merged,
                                                 originality_col=originality_col,
                                                 question_col=question_col,
                                                 lisa_sheet_col=lisa_sheet_col)

    if compute_readability:
        df_merged = calculate_readability_for_df(df_merged,
                                                 readability_col=readability_col,
                                                 question_col=question_col)

    if compute_negation:
        df_merged[negation_col] = df_merged[question_col].apply(starts_with_negation)

    if compute_is_question:
        df_merged[is_question_col] = df_merged[question_col].apply(is_question)

    if compute_relevance:
        df_merged = calculate_relevance_for_df(df_merged,
                                               relevance_col=relevance_col,
                                               question_col=question_col,
                                               lisa_sheet_col=lisa_sheet_col)

    if compute_ambiguity:
        df_merged = calculate_ambiguity_for_df(df_merged,
                                               correct_option_col=correct_option_col,
                                               option_a_col=option_a_col,
                                               option_b_col=option_b_col,
                                               option_c_col=option_c_col,
                                               option_d_col=option_d_col,
                                               ambiguity_col=ambiguity_col)

    if compute_answerability and answerability_system_prompt is not None:
        df_merged = compute_answerability(df_merged,
                                          api_key=openai_key,
                                          question_col=question_col,
                                          option_a_col=option_a_col,
                                          option_b_col=option_b_col,
                                          option_c_col=option_c_col,
                                          option_d_col=option_d_col,
                                          model_answer_col=answerability_col,
                                          context_col=lisa_sheet_col,
                                          system_prompt=answerability_system_prompt,
                                          temp=temp,
                                          max_completion_tokens=max_completion_tokens)

    return df_merged

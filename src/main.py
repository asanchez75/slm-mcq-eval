import pandas as pd

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')


from eval.answer_length import calculate_length_score_for_df
from eval.negation import starts_with_negation
from eval.originality import calculate_originality_for_df
from eval.question_check import is_question
from eval.readability import calculate_readability_for_df
from eval.relevance import calculate_relevance_for_df
from eval.ambiguity import calculate_ambiguity_for_df

data = {
    'question': [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Should we invest in Bitcoin?",
        "Is this a good approach?",
        "The sun rises in the east.",
        "Why is the sky blue?",
        "This is a statement, not a question."
    ],
    'lisa_sheet': [
        "The capital of France is Paris.",
        "Relativity is a theory developed by Einstein.",
        "Investing in Bitcoin depends on market conditions.",
        "Approach validation is subjective.",
        "Earth rotates causing sunrise in the east.",
        "The sky is blue due to Rayleigh scattering.",
        "This sentence has no question."
    ],
    'correct_option': ['a', 'b', 'c', 'd', 'a', 'b', 'c'],
    'option_a': ["Paris", "Newtonian mechanics", "Yes", "Maybe", "Morning", "Atmosphere", "Nothing"],
    'option_b': ["London", "Relativity", "No", "Not sure", "Evening", "Scattering", "Something"],
    'option_c': ["Berlin", "Thermodynamics", "Possibly", "Perhaps", "Sunrise", "Light", "Everything"],
    'option_d': ["Madrid", "Quantum physics", "Definitely", "Doubtful", "Dawn", "Blue sky", "Anything"]
}

df = pd.DataFrame(data)

df = calculate_length_score_for_df(df, question_col='question')

df = calculate_originality_for_df(df, question_col='question', lisa_sheet_col='lisa_sheet')

df = calculate_readability_for_df(df, text_col='question')

df['starts_with_negation'] = df['question'].apply(starts_with_negation)

df['is_question'] = df['question'].apply(is_question)

df = calculate_relevance_for_df(df)
df = calculate_ambiguity_for_df(df)

print(df.iloc[0])
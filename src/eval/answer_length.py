
def calculate_length_score_for_df(df, question_col='question', lower_bound=10, upper_bound=25):

    def get_length_score(length):
        if lower_bound <= length <= upper_bound:
            return 1
        elif length > upper_bound:
            return (upper_bound / length) ** 2
        else:
            return (length / lower_bound) ** 2

    df['question_length'] = df[question_col].apply(lambda x: len(x.split()))
    
    df['length_score'] = df['question_length'].apply(get_length_score)

    return df
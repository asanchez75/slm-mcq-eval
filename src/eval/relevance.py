import torch.nn.functional as F

from src.eval.utils import load_model, generate_embedding


def calculate_relevance_for_df(df,
                               relevance_col,
                               question_col,
                               lisa_sheet_col,
                               model_name="BAAI/bge-base-en-v1.5"):
    model, tokenizer, device = load_model(model_name)

    def relevance_score(row):
        q_emb = generate_embedding(row[question_col], model, tokenizer, device)
        l_emb = generate_embedding(row[lisa_sheet_col], model, tokenizer, device)
        similarity = F.cosine_similarity(q_emb, l_emb, dim=0).item()
        return similarity

    df[relevance_col] = df.apply(relevance_score, axis=1)
    return df


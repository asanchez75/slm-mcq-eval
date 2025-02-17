import re

def is_question(sentence):
    if sentence.strip().endswith("?"):
        return True
    
    question_words = r"^(who|what|where|when|why|how|does|should|do|did|could|will|would)\b"
    if re.match(question_words, sentence.strip(), re.IGNORECASE):
        return True
    
    return False



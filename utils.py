import spacy
from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_lg")
stop_words = set(stopwords.words("english"))

# For training essays
def preprocess_essays(essays):
    processed_essays = []
    for doc in nlp.pipe(essays, disable=["ner", "parser"]):
        essay_tokens = []
        for token in doc:
            if token.is_alpha and token.text.lower() not in stop_words:
                essay_tokens.append(token.text.lower())
        processed_essays.append(essay_tokens)
    return processed_essays

# For a single essay (user input via Flask)
def preprocess_single_essay(essay):
    doc = nlp(essay)
    return ' '.join([token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words])

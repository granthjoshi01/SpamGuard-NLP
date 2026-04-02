import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def _load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set()


STOP_WORDS = _load_stop_words()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    words = [
        stemmer.stem(word)
        for word in normalized.split()
        if word and word not in STOP_WORDS
    ]
    return " ".join(words)

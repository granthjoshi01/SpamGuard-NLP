from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_VECTORIZER_PATH = Path("artifacts/vectorizer.pkl")


def train_vectorizer(texts, save_path=DEFAULT_VECTORIZER_PATH):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(texts)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as file_obj:
            pickle.dump(vectorizer, file_obj)

    return vectorizer


def load_vectorizer(path=DEFAULT_VECTORIZER_PATH):
    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)

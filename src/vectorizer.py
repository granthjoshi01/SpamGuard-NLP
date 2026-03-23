from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def train_vectorizer(texts, save_path="vectorizer.pkl"):
    """
    Trains a TF-IDF vectorizer and saves it to a file.

    Parameters:
        texts (list): List of preprocessed text messages.
        save_path (str): Path to save the vectorizer.

    Returns:
        TfidfVectorizer: Trained vectorizer.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(texts)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer

def load_vectorizer(path="vectorizer.pkl"):
    """
    Loads a saved TF-IDF vectorizer.

    Parameters:
        path (str): Path to the saved vectorizer.

    Returns:
        TfidfVectorizer: Loaded vectorizer.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
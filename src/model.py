from pathlib import Path
import pickle

from sklearn.naive_bayes import MultinomialNB


DEFAULT_MODEL_PATH = Path("artifacts/model.pkl")


def train_model(x_train, y_train, save_path=DEFAULT_MODEL_PATH):
    model = MultinomialNB()
    model.fit(x_train, y_train)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as file_obj:
            pickle.dump(model, file_obj)

    return model


def load_model(path=DEFAULT_MODEL_PATH):
    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)

from sklearn.model_selection import train_test_split

from src.data_loader import clean_labels, load_data
from src.model import train_model
from src.preprocessing import preprocess_text
from src.vectorizer import train_vectorizer


def train_pipeline(data_path="data/spam.csv", save_artifacts=True):
    df = load_data(data_path)
    df = clean_labels(df)
    df["message"] = df["message"].apply(preprocess_text)

    x_train, _, y_train, _ = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = train_vectorizer(
        x_train, save_path=None if not save_artifacts else "artifacts/vectorizer.pkl"
    )
    x_train_vec = vectorizer.transform(x_train)
    model = train_model(
        x_train_vec, y_train, save_path=None if not save_artifacts else "artifacts/model.pkl"
    )
    return vectorizer, model

from sklearn.model_selection import train_test_split

from src.data_loader import clean_labels, load_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.preprocessing import preprocess_text
from src.vectorizer import train_vectorizer
from src.logging import setup_logger


def main():
    logger = setup_logger()
    logger.info("Starting training pipeline")

    df = load_data("data/spam.csv")
    logger.info(f"Loaded dataset with {len(df)} rows")
    df = clean_labels(df)
    df["message"] = df["message"].apply(preprocess_text)

    x_train, x_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    vectorizer = train_vectorizer(x_train)
    x_train_vec = vectorizer.transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = train_model(x_train_vec, y_train)
    logger.info("Model training completed")
    y_pred = model.predict(x_test_vec)

    metrics = evaluate_model(y_test, y_pred)
    logger.info(f"Evaluation metrics: {metrics}")
    print(metrics)


if __name__ == "__main__":
    main()

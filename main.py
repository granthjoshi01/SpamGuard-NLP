from src.data_loader import load_data, clean_labels
from src.preprocessing import preprocess_text
from src.vectorizer import train_vectorizer, load_vectorizer
from src.model import train_model, load_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Load data
    df = load_data("data/spam.csv")
    # Convert labels to numeric
    df = clean_labels(df)

    # Preprocess data
    df['message'] = df['message'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Vectorize data
    vectorizer = train_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = train_model(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    print(metrics)

if __name__ == "__main__":
    main()
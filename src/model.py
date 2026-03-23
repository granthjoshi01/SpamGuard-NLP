from sklearn.naive_bayes import MultinomialNB
import pickle

def train_model(X_train, y_train, save_path="model.pkl"):
    """
    Trains a Multinomial Naive Bayes model and saves it to a file.

    Parameters:
        X_train (array): Training feature matrix.
        y_train (array): Training labels.
        save_path (str): Path to save the model.

    Returns:
        MultinomialNB: Trained model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model

def load_model(path="model.pkl"):
    """
    Loads a saved Multinomial Naive Bayes model.

    Parameters:
        path (str): Path to the saved model.

    Returns:
        MultinomialNB: Loaded model.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
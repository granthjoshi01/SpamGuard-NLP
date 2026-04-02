import pandas as pd


def load_data(path="data/spam.csv"):
    df = pd.read_csv(path)
    expected_columns = {"label", "message"}
    if expected_columns.issubset(df.columns):
        return df[["label", "message"]].copy()

    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
        return df

    raise ValueError("Dataset must contain either label/message or v1/v2 columns.")


def clean_labels(df):
    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1}
    cleaned = df.copy()
    cleaned["label"] = cleaned["label"].map(label_map)

    if cleaned["label"].isnull().any():
        raise ValueError("Dataset contains unsupported label values.")

    return cleaned

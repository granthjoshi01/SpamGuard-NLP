import pandas as pd
#docstrings + structure explanation inside file
"""
data_loader.py

This module handles:
- Loading the dataset
- Cleaning column names
- Converting labels
- Performing basic data checks
"""

def load_data(path="data/spam.csv"):
    """
    Loads the spam dataset, selects relevant columns, and renames them.

    Parameters:
        path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Cleaned dataframe with columns ['label', 'message']
    """
    try:
        # Load dataset
        df = pd.read_csv(path, encoding="latin-1")
        
        # Check if required columns exist
        if 'v1' not in df.columns or 'v2' not in df.columns:
            raise ValueError("Required columns 'v1' and 'v2' are missing in the dataset.")
        
        # Select and rename columns
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path '{path}'.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def clean_labels(df):
    """
    Converts 'ham'/'spam' labels into numeric values (0/1).

    Parameters:
        df (pandas.DataFrame): Dataframe with a 'label' column.

    Returns:
        pandas.DataFrame: Dataframe with numeric labels.
    """
    if 'label' not in df.columns:
        raise ValueError("The dataframe does not contain a 'label' column.")
    
    # Convert labels to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


def basic_checks(df):
    """
    Prints basic information about the dataset for validation and understanding.
    """
    print("\n--- Basic Info ---")
    print(df.info())

    print("\n--- Null Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Label Distribution ---")
    print(df['label'].value_counts())

    print("\n--- Message Length Stats ---")
    print(df['message'].apply(len).describe())


def remove_duplicates(df):
    """
    Removes duplicate rows from the dataset.
    """
    return df.drop_duplicates()
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    """
    Preprocesses a given text by:
    - Lowercasing
    - Removing special characters (keeping numbers)
    - Removing stopwords
    - Applying stemming

    Parameters:
        text (str): Raw text input.

    Returns:
        str: Preprocessed text.
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)
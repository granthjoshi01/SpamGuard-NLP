import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize once
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses a given text by:
    - Lowercasing
    - Removing special characters (keeping numbers)
    - Removing stopwords
    - Applying stemming
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

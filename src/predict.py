import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_message(text):
    model = joblib.load('models/spam_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    return model.predict(vector)[0]

if __name__ == "__main__":
    text = input("Enter message: ")
    print("Prediction:", predict_message(text))
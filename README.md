# SpamGuard

SpamGuard is a Flask-based multilingual SMS spam detection demo built on top of a classic NLP pipeline.

## Features

- Manual SMS entry
- Language selection for multilingual support
- Spam/Ham prediction popup with confidence and safety tips
- English NLP pipeline using TF-IDF + Multinomial Naive Bayes
- Translation abstraction for non-English SMS input

## Run locally

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python app.py
```

Then open `http://127.0.0.1:5000`.

## Train the model

```bash
python main.py
```

Training creates `artifacts/model.pkl` and `artifacts/vectorizer.pkl`.

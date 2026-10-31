"""Fine‑tune DistilBERT for binary SMS spam detection.

Usage:
    python -m src.deep_learning.train_transformer \
        --data_path data/spam.csv \
        --output_dir artifacts/transformer \
        --epochs 3 --batch_size 16 --lr 2e-5 --max_len 128
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.data_loader import load_data, clean_labels
from src.preprocessing import preprocess_text
from src.logging import setup_logger

LOGGER = setup_logger()


class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main():
    parser = argparse.ArgumentParser(description="DistilBERT fine‑tuning for SpamGuard")
    parser.add_argument("--data_path", default="data/spam.csv", help="Path to CSV dataset")
    parser.add_argument("--output_dir", default="artifacts/transformer", help="Where to save model & tokenizer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    # Reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load & preprocess data
    df = load_data(args.data_path)
    df = clean_labels(df)
    df["message"] = df["message"].apply(preprocess_text)

    # Stratified split
    from sklearn.model_selection import train_test_split

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["message"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = SMSDataset(train_texts, train_labels, tokenizer, args.max_len)
    eval_dataset = SMSDataset(test_texts, test_labels, tokenizer, args.max_len)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "save_total_limit": 2,
    }
    try:
        training_args = TrainingArguments(eval_strategy="epoch", save_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Starting DistilBERT fine‑tuning")
    trainer.train()
    LOGGER.info("Training complete – saving artifacts")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Save label map for inference
    (Path(args.output_dir) / "label_map.json").write_text('{"0": "Ham", "1": "Spam"}')
    LOGGER.info(f"Artifacts stored in {args.output_dir}")


if __name__ == "__main__":
    main()


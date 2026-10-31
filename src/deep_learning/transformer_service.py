"""Service to load a fine‑tuned DistilBERT model and run inference.

The model, tokenizer and a simple label‑map are stored under
`artifacts/transformer/` by the training script.
"""

import json
from pathlib import Path
from typing import Tuple

from src.logging import setup_logger

LOGGER = setup_logger()
DEFAULT_ARTIFACT_DIR = Path("artifacts/transformer")


class TransformerService:
    """Lazy‑loading wrapper around the fine‑tuned DistilBERT model.

    The service is instantiated once per process (i.e. once per Flask worker).
    It loads the tokenizer, the model, and the label map the first time a
    prediction is requested. Subsequent calls reuse the loaded objects.
    """

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self._model = None
        self._tokenizer = None
        self._label_map = None
        self._device = None

    # ---------------------------------------------------------------------
    def _load_artifacts(self):
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and HuggingFace Transformers must be installed to use DistilBERT. "
                "Install them via: pip install torch transformers"
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_exist = (
            (self.artifact_dir / "pytorch_model.bin").exists()
            or (self.artifact_dir / "model.safetensors").exists()
        )
        if not weights_exist:
            raise FileNotFoundError(
                f"DistilBERT artifacts not found in '{self.artifact_dir}'. "
                "Please train the model first by running: python -m src.deep_learning.train_transformer"
            )

        LOGGER.info(f"Loading DistilBERT artifacts from {self.artifact_dir} (device: {self._device})")
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.artifact_dir))
        self._model = DistilBertForSequenceClassification.from_pretrained(str(self.artifact_dir))
        self._model.to(self._device)
        self._model.eval()

        label_map_path = self.artifact_dir / "label_map.json"
        if label_map_path.exists():
            self._label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        else:
            self._label_map = {"0": "Ham", "1": "Spam"}

    # ---------------------------------------------------------------------
    def predict(self, text: str) -> Tuple[str, float]:
        """Return a `(label, confidence)` tuple for a single `text`.

        * `label` is the human‑readable class name ("Spam" or "Ham").
        * `confidence` is the probability assigned to that class (0‑1).
        """
        self._load_artifacts()
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        label = self._label_map.get(str(pred_idx), "Spam" if pred_idx == 1 else "Ham")
        confidence = float(probs[pred_idx])
        return label, confidence

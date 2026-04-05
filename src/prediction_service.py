from dataclasses import dataclass

from src.config import SUPPORTED_LANGUAGE_CODES
from src.model import load_model
from src.preprocessing import preprocess_text
from src.training import train_pipeline
from src.translation import GoogleTranslationProvider, TranslationError
from src.vectorizer import load_vectorizer
from src.logging import setup_logger


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PredictionResult:
    label: str
    confidence: float
    tips: str
    translated_text: str
    normalized_text: str
    language: str


class PredictionService:
    def __init__(self, translator=None):
        self.translator = translator or GoogleTranslationProvider()
        self._vectorizer = None
        self._model = None
        self._logger = setup_logger()

    def _ensure_artifacts(self):
        if self._vectorizer is not None and self._model is not None:
            return

        self._logger.info("Ensuring model and vectorizer artifacts are loaded")

        try:
            self._vectorizer = load_vectorizer()
            self._model = load_model()
        except FileNotFoundError:
            self._logger.warning("Artifacts not found. Training pipeline triggered.")
            self._vectorizer, self._model = train_pipeline(save_artifacts=False)

    def predict_message(self, message: str, language: str) -> PredictionResult:
        if not isinstance(message, str) or not message.strip():
            raise PredictionError("Please enter an SMS message before checking it.")

        if language not in SUPPORTED_LANGUAGE_CODES:
            raise PredictionError("Please choose a supported SMS language.")

        self._logger.info(f"Prediction request received | language={language}")

        try:
            translated_text = self.translator.translate(message.strip(), language)
        except TranslationError as exc:
            self._logger.error(f"Translation failed: {str(exc)}")
            raise PredictionError(str(exc), status_code=502) from exc

        normalized_text = preprocess_text(translated_text)
        self._logger.info(f"Text normalized | length={len(normalized_text.split())}")
        if not normalized_text:
            raise PredictionError(
                "That message could not be processed. Please try a longer SMS."
            )

        self._ensure_artifacts()

        features = self._vectorizer.transform([normalized_text])
        predicted_class = int(self._model.predict(features)[0])
        probabilities = self._model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        self._logger.info(f"Model prediction complete | class={predicted_class} | confidence={confidence:.4f}")

        label = "Spam" if predicted_class == 1 else "Ham"
        tips = self._build_tips(label)

        return PredictionResult(
            label=label,
            confidence=confidence,
            tips=tips,
            translated_text=translated_text,
            normalized_text=normalized_text,
            language=language,
        )

    @staticmethod
    def _build_tips(label: str) -> str:
        if label == "Spam":
            return "Avoid clicking links, sharing OTPs, or replying before you verify the sender."
        return "This message looks safer, but still confirm links, money requests, and unknown senders."

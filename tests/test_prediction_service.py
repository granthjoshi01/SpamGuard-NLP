import pytest

from src.prediction_service import PredictionError, PredictionResult, PredictionService
from src.translation import TranslationError


class StubTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, source_language):
        self.calls.append((text, source_language))
        return "free reward click now"


class FailingTranslator:
    def translate(self, text, source_language):
        raise TranslationError("translation unavailable")


def test_prediction_returns_required_fields():
    service = PredictionService(translator=StubTranslator())
    result = service.predict_message("Mensaje sospechoso", "es")

    assert isinstance(result, PredictionResult)
    assert result.label in {"Spam", "Ham"}
    assert 0 <= result.confidence <= 1
    assert result.tips
    assert result.translated_text


def test_translation_step_uses_selected_language():
    translator = StubTranslator()
    service = PredictionService(translator=translator)

    service.predict_message("Hola oferta gratis", "es")

    assert translator.calls == [("Hola oferta gratis", "es")]


def test_unsupported_language_returns_controlled_error():
    service = PredictionService(translator=StubTranslator())

    with pytest.raises(PredictionError) as exc_info:
        service.predict_message("test message", "de")

    assert "supported SMS language" in str(exc_info.value)


def test_translation_failure_is_graceful():
    service = PredictionService(translator=FailingTranslator())

    with pytest.raises(PredictionError) as exc_info:
        service.predict_message("bonjour", "fr")

    assert exc_info.value.status_code == 502
    assert "translation unavailable" in str(exc_info.value)

from dataclasses import dataclass

from deep_translator import GoogleTranslator


class TranslationError(Exception):
    """Raised when text translation cannot be completed."""


@dataclass
class TranslationProvider:
    def translate(self, text: str, source_language: str) -> str:
        raise NotImplementedError


class GoogleTranslationProvider(TranslationProvider):
    def translate(self, text: str, source_language: str) -> str:
        if source_language == "en":
            return text

        try:
            translated = GoogleTranslator(source=source_language, target="en").translate(
                text
            )
        except Exception as exc:
            raise TranslationError(
                "We could not translate that SMS right now. Please try again in a moment."
            ) from exc

        if not translated:
            raise TranslationError(
                "We could not translate that SMS right now. Please try again in a moment."
            )

        return translated

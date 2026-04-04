from flask import Flask, jsonify, render_template, request

from src.config import SUPPORTED_LANGUAGES
from src.prediction_service import PredictionError, PredictionService
from src.logging import setup_logger


def create_app() -> Flask:
    app = Flask(__name__)
    service = PredictionService()
    logger = setup_logger()

    @app.get("/")
    def index():
        return render_template("index.html", languages=SUPPORTED_LANGUAGES)

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")
        language = payload.get("language", "")
        logger.info(f"Incoming request | message='{message}' | language='{language}'")

        try:
            result = service.predict_message(message=message, language=language)
            logger.info(
                f"Prediction | label={result.label} | confidence={result.confidence:.4f}"
            )
        except PredictionError as exc:
            logger.error(f"PredictionError: {str(exc)}")
            return jsonify({"ok": False, "error": str(exc)}), exc.status_code
        except Exception as exc:
            logger.exception(f"Unhandled error: {str(exc)}")
            return jsonify({"ok": False, "error": "Internal server error"}), 500

        return jsonify(
            {
                "ok": True,
                "result": {
                    "label": result.label,
                    "confidence": round(result.confidence * 100, 2),
                    "tips": result.tips,
                    "translated_text": result.translated_text,
                    "normalized_text": result.normalized_text,
                    "language": result.language,
                },
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

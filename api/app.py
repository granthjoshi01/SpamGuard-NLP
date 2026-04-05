from flask import Flask, jsonify, render_template, request, send_from_directory

from src.config import SUPPORTED_LANGUAGES
from src.prediction_service import PredictionError, PredictionService


def create_app() -> Flask:
    app = Flask(__name__)
    service = PredictionService()

    @app.get("/")
    def index():
        return render_template("index.html", languages=SUPPORTED_LANGUAGES)

    @app.get("/style.css")
    def style():
        return send_from_directory("public", "style.css")

    @app.get("/app.js")
    def script():
        return send_from_directory("public", "app.js")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")
        language = payload.get("language", "")

        try:
            result = service.predict_message(message=message, language=language)
        except PredictionError as exc:
            return jsonify({"ok": False, "error": str(exc)}), exc.status_code

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

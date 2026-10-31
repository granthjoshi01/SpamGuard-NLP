import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory

from src.config import SUPPORTED_LANGUAGES
from src.prediction_service import PredictionError, PredictionService
from src.logging import setup_logger


def create_app() -> Flask:
    app = Flask(__name__)
    service = PredictionService()
    logger = setup_logger()

    @app.get('/')
    def index():
        return render_template('index.html', languages=SUPPORTED_LANGUAGES)

    @app.get('/style.css')
    def style():
        return send_from_directory('public', 'style.css')

    @app.get('/app.js')
    def script():
        return send_from_directory('public', 'app.js')

    @app.get('/dashboard')
    def dashboard_page():
        return render_template('dashboard.html')

    # Serve artifact files (confusion matrices, etc.)
    @app.get('/artifacts/<path:filename>')
    def serve_artifact(filename: str):
        return send_from_directory('artifacts', filename)

    @app.get('/report')
    def report():
        if not os.path.exists('artifacts/model_comparison.json'):
            return jsonify({"ok": False, "error": "Report not generated"}), 404
        try:
            with open('artifacts/model_comparison.json', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({"ok": True, "report": data})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Report not generated"}), 404

    @app.post('/predict')
    def predict():
        payload = request.get_json(silent=True) or {}
        message = payload.get('message', '')
        language = payload.get('language', '')
        model_name = payload.get('model', 'naive_bayes')  # default baseline
        logger.info(f"Incoming request | message='{message}' | language='{language}' | model='{model_name}'")
        try:
            result = service.predict_message(message=message, language=language, model_name=model_name)
            logger.info(
                f"Prediction | label={result.label} | confidence={result.confidence:.4f} | model={model_name}"
            )
        except PredictionError as exc:
            logger.error(f"PredictionError: {str(exc)}")
            return jsonify({"ok": False, "error": str(exc)}), exc.status_code
        except Exception as exc:
            logger.exception(f"Unhandled error: {str(exc)}")
            return jsonify({"ok": False, "error": "Internal server error"}), 500

        return jsonify({
            "ok": True,
            "result": {
                "label": result.label,
                "confidence": round(result.confidence * 100, 2),
                "tips": result.tips,
                "translated_text": result.translated_text,
                "normalized_text": result.normalized_text,
                "language": result.language,
                "model": model_name,
            },
        })

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

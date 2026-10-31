import os
import json
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    with app.test_client() as client:
        yield client

def test_dashboard_route_returns_200(client):
    resp = client.get('/dashboard')
    assert resp.status_code == 200
    assert b"Model Benchmark & Comparison" in resp.data

def test_report_route_missing_file_returns_404(client, monkeypatch):
    # Ensure artifacts folder does not contain the report file via monkeypatch
    def mock_exists(path):
        return False
    monkeypatch.setattr(os.path, 'exists', mock_exists)
    resp = client.get('/report')
    assert resp.status_code == 404
    json_data = resp.get_json()
    assert json_data["ok"] is False

def test_report_route_returns_report_when_file_exists(client):
    report_path = os.path.join("artifacts", "model_comparison.json")
    if not os.path.exists(report_path):
        os.makedirs("artifacts", exist_ok=True)
        report = {"naive_bayes": {"accuracy": 0.95}, "distilbert": {"accuracy": 0.985}}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

    resp = client.get('/report')
    assert resp.status_code == 200
    json_data = resp.get_json()
    assert json_data["ok"] is True
    assert "report" in json_data

def test_predict_route_with_naive_bayes(client):
    resp = client.post('/predict', json={
        "message": "Congratulations! You have won $1000 prize. Claim now.",
        "language": "en",
        "model": "naive_bayes"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["result"]["model"] == "naive_bayes"
    assert data["result"]["label"] in ["Spam", "Ham"]

def test_predict_route_invalid_model(client):
    resp = client.post('/predict', json={
        "message": "Hello friend",
        "language": "en",
        "model": "invalid_model_type"
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False

def test_artifacts_route_serves_confusion_matrix(client):
    if os.path.exists('artifacts/nb_confusion.png'):
        resp = client.get('/artifacts/nb_confusion.png')
        assert resp.status_code == 200

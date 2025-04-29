from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Turbofan Engine RUL Prediction API",
        "docs": "/docs"
    }

def test_predict_endpoint():
    test_data = {
        "engine_id": 1,
        "settings": [0.0, -0.0002, 100.0],
        "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 
                   554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66,
                   2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100, 38.5, 23.0]
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "rul" in response.json()
    assert "confidence" in response.json()

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class TestPredictionEndpoint(unittest.TestCase):
    
    def test_rejects_blank_text_before_prediction(self):
        with patch("app.main.load_sentiment_model"):
            with TestClient(app) as client:
                with patch("app.main.predict_sentiment") as mocked_predict:
                    response = client.post(
                        "/predict",
                        json={"text": "   "},
                    )

        self.assertEqual(response.status_code, 422)
        mocked_predict.assert_not_called()

    def test_returns_prediction_for_valid_text(self):
        expected = {"prediction": "positive", "score": 0.9}

        with patch("app.main.load_sentiment_model"):
            with TestClient(app) as client:
                with patch(
                    "app.main.predict_sentiment",
                    return_value=expected,
                ) as mocked_predict:
                    response = client.post(
                        "/predict",
                        json={"text": "Great movie!"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mocked_predict.assert_called_once_with("Great movie!")
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import predict
from app.schemas import TextInput


class TestPredictionErrors(unittest.TestCase):
    def test_logs_internal_error_and_returns_generic_detail(self):
        payload = TextInput(text="Great movie!")

        with patch(
            "app.main.predict_sentiment",
            side_effect=ValueError("Simulated prediction failure"),
        ):
            with self.assertLogs("app.main", level="ERROR") as logs:
                with self.assertRaises(HTTPException) as caught:
                    predict(payload)

        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(
            caught.exception.detail,
            "Prediction failed. Please try again later.",
        )

        log_output = "\n".join(logs.output)
        self.assertIn("Sentiment prediction failed.", log_output)
        self.assertIn("Traceback", log_output)
        self.assertIn(
            "ValueError: Simulated prediction failure",
            log_output,
        )
from pathlib import Path

import tensorflow as tf

from src.sentiment_model.preprocessing import custom_standardization

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_model.keras"
model = None


def load_sentiment_model() -> None:
    """Load the trained sentiment analysis model into memory."""
    global model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"custom_standardization": custom_standardization},
    )


def predict_sentiment(text: str) -> dict:
    """Run inference on a single text input and return prediction + score."""
    global model

    if model is None:
        raise RuntimeError("Model is not loaded.")

    input_text = tf.constant([[text]], dtype=tf.string)
    raw_prediction = model.predict(input_text, verbose=0)
    prediction_value = float(raw_prediction[0][0])

    label = "positive" if prediction_value >= 0.5 else "negative"
    confidence = (
        prediction_value if prediction_value >= 0.5 else 1 - prediction_value
    )

    return {
        "prediction": label,
        "score": round(float(confidence), 4),
    }

import os
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, HTTPException

from app.schemas import TextInput

MODEL_PATH = "model/sentiment_model.keras"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple deep learning API for sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextInput):
    try:
        input_text = tf.constant([payload.text], dtype=tf.string)
        raw_prediction = model.predict(input_text, verbose=0)
        prediction_value = float(raw_prediction[0][0])

        label = "positive" if prediction_value >= 0.5 else "negative"
        confidence = prediction_value if prediction_value >= 0.5 else 1 - prediction_value

        return {
            "prediction": label,
            "score": round(float(confidence), 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
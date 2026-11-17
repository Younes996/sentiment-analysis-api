import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.predict import load_sentiment_model, predict_sentiment
from app.schemas import HealthResponse, PredictionResponse, TextInput

logger = logging.getLogger(__name__)
MODEL_PATH = "model/sentiment_model.keras"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_sentiment_model()
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


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TextInput):
    try:
        return predict_sentiment(payload.text)
    except Exception:
        logger.exception("Sentiment prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        )

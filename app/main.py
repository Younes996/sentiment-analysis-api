from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.predict import load_sentiment_model, predict_sentiment
from app.schemas import HealthResponse, PredictionResponse, TextInput

MODEL_PATH = "model/sentiment_model.keras"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Sentiment Analysis API

This project is an end-to-end NLP application that performs sentiment analysis using a deep learning model (BiLSTM) and serves predictions through a FastAPI REST API.

---

## Features

- Deep learning model built with TensorFlow (Embedding + BiLSTM)
- Text preprocessing integrated into the model
- REST API using FastAPI
- Real-time sentiment prediction
- Ready for containerization with Docker

---

## Project Structure

```text
sentiment-analysis-api/
├── app/
├── model/
├── data/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

---

## Train the Model

```bash
python model/train.py
```

---

## Run the API

```bash
uvicorn app.main:app --reload
```

---

## API Documentation

Once the API is running, open:

`http://127.0.0.1:8000/docs`

---

## Example Request

```json
{
  "text": "I really loved this product"
}
```

---

## Example Response

```json
{
  "prediction": "positive",
  "score": 0.82
}
```

---

## Tech Stack

- Python
- TensorFlow
- FastAPI
- Uvicorn
- Scikit-learn

---

## Future Improvements

- Deploy the API on Google Cloud Run
- Replace the toy dataset with a larger real-world dataset
- Add a frontend with Streamlit or React
- Improve model performance with Transformers from Hugging Face
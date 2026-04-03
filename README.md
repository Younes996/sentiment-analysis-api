# Sentiment Analysis API

An end-to-end NLP project for binary sentiment classification on IMDb movie reviews, built with **TensorFlow** and exposed through a **FastAPI** REST API.

The project includes:
- a custom text preprocessing pipeline,
- a deep learning model based on **TextVectorization + Embedding + BiLSTM**,
- model serving with FastAPI,
- a clean modular structure ready for future improvements such as **Streamlit UI** and **Docker**.


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Installation](#installation)
- [Train the Model](#train-the-model)
- [Run the API](#run-the-api)
- [API Documentation](#api-documentation)
- [Example Request](#example-request)
- [Example Response](#example-response)
- [Notes on Model Loading](#notes-on-model-loading)
- [Future Improvements](#future-improvements)


---

## Overview

This project performs **binary sentiment analysis** on movie reviews.  
Given a raw text review, the model predicts whether the sentiment is **positive** or **negative**.

It was designed as a complete mini production-style NLP project with:
- model training in TensorFlow,
- reusable preprocessing logic,
- model serialization,
- and serving through a FastAPI API.

---

## Features

- Binary sentiment classification on **IMDb** reviews
- Deep learning pipeline built with **TensorFlow / Keras**
- Custom text cleaning with a reusable `custom_standardization` function
- `TextVectorization` integrated into the model pipeline
- **Bidirectional LSTM** for contextual sequence learning
- Validation split + **EarlyStopping**
- FastAPI inference service
- Architecture image included for documentation
- Project structure ready for **Streamlit** and **Docker**

---

## Tech Stack

- Python
- TensorFlow
- FastAPI
- Uvicorn
- Scikit-learn
- pandas
- NumPy

---

## Project Structure

```text
sentiment-analysis-api/
├── app/
│   └── main.py
├── model/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── model_architecture.png
├── data/
│   ├── imdb_train.csv
│   ├── imdb_test.csv
│   └── aclImdb/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The model is trained on the **IMDb Large Movie Review Dataset** for binary sentiment classification.

### Official Dataset Source

The official dataset source used in this project is the **Large Movie Review Dataset** published by Stanford:

- [Official dataset page](https://ai.stanford.edu/~amaas/data/sentiment/)

In this project, the dataset is handled through:
- raw IMDb data (`aclImdb/`)
- CSV files prepared for training and evaluation:
  - `data/imdb_train.csv`
  - `data/imdb_test.csv`

Expected CSV format:

```text
text,label
"This movie was amazing",1
"I hated the ending",0
```

Where:
- `text` = review text
- `label` = sentiment label (`1` for positive, `0` for negative)

---

## Model Architecture

The model is built as an end-to-end TensorFlow pipeline that takes **raw text** as input and outputs a **binary sentiment probability**.

### Architecture Overview

- **InputLayer**: receives raw text as strings
- **TextVectorization**: converts text into integer token sequences of fixed length
- **Embedding**: maps tokens to dense vector representations
- **Bidirectional LSTM**: captures context from both directions
- **Dense + ReLU**: learns higher-level features
- **Dropout**: reduces overfitting
- **Dense + Sigmoid**: outputs a probability between `0` and `1`

### Model Diagram

![Model Architecture](model/model_architecture.png)

### Padding and Masking

The embedding layer uses:

```python
mask_zero=True
```

This allows the model to ignore padding tokens added after vectorization.  
As a result, padded values do not interfere with sequence learning inside the LSTM.

### Input / Output

- **Input**: raw text review
- **Output**: probability between `0` and `1`
  - close to `1` → positive sentiment
  - close to `0` → negative sentiment

### Architecture Plot Generation

The PNG architecture file is generated with:

```python
tf.keras.utils.plot_model(...)
```

On Windows, **Graphviz** must be installed separately and added to the system `PATH`.

Check installation with:

```bash
dot -V
```

If Graphviz is not installed, training still works normally, but the architecture image will not be generated.

---

## Training Configuration

The training pipeline includes:
- TensorFlow `tf.data.Dataset`
- train / validation split
- custom text preprocessing
- model checkpointing through `.keras` serialization
- `EarlyStopping` to reduce overfitting

Main ideas used in training:
- raw text is cleaned through `custom_standardization`
- text is vectorized inside the model pipeline
- the final classifier predicts a sentiment score with sigmoid activation

---

## Results

Current training results obtained on IMDb:

| Metric | Score |
|---|---:|
| Train Accuracy | 90.65% |
| Validation Accuracy | 84.26% |
| Test Accuracy | 83.40% |

Best validation accuracy observed during training:

| Metric | Score |
|---|---:|
| Best Validation Accuracy | 85.08% |

These results show that the model learns meaningful sentiment patterns while maintaining decent generalization on unseen reviews.

---

## Installation

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

On macOS / Linux:

```bash
source venv/bin/activate
```
---

## Train the Model

```bash
python model/train.py
```

This script:
- loads the training data,
- prepares the TensorFlow pipeline,
- trains the BiLSTM model,
- evaluates performance,
- and saves the trained model.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Once the server is running, the API can be used for real-time sentiment prediction.

---

## API Documentation

FastAPI automatically generates interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
  "text": "I really loved this movie. The acting was great and the story was moving."
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

## Notes on Model Loading

The project uses a custom preprocessing function:

```python
custom_standardization
```

Because this function is part of the serialized TensorFlow pipeline, the model must be loaded with `custom_objects` in FastAPI:

```python
custom_objects={"custom_standardization": custom_standardization}
```

This ensures that inference uses the exact same preprocessing logic as training.

---

## Future Improvements

- Improve performance through hyperparameter tuning
- Compare BiLSTM with GRU, CNN, and Transformer-based models
- Add a Streamlit frontend
- Containerize the application with Docker
- Deploy the API to Google Cloud Run
- Add automated tests for inference and API endpoints
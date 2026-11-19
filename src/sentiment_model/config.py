from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "imdb_train.csv"
TEST_PATH = DATA_DIR / "imdb_test.csv"
PARTITIONS_PATH = DATA_DIR / "imdb_partitions.csv"

MODEL_OUTPUT_PATH = "model/sentiment_model.keras"
PLOT_OUTPUT_PATH = "model/model_architecture.png"

RANDOM_SEED = 42

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.5

BATCH_SIZE = 16
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
from pathlib import Path


CHECKPOINT_ID = "distilbert/distilbert-base-uncased"
CHECKPOINT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"

MAX_LENGTH = 256
RANDOM_SEED = 42
LEARNING_RATE = 2e-5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "model" / "distilbert"
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.sentiment_model.config import RANDOM_SEED, VALIDATION_SIZE


def load_data(path: str) -> pd.DataFrame:
    """Load the sentiment dataset from a CSV file."""
    df = pd.read_csv(path)

    expected_columns = {"text", "label"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("The dataset must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def split_train_validation(df: pd.DataFrame):
    """Split the official IMDb train set into train and validation sets."""
    return train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )


def dataframe_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Convert a pandas DataFrame into a TensorFlow dataset."""
    ds = tf.data.Dataset.from_tensor_slices((df["text"].values, df["label"].values))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), seed=RANDOM_SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
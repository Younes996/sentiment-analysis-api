from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.sentiment_model.config import RANDOM_SEED


def load_train_validation(
    train_path: Path,
    partitions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_train_df = pd.read_csv(train_path)
    partitions_df = pd.read_csv(partitions_path)

    train_val_assignments = partitions_df.loc[
        partitions_df["partition"].isin(["train", "validation"])
    ]
    joined_df = pd.merge(
        train_val_assignments,
        official_train_df,
        how="left",
        on="review_id",
        validate="one_to_one",
    )
    if joined_df[["text", "label"]].isna().any().any():
        raise ValueError(
            "One or more train or validation assignments have no matching IMDb "
            "review or are missing a review text or label. "
            "Check that the partitions file and the official IMDb train CSV match."
        )
    train_df = joined_df.loc[joined_df["partition"] == "train"].copy()
    val_df = joined_df.loc[joined_df["partition"] == "validation"].copy()
    return train_df, val_df


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

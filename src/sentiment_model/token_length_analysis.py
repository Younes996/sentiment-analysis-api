"""Measure BiLSTM token lengths on the frozen IMDb training split."""

import tensorflow as tf

from src.sentiment_model.config import PARTITIONS_PATH, SEQUENCE_LENGTH, TRAIN_PATH
from src.sentiment_model.data import load_train_validation
from src.sentiment_model.preprocessing import custom_standardization


def main():
    train_df, _ = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)

    batch_size = 256
    lengths = []
    for start in range(0, len(train_df), batch_size):
        texts = train_df.iloc[start : start + batch_size]["text"].tolist()
        cleaned_texts = custom_standardization(tf.constant(texts))
        tokens = tf.strings.split(cleaned_texts)
        lengths.extend(tokens.row_lengths().numpy().tolist())

    total = len(lengths)
    print(f"IMDb training reviews measured: {total}")
    for threshold in dict.fromkeys((SEQUENCE_LENGTH, 250, 300)):
        count = sum(length > threshold for length in lengths)
        print(f"More than {threshold} tokens: {count}/{total} ({count / total:.2%})")


if __name__ == "__main__":
    main()
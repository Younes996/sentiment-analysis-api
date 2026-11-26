"""Measure WordPiece sequence lengths on the frozen IMDb training split."""

from transformers import AutoTokenizer

from src.distilbert_model.config import CHECKPOINT_ID, CHECKPOINT_REVISION
from src.sentiment_model.config import PARTITIONS_PATH, TRAIN_PATH
from src.sentiment_model.data import load_train_validation


def main():
    train_df, _ = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
    )

    batch_size = 256
    lengths = []
    for start in range(0, len(train_df), batch_size):
        texts = train_df.iloc[start:start + batch_size]["text"].tolist()
        batch = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            verbose=False,
        )
        lengths.extend(len(ids) for ids in batch["input_ids"])

    total = len(lengths)
    print(f"IMDb training reviews measured: {total}")
    for threshold in (128, 256, 384, 512):
        count = sum(length > threshold for length in lengths)
        print(f"More than {threshold} tokens: {count}/{total} ({count / total:.1%})")


if __name__ == "__main__":
    main()

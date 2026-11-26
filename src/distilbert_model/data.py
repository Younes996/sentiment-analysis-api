import torch
from pandas import DataFrame
from transformers import BatchEncoding, PreTrainedTokenizerBase

from src.distilbert_model.config import MAX_LENGTH


def encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    examples: DataFrame,
) -> tuple[BatchEncoding, torch.Tensor]:
    """Encode a batch of reviews and return their labels in the same order."""
    reviews = examples["text"].tolist()
    labels = examples["label"].tolist()

    model_inputs = tokenizer(
        reviews,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return model_inputs, label_tensor
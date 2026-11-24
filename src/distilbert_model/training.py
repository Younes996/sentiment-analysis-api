import argparse
import torch
from pandas import DataFrame
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, DistilBertForSequenceClassification, PreTrainedTokenizerBase

from src.distilbert_model.data import encode_batch
from src.distilbert_model.config import CHECKPOINT_ID, CHECKPOINT_REVISION

from src.sentiment_model.config import TRAIN_PATH, PARTITIONS_PATH
from src.sentiment_model.data import load_train_validation


def train_step(
    model: DistilBertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    model_inputs: BatchEncoding,
    labels: torch.Tensor,
) -> float:
    """Run one training step and return the batch loss before the update."""
    optimizer.zero_grad()

    loss = model(**model_inputs, labels=labels).loss
    loss.backward()

    optimizer.step()
    return loss.item()

def validation_step(
    model: DistilBertForSequenceClassification,
    model_inputs: BatchEncoding,
    labels: torch.Tensor,
) -> float:
    """Return the batch loss for a model already in evaluation mode."""
    with torch.inference_mode():
        loss = model(**model_inputs, labels=labels).loss

    return loss.item()


def train_epoch(
    model: DistilBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    examples: DataFrame,
    batch_size: int,
) -> float:
    """Train for one epoch and return the mean loss per example."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if examples.empty:
        raise ValueError("Training requires at least one example.")

    model.train()
    total_loss = 0.0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples.iloc[start:start + batch_size]
        batch_inputs, batch_labels = encode_batch(tokenizer, batch_examples)
        batch_loss = train_step(model, optimizer, batch_inputs, batch_labels)

        total_loss += batch_loss * len(batch_examples)

    return total_loss / len(examples)


def validate_epoch(
    model: DistilBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    examples: DataFrame,
    batch_size: int,
) -> float:
    """Return the mean validation loss per example."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if examples.empty:
        raise ValueError("Validation requires at least one example.")

    model.eval()
    total_loss = 0.0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples.iloc[start:start + batch_size]
        batch_inputs, batch_labels = encode_batch(tokenizer, batch_examples)
        batch_loss = validation_step(model, batch_inputs, batch_labels)

        total_loss += batch_loss * len(batch_examples)

    return total_loss / len(examples)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the DistilBERT smoke test."""
    parser = argparse.ArgumentParser(
        description="Run a DistilBERT smoke test on small IMDb subsets.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of examples per batch (default: 2).",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=4,
        help="Number of examples in the training set (default: 4).",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=3,
        help="Number of examples in the validation set (default: 3).",
    )

    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.train_limit <= 0:
        parser.error("--train-limit must be positive.")
    if args.val_limit <= 0:
        parser.error("--val-limit must be positive.")

    return args


def main() -> None:
    """Run a configurable smoke test on small IMDb subsets."""
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
    )

    torch.manual_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
        num_labels=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_df, val_df = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)
    train_examples = train_df.iloc[:args.train_limit]
    val_examples = val_df.iloc[:args.val_limit]

    print(f"Training set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")

    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model, tokenizer, optimizer, train_examples, args.batch_size
        )
        validation_loss = validate_epoch(
            model, tokenizer, val_examples, args.batch_size
        )

        print(f"Epoch {epoch}:")
        print(f"Training loss: {training_loss:.8f}")
        print(f"Validation loss: {validation_loss:.8f}")


if __name__ == "__main__":
    main()
from subprocess import check_output
from hashlib import sha256
import argparse
import json
import re
from platform import python_version

import torch
from pandas import DataFrame
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerBase,
    __version__
)

from src.distilbert_model.data import encode_batch
from src.distilbert_model.config import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MAX_LENGTH,
    RANDOM_SEED,
    LEARNING_RATE,
    RUNS_DIR
)

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
    head_only: bool = False,
) -> float:
    """Train for one epoch and return the mean loss per example."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if examples.empty:
        raise ValueError("Training requires at least one example.")

    model.train()
    if head_only:
        model.distilbert.eval()

    total_loss = 0.0
    total_batches = (len(examples) + batch_size - 1) // batch_size
    log_interval = 500

    for batch_number, start in enumerate(
        range(0, len(examples), batch_size),
        start=1,
    ):
        batch_examples = examples.iloc[start:start + batch_size]
        batch_inputs, batch_labels = encode_batch(tokenizer, batch_examples)
        batch_loss = train_step(model, optimizer, batch_inputs, batch_labels)

        total_loss += batch_loss * len(batch_examples)

        if batch_number % log_interval == 0 or batch_number == total_batches:
            print(f"Training batch {batch_number}/{total_batches}", flush=True)

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
    parser.add_argument(
            "--run-name",
            type=str,
            default="smoke_01",
            help="A simple folder name (default: smoke_01).",
        )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze the DistilBERT encoder and train only the classification head.",
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
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error(
            "--run-name must contain only letters (A-Z, a-z), "
            "digits (0-9), hyphens (-), and underscores (_)."
        )

    return args


def main() -> None:
    """Run a configurable smoke test on small IMDb subsets."""
    args = parse_args()

    run_dir = RUNS_DIR / f"seed_{RANDOM_SEED}" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
    )

    torch.manual_seed(RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
        num_labels=2,
    )
    if args.head_only:
        model.distilbert.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )

    train_df, val_df = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)
    train_examples = train_df.iloc[:args.train_limit]
    val_examples = val_df.iloc[:args.val_limit]

    print(f"Training set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")

    git_commit = check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    run_config = {
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "max_length": MAX_LENGTH,
        "seed": RANDOM_SEED,
        "learning_rate": LEARNING_RATE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "head_only": args.head_only,
        "train_limit": args.train_limit,
        "validation_limit": args.val_limit,
        "train_size": len(train_examples),
        "validation_size": len(val_examples),
        "python_version": python_version(),
        "torch_version": torch.__version__,
        "transformers_version": __version__,
        "partitions_sha256": sha256(PARTITIONS_PATH.read_bytes()).hexdigest(),
        "train_csv_sha256": sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "git_commit": git_commit,
    }
    config_path = run_dir / "config.json"
    with open(config_path, mode="w", encoding="utf-8") as c:
        json.dump(run_config, c, indent=2)

    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        training_loss = train_epoch(
            model, tokenizer, optimizer, train_examples, args.batch_size, args.head_only
        )
        validation_loss = validate_epoch(
            model, tokenizer, val_examples, args.batch_size
        )

        print(f"Training loss: {training_loss:.8f}")
        print(f"Validation loss: {validation_loss:.8f}")

        history.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        })

    history_path = run_dir / "history.json"
    with open(history_path, mode="w", encoding="utf-8") as h:
        json.dump(history, h, indent=2)

    checkpoint_dir = run_dir / "checkpoint"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    main()
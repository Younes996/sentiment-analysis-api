from subprocess import check_output
from platform import python_version
from pathlib import Path
import json
from hashlib import sha256

import tensorflow as tf

from src.sentiment_model.architecture import build_model, save_model_plot
from src.sentiment_model.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    EPOCHS,
    MAX_TOKENS,
    LSTM_UNITS,
    PARTITIONS_PATH,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAIN_PATH,
)
from src.sentiment_model.data import (
    dataframe_to_dataset,
    load_train_validation,
)
from src.sentiment_model.evaluation import (
    evaluate_model,
    summarize_best_validation,
)


def main(run_name="run_01"):
    run_dir = Path(f"model/seed_{RANDOM_SEED}/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=False)

    git_commit = check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    run_config = {
        "seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "max_tokens": MAX_TOKENS,
        "sequence_length": SEQUENCE_LENGTH,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "python_version": python_version(),
        "tensorflow_version": tf.__version__,
        "partitions_sha256": sha256(PARTITIONS_PATH.read_bytes()).hexdigest(),
        "train_csv_sha256": sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "git_commit": git_commit,
    }

    config_path = run_dir / Path("config.json")
    with open(config_path, mode="w", encoding="utf-8") as c:
        json.dump(run_config, c, indent=2)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    train_df, val_df = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)

    model = build_model(
        max_tokens=MAX_TOKENS,
        sequence_length=SEQUENCE_LENGTH,
        train_texts=train_df["text"].values,
    )

    print("\nModel summary:\n")
    model.summary()

    save_model_plot(model, run_dir / Path("model_architecture.png"))

    train_ds = dataframe_to_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = dataframe_to_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    history_path = run_dir / Path("history.json")
    with open(history_path, mode="w", encoding="utf-8") as h:
        json.dump(history.history, h, indent=2)


    print("\nFinal evaluation:")

    metrics = evaluate_model(model, train_ds, val_ds)

    print(
        f"Train loss: {metrics['train']['loss']:.4f} | "
        f"Train accuracy: {metrics['train']['accuracy']:.4f}"
    )
    print(
        f"Val   loss: {metrics['val']['loss']:.4f} | "
        f"Val   accuracy: {metrics['val']['accuracy']:.4f}"
    )

    best_validation = summarize_best_validation(history)

    print("\nBest validation results during training:")
    print(f"Best epoch (by val_loss): {best_validation['best_epoch']}")
    print(f"Best val_loss: {best_validation['best_val_loss']:.4f}")
    print(
        f"Val accuracy at best epoch: "
        f"{best_validation['best_val_accuracy']:.4f}"
    )

    model_path = run_dir / Path("sentiment_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
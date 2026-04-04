import os

import tensorflow as tf

from src.sentiment_model.architecture import build_model, save_model_plot
from src.sentiment_model.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MAX_TOKENS,
    MODEL_OUTPUT_PATH,
    PLOT_OUTPUT_PATH,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.sentiment_model.data import (
    dataframe_to_dataset,
    load_data,
    split_train_validation,
)
from src.sentiment_model.evaluation import (
    evaluate_model,
    summarize_best_validation,
)


def main():
    tf.random.set_seed(RANDOM_SEED)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test CSV not found: {TEST_PATH}")

    full_train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    train_df, val_df = split_train_validation(full_train_df)

    model = build_model(
        max_tokens=MAX_TOKENS,
        sequence_length=SEQUENCE_LENGTH,
        train_texts=train_df["text"].values,
    )

    print("\nModel summary:\n")
    model.summary()

    save_model_plot(model, PLOT_OUTPUT_PATH)

    train_ds = dataframe_to_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = dataframe_to_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False)
    test_ds = dataframe_to_dataset(test_df, batch_size=BATCH_SIZE, shuffle=False)

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

    print("\nFinal evaluation:")

    metrics = evaluate_model(model, train_ds, val_ds, test_ds)

    print(
        f"Train loss: {metrics['train']['loss']:.4f} | "
        f"Train accuracy: {metrics['train']['accuracy']:.4f}"
    )
    print(
        f"Val   loss: {metrics['val']['loss']:.4f} | "
        f"Val   accuracy: {metrics['val']['accuracy']:.4f}"
    )
    print(
        f"Test  loss: {metrics['test']['loss']:.4f} | "
        f"Test  accuracy: {metrics['test']['accuracy']:.4f}"
    )

    best_validation = summarize_best_validation(history)

    print("\nBest validation results during training:")
    print(f"Best epoch (by val_loss): {best_validation['best_epoch']}")
    print(f"Best val_loss: {best_validation['best_val_loss']:.4f}")
    print(
        f"Val accuracy at best epoch: "
        f"{best_validation['best_val_accuracy']:.4f}"
    )

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
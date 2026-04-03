import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.preprocessing import custom_standardization


# Reproducibility
tf.random.set_seed(42)


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


def build_model(max_tokens: int, sequence_length: int, train_texts):
    """Build a sentiment analysis model with integrated text preprocessing."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=custom_standardization,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorizer.adapt(train_texts)

    text_input = tf.keras.Input(shape=(1,), 
                                dtype=tf.string, 
                                name="text_input")
    x = vectorizer(text_input)
    x = tf.keras.layers.Embedding(input_dim=max_tokens, 
                                  output_dim=128, 
                                  mask_zero=True,
                                  name="token_embedding",
                                  )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, name="lstm_encoder"),
        name="lstm_encoder",
        )(x)
    x = tf.keras.layers.Dense(64, 
                              activation="relu", 
                              name="dense_classifier",
                              )(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_regularization")(x)
    output = tf.keras.layers.Dense(1, 
                                   activation="sigmoid", 
                                   name="sentiment_output",
                                   )(x)

    model = tf.keras.Model(inputs=text_input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_model_plot(model, output_path: str) -> None:
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
        )
        print(f"Model architecture plot saved to: {output_path}")
    except Exception as e:
        print("Could not generate model plot.")
        print("Reason:", e)
        print("If needed, install pydot and Graphviz, then try again.")

def dataframe_to_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["text"].values, df["label"].values))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), seed=42)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    train_path = "data/imdb_train.csv"
    test_path = "data/imdb_test.csv"
    model_output_path = "model/sentiment_model.keras"
    plot_output_path = "model/model_architecture.png"


    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    full_train_df = load_data(train_path)
    test_df = load_data(test_path)

    # Split the official IMDb train set into train and validation sets
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        stratify=full_train_df["label"],
        random_state=42,
    )

    max_tokens = 10000
    sequence_length = 200
    batch_size = 16
    epochs = 10 # let early stopping decide when to stop

    # Adapt the TextVectorization layer only on the real training texts
    model = build_model(
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        train_texts=train_df["text"].values,
    )


    print("\nModel summary:\n")
    model.summary()

    save_model_plot(model, plot_output_path)

    train_ds = dataframe_to_dataset(train_df, batch_size=batch_size, shuffle=True)
    val_ds = dataframe_to_dataset(val_df, batch_size=batch_size, shuffle=False)
    test_ds = dataframe_to_dataset(test_df, batch_size=batch_size, shuffle=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    print("\nFinal evaluation:")

    train_metrics = model.evaluate(train_ds, verbose=0, return_dict=True)
    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)

    print(f"Train loss: {train_metrics['loss']:.4f} | Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val   loss: {val_metrics['loss']:.4f} | Val   accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test  loss: {test_metrics['loss']:.4f} | Test  accuracy: {test_metrics['accuracy']:.4f}")

    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=lambda i: history.history["val_loss"][i]
        ) + 1

    best_val_loss = min(history.history["val_loss"])
    best_val_acc_at_best_loss_epoch = history.history["val_accuracy"][best_epoch - 1]

    print("\nBest validation results during training:")
    print(f"Best epoch (by val_loss): {best_epoch}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Val accuracy at best epoch: {best_val_acc_at_best_loss_epoch:.4f}")

    model.save(model_output_path)
    print(f"\nModel saved to: {model_output_path}")


if __name__ == "__main__":
    main()
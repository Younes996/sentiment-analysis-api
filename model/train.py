import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


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


def main():
    data_path = "data/sentiment.csv"
    model_output_path = "model/sentiment_model.keras"
    plot_output_path = "model/model_architecture.png"


    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["label"].values if df["label"].nunique() > 1 else None,
    )

    max_tokens = 10000
    sequence_length = 100
    batch_size = 32
    epochs = 5

    model = build_model(max_tokens, sequence_length, X_train)

    print("\nModel summary:\n")
    model.summary()

    save_model_plot(model, plot_output_path)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
    )

    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(model_output_path)
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    main()
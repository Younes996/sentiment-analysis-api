import tensorflow as tf

from src.sentiment_model.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    LSTM_UNITS,
)
from src.sentiment_model.preprocessing import custom_standardization


def build_model(max_tokens: int, sequence_length: int, train_texts):
    """Build a sentiment analysis model with integrated text preprocessing."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=custom_standardization,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorizer.adapt(train_texts)

    text_input = tf.keras.Input(
        shape=(1,),
        dtype=tf.string,
        name="text_input",
    )
    x = vectorizer(text_input)
    x = tf.keras.layers.Embedding(
        input_dim=max_tokens,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="token_embedding",
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS, name="lstm_encoder"),
        name="bidirectional_lstm",
    )(x)
    x = tf.keras.layers.Dense(
        DENSE_UNITS,
        activation="relu",
        name="dense_classifier",
    )(x)
    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_regularization",
    )(x)
    output = tf.keras.layers.Dense(
        1,
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
    """Save a PNG visualization of the model architecture."""
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
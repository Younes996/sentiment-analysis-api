import re
import string

import tensorflow as tf

def custom_standardization(input_text: tf.Tensor) -> tf.Tensor:
    """
    Standardize input text for sentiment analysis.

    Processing steps:
    - convert text to lowercase
    - remove HTML line breaks such as <br>, <br/>, and <br />
    - remove URLs starting with http, https, or www
    - replace problematic dash-like characters with spaces
    - remove control and non-printable characters
    - remove punctuation except apostrophes
    - collapse multiple whitespace characters into a single space
    - strip leading and trailing spaces

    Args:
        input_text: Input tensor containing raw text strings.

    Returns:
        A tensor of cleaned text strings.
    """
    text = tf.strings.lower(input_text)
    text = tf.strings.regex_replace(text, r"<br\s*/?>", " ")
    text = tf.strings.regex_replace(text, r"https?://\S+|www\.\S+", " ")

    # Replace problematic dash-like / weird chars
    text = tf.strings.regex_replace(text, "\u0096", " ")
    text = tf.strings.regex_replace(text, "[–—]", " ")

    # Remove control characters
    text = tf.strings.regex_replace(text, r"[\x00-\x1f\x7f-\x9f]", " ")

    punctuation_to_remove = string.punctuation.replace("'", "")
    text = tf.strings.regex_replace(
        text,
        f"[{re.escape(punctuation_to_remove)}]",
        ""
    )

    text = tf.strings.regex_replace(text, r"\s+", " ")
    text = tf.strings.strip(text)
    return text
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


def prepare_data(settings, raw_train_ds):
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=settings['max_features'],
        output_mode="int",
        output_sequence_length=settings['sequence_length'],
    )
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    return vectorize_layer

from tensorflow.keras import layers
from tensorflow.keras import Model


def build_model(vectorize_layer, settings):
    text_input = layers.Input(shape=(1,), dtype='string', name='text')
    x = vectorize_layer(text_input)
    x = layers.Embedding(settings['max_features'] + 1, settings['embedding_dim'])(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = Model(text_input, predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

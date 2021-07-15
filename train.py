from utils.data_load import get_training_data
from utils.data_preparation import prepare_data
from utils.model_building import build_model


def main(data_dir, batch_size, settings, model_dir):
    """
    don't forget to document your code guys.
    :param data_dir:
    :param batch_size:
    :param settings:
    :param model_dir:
    :return:
    """
    raw_train_ds, raw_val_ds = get_training_data(data_dir, batch_size)
    vectorize_layer = prepare_data(settings, raw_train_ds)
    model = build_model(vectorize_layer, settings)
    model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=settings['epochs'])
    model.save(model_dir)
    return model


if __name__ == '__main__':
    DATA_DIR = '../imdb/train'
    BATCH_SIZE = 32
    MODEL_CONSTANTS = {
        'max_features': 20000,
        'embedding_dim': 128,
        'sequence_length': 500,
        'epochs': 2
    }
    MODEL_DIR = "models"
    main(DATA_DIR, BATCH_SIZE, MODEL_CONSTANTS, MODEL_DIR)

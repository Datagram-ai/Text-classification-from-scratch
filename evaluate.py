from utils.data_load import get_training_data
from utils.data_load import get_testing_data
from tensorflow.keras.models import load_model
from utils.data_preparation import custom_standardization
import pandas as pd


def main(data_dir, model_dir):
    raw_test_ds = get_testing_data(data_dir)
    raw_train_ds, raw_val_ds = get_training_data(data_dir)
    model = load_model(model_dir)
    train_eval = model.evaluate(raw_train_ds)
    valid_eval = model.evaluate(raw_val_ds)
    test_eval = model.evaluate(raw_test_ds)
    evaluations = pd.DataFrame([train_eval, valid_eval, test_eval],
                               index=["Training", "Validation", "Testing"],
                               columns=["Loss", "Accuracy"])
    return evaluations


if __name__ == '__main__':
    DATA_DIR = "../imdb/test"
    MODEL_DIR = "models"
    evaluation = main(DATA_DIR, MODEL_DIR)
    print(evaluation)

from tensorflow.keras.preprocessing import text_dataset_from_directory


def get_training_data(data_dir, batch_size=32):
    raw_train_ds = text_dataset_from_directory(
        data_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )
    raw_val_ds = text_dataset_from_directory(
        data_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=1337,
    )
    return raw_train_ds, raw_val_ds


def get_testing_data(data_dir):
    raw_test_ds = text_dataset_from_directory(data_dir)
    return raw_test_ds
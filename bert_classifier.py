### This file contain functions related to the implementation of the BERT models found in tf_bert.py ###

import pandas as pd
import numpy as np
from functions import ingredients_to_string
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datetime import date, datetime
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def prepare_data(csv_path: str, num_observations: int, vegan: bool = False, reproducible: int = True):
    # prepares data for use in BERT
    use_col = 'is_vegan' if vegan else 'is_vegetarian'
    data = pd.read_csv(csv_path, header=0, usecols=['ingredients', use_col])
    if reproducible:
        data = data.sample(n=num_observations, random_state=2205)
    else:
        data = data.sample(n=num_observations)
    data['ingredients'] = data['ingredients'].apply(literal_eval)
    data['ingredients'] = data['ingredients'].apply(ingredients_to_string)
    pp_data = data[['ingredients', use_col]]
    pp_data.columns = ['text', 'labels']
    return pp_data


def balance_train_data(train_data):
    # balances proportion of classes in training data to not induce bias
    class_0 = train_data[train_data['labels'] == 0]
    class_1 = train_data[train_data['labels'] == 1]
    if class_0.shape[0] == class_1.shape[0]:
        return train_data
    minority = class_1 if class_0.shape[0] > class_1.shape[0] else class_0
    additional_samples = resample(minority, replace=True, n_samples=abs(class_0.shape[0] - class_1.shape[0]),
                                  random_state=2205)
    return pd.concat([train_data, additional_samples])


def split_data(prepared_data, test_size=0.1, train_size=0.5, reproducible=True):
    # splits data into train set, test set, and evaluation set and balances the train
    if reproducible:
        train_test = train_test_split(prepared_data, test_size=test_size, train_size=train_size, random_state=2205)
        eval_train = train_test_split(train_test[0], test_size=0.2, random_state=2205)
    else:
        train_test = train_test_split(prepared_data, test_size=test_size, train_size=train_size)
        eval_train = train_test_split(train_test[0], test_size=0.2)
    train_data = balance_train_data(eval_train[0])
    eval_data = eval_train[1]
    test_data = train_test[1]
    return train_data, eval_data, test_data


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    # converts pandas DataFrame to data set than can be used by the tensorflow BERT implementation
    AUTOTUNE = tf.data.AUTOTUNE
    dataframe = dataframe.copy()
    labels = dataframe.pop('labels')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def load_trained_model(save_file_path: str, preprocessed_data_path: str):
    # loads a saved model
    data_sample_size = 156250
    prep_data = prepare_data(preprocessed_data_path, data_sample_size)
    train_data, val_data, test_data = split_data(prep_data)
    train_ds = df_to_dataset(train_data, preprocessed_data_path)
    epochs = 100
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = np.float32(3e-5)
    optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    return tf.keras.models.load_model(save_file_path, custom_objects={'AdamWeightDecay': optimizer})


def predict_series(model, data: pd.core.series.Series, vegan:bool =False):
    # this function translates the numerical predictions made by the BERT models into binary
    # I did not know how to extract the thresholds out of the networks directly, so I looked empirically at the outputs
    # of the networks and how they relate to the actual labels of the data
    predictions = model.predict(data)
    if vegan:
        # the threshold for the vegan classifier for classifying a recipe as vegan is 0 (found out empirically)
        results = [1 if elem > 0 else 0 for elem in np.concatenate(predictions).ravel()]
    else:
        # the threshold for the vegetarian classifier for classifying a recipe as vegetarian is 9 (found out empirically)
        results = [1 if elem > 9 else 0 for elem in np.concatenate(predictions).ravel()]
    return pd.Series(results)

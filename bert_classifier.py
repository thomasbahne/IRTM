import pandas as pd
import numpy as np
from functions import ingredients_to_string
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datetime import date


def prepare_data(csv_path: str, num_observations: int, vegan: bool = False, reproducible: bool = True):
    # prepares data for use in BERT
    use_col = 'is_vegan' if vegan else 'is_vegetarian'
    print(use_col)
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

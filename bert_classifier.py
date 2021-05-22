import torch
import pandas as pd
import numpy as np
import wandb  # weights-and-biases framework: for experiment tracking and visualizing training in a web browser
import simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from functions import ingredients_to_string
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def prepare_data(csv_path: str, vegan: bool = False):
    # prepares data for use in BERT
    use_col = 'is_vegan' if vegan else 'is_vegetarian'
    data = pd.read_csv(csv_path, header=0, usecols=['ingredients', use_col], nrows=100)
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


def split_data(prepared_data, test_size=0.2):
    # splits data into train set and test set and balances the train set
    stratified_split = train_test_split(prepared_data, test_size=test_size, random_state=2205)
    stratified_split[0] = balance_train_data(stratified_split[0])
    return stratified_split

# cuda_available = torch.cuda.is_available()
# model_args = ClassificationArgs()
# model_args.do_lower_case = True  # necessary when using uncased models
# model_args.use_early_stopping = True
# model_args.early_stopping_delta = 0.01
# model_args.early_stopping_metric = "mcc"
# model_args.early_stopping_metric_minimize = False
# model_args.early_stopping_patience = 5
# model_args.evaluate_during_training_steps = 1000
#
# model = ClassificationModel(
#     model_type="bert", model_name="bert-base-uncased", args=model_args, use_cuda=cuda_available
# )

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


def split_data(prepared_data, test_size=0.2, random_state=True):
    # splits data into train set, test set, and evaluation set and balances the train
    if random_state:
        stratified_split = train_test_split(prepared_data, test_size=test_size, random_state=2205)
        eval_test_split = train_test_split(stratified_split[1], test_size=0.5, random_state=2205)
    else:
        stratified_split = train_test_split(prepared_data, test_size=test_size)
        eval_test_split = train_test_split(stratified_split[1], test_size=0.5)
    test_data = eval_test_split[0]
    eval_data = eval_test_split[1]
    train_data = balance_train_data(stratified_split[0])
    return train_data, test_data, eval_data


def get_model_trained_model(training_data, evaluation_data):
    model_name = 'bert-base-uncased'
    output_dir = '../../IRTM/networks/'
    cuda_available = torch.cuda.is_available()
    model_args = ClassificationArgs()
    model_args.best_model_dir = '../../IRTM/networks/best_model'
    model_args.do_lower_case = True  # necessary when using uncased models
    # model_args.use_early_stopping = True
    # model_args.early_stopping_delta = 0.01
    # model_args.early_stopping_metric = "mcc"
    # model_args.early_stopping_metric_minimize = False
    model_args.logging_steps = 1000
    model_args.early_stopping_patience = 5
    model_args.reprocess_input_data = True
    model_args.train_batch_size = 128
    model_args.num_train_epochs = 3
    model_args.save_model_every_epoch = False
    model_args.train_batch_size = 128
    model_args.save_eval_checkpoints = False
    model_args.evaluate_during_training = True
    model_args.eval_batch_size = 128
    model_args.overwrite_output_dir = True
    model_args.wandb_project = 'IRTM bert-base-uncased'
    model_args.wandb_kwargs = {'name': model_name}
    wandb.login(key='b8bb043ad17107ca5bd92da9114c41e106f8069a')
    model = ClassificationModel(
        model_type="bert", model_name=model_name, args=model_args, use_cuda=cuda_available
    )
    model.train_model(train_df=training_data, eval_df=evaluation_data, output_dir=output_dir)
    return model

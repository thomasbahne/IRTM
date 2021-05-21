import torch
import pandas as pd
import wandb        # weights-and-biases framework: for experiment tracking and visualizing training in a web browser
import simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs

cuda_available = torch.cuda.is_available()
model_args = ClassificationArgs()
model_args.do_lower_case = True         # necessary when using uncased models
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000

model = ClassificationModel(
    model_type="bert", model_name="bert-base-uncased", args=model_args, use_cuda=cuda_available
)

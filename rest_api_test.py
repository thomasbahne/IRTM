import requests
import pprint
import json

# api_key = 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F'
# query = {
#     'api_key': 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F',
#     #'dataType': 'Foundation',
#     #'foodTypes': 'Foundation',
#     'query': 'pepper',
#     'pageSize': 1
# }
# response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
# pprint.pprint(response.json())

import pandas as pd
import string
import matplotlib
from ast import literal_eval
from functions import tokenize, remove_measure_units_single_recipe, remove_measure_units

data = pd.read_csv('../IRTM Assignment/data/RAW_recipes.csv', header=0, usecols=['submitted', 'tags', 'steps', 'ingredients'], parse_dates=['submitted'], infer_datetime_format=True, nrows=100)
data['ingredients'] = data['ingredients'].apply(literal_eval)
data['steps'] = data['steps'].apply(literal_eval)
data['tags'] = data['tags'].apply(literal_eval)
data['instructions'] = data['steps'].apply('. '.join)
data['instructions'] = data['instructions'].str.replace(' , ', ', ')
data['pp_ingredients'] = remove_measure_units(path_reference_units='../../IRTM/recipe data/measure_units.csv', data=data['ingredients'])



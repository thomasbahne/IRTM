import requests
import pprint
import json
import pandas as pd
import string
import matplotlib
from ast import literal_eval
import time
import csv
import numpy as np
from functions import remove_noise_terms, plot_value_counts, request_database, categorize_multiple_ingredients, \
    categorize_recipes, batch_categorize_and_save, tokenize_fdc, remove_noise_terms_single_ingredient, \
    remove_noise_terms_multiple_ingredients, tokenize, similarity, \
    predict_food_category_of_request, categorize_single_ingredient, add_classes, is_vegetarian, is_vegan, \
    tagged_vegetarian

categorization_dict = pd.read_csv('../../IRTM/recipe data/categorized_foods_corrected.csv', header=None, index_col=0,
                                  squeeze=True, sep=';').to_dict()
noise_terms = pd.read_csv('../../IRTM/recipe data/noise_terms.csv', squeeze=True).to_numpy().tolist()
data = pd.read_csv('../../IRTM/recipe data/RAW_recipes.csv', header=0,
                   usecols=['submitted', 'tags', 'steps', 'ingredients'], parse_dates=['submitted'],
                   infer_datetime_format=True)
data['ingredients'] = data['ingredients'].apply(literal_eval)
# data['steps'] = data['steps'].apply(literal_eval)
data['tags'] = data['tags'].apply(literal_eval)
# data['instructions'] = data['steps'].apply('. '.join)
# data['instructions'] = data['instructions'].str.replace(' , ', ', ')
data['pp_ingredients'] = remove_noise_terms(noise_terms=noise_terms, data=data['ingredients'])
data['categories'] = data['pp_ingredients'].apply(add_classes, categorized_foods=categorization_dict)
data['vegetarian_tagged'] = data['tags'].apply(tagged_vegetarian)
data['is_vegetarian'] = data['categories'].apply(is_vegetarian)
data['is_vegan'] = data['categories'].apply(is_vegan)
tagged = data[data['vegetarian_tagged'] > 0]
tagged = tagged[tagged['is_vegetarian'] < 1]
print(tagged.shape)
print(tagged[['ingredients', 'categories']])
print('Ingredients:', tagged['ingredients'][94128])
print('Categories', tagged['categories'][94128])
print(sum(data['vegetarian_tagged']))
print(sum(data['is_vegetarian']))
print(sum(data['is_vegan']))
value_counts = plot_value_counts(data['categories'])
# batch_categorize_and_save(temp_category_save_path='../../IRTM/recipe data/temp_category_save.csv', batch_size=100, skip_batches=2400,
#                           data_path='../../IRTM/recipe data/RAW_recipes_copy.csv',
#                           path_noise_term_file='../../IRTM/recipe data/noise_terms.csv', num_batches=1,
#                           categorized_foods_path='../../IRTM/recipe data/categorized_foods.npy')
# data['categories'] = categorize_recipes(data['pp_ingredients'])

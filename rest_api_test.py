import requests
import pprint
import json
import pandas as pd
import string
import matplotlib
from ast import literal_eval
import time
from functions import remove_noise_terms, plot_value_counts, request_database, categorize_multiple_ingredients, \
        categorize_recipes, batch_categorize_and_save, tokenize_fdc, remove_noise_terms_single_ingredient, \
        remove_noise_terms_multiple_ingredients, jaccard_coefficient, tokenize, similarity, \
        predict_food_category_of_request, categorize_single_ingredients

api_key = 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F'
food = 'jalapeno'
page_size = 3
query = {
        'api_key': api_key,
        # either 'Foundation', 'SR Legacy', 'Survey (FNDDS)', or 'Branded'. Must be given as a list
        'dataType': ['Branded'],
        'query': food,
        'pageSize': page_size,          # maximum number of results shown for the current page
        'pageNumber': 1,                # determines offset of retrieved results: pageNumber*pageSize
        'sortBy': 'dataType.keyword',   # either dataType.keyword, description, fdcId, publishedDate
        # 'sortOrder':                  # either asc or desc
        # 'brandOwner':                 # only applies to foods of 'Branded' dataType
        # 'requireAllWords': False      # default is false
    }
# response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
# print(response.headers["X-RateLimit-Limit"])
# print(response.headers["X-RateLimit-Remaining"])
# pprint.pprint(response.json())


# noise_terms = pd.read_csv('../../IRTM/recipe data/noise_terms.csv', squeeze=True).to_numpy().tolist()
# data = pd.read_csv('../../IRTM/recipe data/RAW_recipes.csv', header=0, usecols=['submitted', 'tags', 'steps', 'ingredients'], parse_dates=['submitted'], infer_datetime_format=True, nrows=50)
# data['ingredients'] = data['ingredients'].apply(literal_eval)
# data['steps'] = data['steps'].apply(literal_eval)
# data['tags'] = data['tags'].apply(literal_eval)
# data['instructions'] = data['steps'].apply('. '.join)
# data['instructions'] = data['instructions'].str.replace(' , ', ', ')
# data['pp_ingredients'] = remove_noise_terms(noise_terms=noise_terms, data=data['ingredients'])
# data['pp_ingredients'] = remove_noise_terms(noise_terms=noise_terms, data=data['pp_ingredients'])
# value_counts = plot_value_counts(data['pp_ingredients'], lower_bound=70)
batch_categorize_and_save(temp_category_save_path='../../IRTM/recipe data/temp_category_save.csv', batch_size=100, skip_batches=100,
                          data_path='../../IRTM/recipe data/RAW_recipes_copy.csv',
                          path_noise_term_file='../../IRTM/recipe data/noise_terms.csv', num_batches=2000,
                          categorized_foods_path='../../IRTM/recipe data/categorized_foods.npy')
# data['categories'] = categorize_recipes(data['pp_ingredients'])



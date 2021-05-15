import requests
import pprint
import json
import pandas as pd
import string
import matplotlib
from ast import literal_eval
import time
from functions import remove_measure_units, plot_value_counts, request_database, categorize_foods, categorize_recipes, batch_categorize_and_save

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
#response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
#pprint.pprint(response.json())


# data = pd.read_csv('../IRTM Assignment/data/RAW_recipes.csv', header=0, usecols=['submitted', 'tags', 'steps', 'ingredients'], parse_dates=['submitted'], infer_datetime_format=True, nrows=10)
# data['ingredients'] = data['ingredients'].apply(literal_eval)
# data['steps'] = data['steps'].apply(literal_eval)
# data['tags'] = data['tags'].apply(literal_eval)
# data['instructions'] = data['steps'].apply('. '.join)
# data['instructions'] = data['instructions'].str.replace(' , ', ', ')
# data['pp_ingredients'] = remove_measure_units(path_reference_units='../../IRTM/recipe data/measure_units.csv', data=data['ingredients'])
# data['pp_ingredients'] = remove_measure_units(path_reference_units='../../IRTM/recipe data/measure_units.csv', data=data['pp_ingredients'])
batch_categorize_and_save(temp_category_save_path='../../IRTM/recipe data/temp_category_save.csv', batch_size=10,
                          skip_batches=1, data_path='../../IRTM/recipe data/RAW_recipes_copy.csv',
                          reference_units_path='../../IRTM/recipe data/measure_units.csv', num_batches=10,
                          categorized_foods_path='../../IRTM/recipe data/categorized_foods.csv')
# data['categories'] = categorize_recipes(data['pp_ingredients'])
# value_counts = plot_value_counts(data['pp_ingredients'], lower_bound=15)

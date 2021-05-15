import requests
import pprint
import json
import pandas as pd
import string
import matplotlib
from ast import literal_eval
import time
from functions import remove_measure_units, plot_value_counts, request_database, categorize_foods, categorize_recipes

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


data = pd.read_csv('../IRTM Assignment/data/RAW_recipes.csv', header=0, usecols=['submitted', 'tags', 'steps', 'ingredients'], parse_dates=['submitted'], infer_datetime_format=True, nrows=10)
data['ingredients'] = data['ingredients'].apply(literal_eval)
data['steps'] = data['steps'].apply(literal_eval)
data['tags'] = data['tags'].apply(literal_eval)
data['instructions'] = data['steps'].apply('. '.join)
data['instructions'] = data['instructions'].str.replace(' , ', ', ')
data['pp_ingredients'] = remove_measure_units(path_reference_units='../../IRTM/recipe data/measure_units.csv', data=data['ingredients'])
data['pp_ingredients'] = remove_measure_units(path_reference_units='../../IRTM/recipe data/measure_units.csv', data=data['pp_ingredients'])
start = time.time()
data['categories'] = categorize_recipes(data['pp_ingredients'])
end = time.time()
print(data['categories'][0])
print(data['categories'][1])
print(data['categories'])
minutes = (start-end)/60
seconds = (start-end) - 60*minutes
print('that took', minutes, 'minutes and', seconds, 'seconds.')
print('for all 230.000 recipes that would be roughly', (minutes*23000)/60, 'hours')
#value_counts = plot_value_counts(data['pp_ingredients'], lower_bound=15)

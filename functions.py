import Levenshtein as Levenshtein
import matplotlib.pyplot
import numpy as np
import pandas as pd
import requests
import string
from matplotlib import rcParams  # makes labels not run off the bottom of the graphic
import time
from ast import literal_eval
import os.path
import Levenshtein

# cleaning up the data:
# for the purpose of my analysis (finding out if a recipe is vegan/vegetarian/omnivore, I can disregard many
# descriptions of ingredients. Examples would be "fresh" or "mix", like in "fresh parsley", or "pudding mix".

# What I did to the data for the ingredients_one_list:
# Took "ingredients" column and fused the individual entries to a huge list - ingredients_one_list.csv
from typing import Tuple

data_folder_path = 'Data/'
filename_recipes = 'RAW_recipes.csv'
filename_ingredients = 'RAW_ingredients.csv'
filename_ingredients_one_list = 'ingredients_one_list.csv'


def plot_value_counts(data, upper_bound: int = None, lower_bound: int = None):
    # plots and return value counts for elements (ingredients) in data
    # lower and upper bounds put restrictions on which value counts are shown
    # specifying lower_bound => only ingredients with value counts > lower_bound are shown
    # specifying upper_bound => only ingredients with value counts < upper_bound are shown
    work_data = None
    if isinstance(data[0], list):
        work_data = data.explode()
    value_counts = work_data.value_counts(sort=True)
    rcParams.update({'figure.autolayout': True})  # makes labels not run off the bottom of the graphic
    if (upper_bound is None) & (lower_bound is None):
        value_counts.plot(kind='barh')
        matplotlib.pyplot.show()
        return
    elif upper_bound is None:
        mask = value_counts > lower_bound
    elif lower_bound is None:
        mask = value_counts < upper_bound
    else:
        mask = (value_counts < upper_bound) & (value_counts > lower_bound)
    value_counts = value_counts.loc[mask]
    rcParams.update({'figure.autolayout': True})  # makes labels not run off the bottom of the graphic
    value_counts.plot(kind='barh')
    matplotlib.pyplot.show()
    return value_counts


def find_occurrence(data, word: string):
    # finds all elements (ingredients) containing word in data
    temp = set()
    for item in data:
        for ingredient in item:
            if word in ingredient:
                temp.add(ingredient)
    return temp


def tokenize(phrase: str) -> list:
    # tokenizes ingredients: replaces (&) with (and), strips trailing commas, deletes numbers
    # splits ingredients that include the word "with" into two ingredients and eliminates "with"
    tokens = phrase.lower().split()
    tokens = ['and' if token == '&' else token for token in tokens]
    tokens = [token.rstrip(',') for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    if 'with' in tokens:
        after_with_tokens = tokens[tokens.index('with') + 1:]
        tokens = [t for t in tokens if t not in [*after_with_tokens, 'with']]
        if 'and' in after_with_tokens:
            after_with_after_and_tokens = after_with_tokens[after_with_tokens.index('and') + 1:]
            after_with_tokens = [t for t in after_with_tokens if t not in [*after_with_after_and_tokens, 'and']]
            return [*tokens, *after_with_tokens, *after_with_after_and_tokens]
        return [*tokens, *after_with_tokens]
    return tokens


def tokenize_fdc(phrase: str) -> list:
    # Food descriptions in FoodData Central database (FDC) have a different structure than foods in the recipe
    # collection. Therefore, they need a slightly different tokenizer
    tokens = phrase.lower().split(',')
    tokens = [token.rstrip(' ') for token in tokens]
    tokens = [token.lstrip(' ') for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    # to-do: remove 'measure unit' tokens
    return tokens


def remove_measure_units_single_ingredient(ingredient: str, noise_terms: list):
    # removes terms that are not necessary to determine if ingredient is vegan/vegetarian (noise terms)
    # Example of noise terms: 'fresh', 'canned', 'frozen', 'tablespoon', 'sugar-free')
    # noise terms are specified in external .csv file
    tokens: list = tokenize(ingredient)
    tokens = [token for token in tokens if token not in set(noise_terms)]
    return list(map(' '.join, tokens))


def remove_measure_units_multiple_ingredients(ingredients: list, noise_terms: list):
    # for explanation, see function 'remove_measure_units_single_ingredient'
    return [remove_measure_units_single_ingredient(ingredient=ingredient, noise_terms=noise_terms) for ingredient in
            ingredients]


def remove_measure_units(noise_terms: list, data: pd.core.series.Series):
    # for explanation, see function 'remove_measure_units_single_ingredient'
    return [remove_measure_units_multiple_ingredients(ingredients=recipe, noise_terms=noise_terms) for recipe in data]


def jaccard_coefficient(tokens1: set, tokens2: set) -> float:
    # proportion of identical tokens
    return float(len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2)))


def similarity(food_recipe: str, food_fdc: str) -> float:
    # custom measure for similarity of two phrases (between 0 and 1)
    # uses jaccard-coefficient as well as Levensthein distance
    # How it works:
    # 1) take all tokens in ingredient from recipe
    # 2) assign value 1 to tokens that also appear in the search result from the database
    # 3) for remaining tokens, find most similar token (by Levensthein-Distance) in the set of tokens from the database
    # 4) assign value exp(-0.2*Levensthein-Distance) to those tokens (factor of 0.2 chosen arbitrarily)
    # 5) take average of values of all tokens in ingrdient from recipe and multiply by jaccard-coefficient
    recipe_tokens: list = tokenize(food_recipe)
    fdc_tokens: list = tokenize_fdc(food_fdc)
    s1: set = set(recipe_tokens)
    s2: set = set(fdc_tokens)
    jacc_coefficient: float = jaccard_coefficient(s1, s2)
    unique_s1: set = s1.difference(s2)
    unique_s2: set = s2.difference(s1)
    if (jacc_coefficient == 1) or (len(unique_s1) == 0) or (len(unique_s2) == 0):
        return 1

    # Levensthein-Distance: number of operations to transform one string into another
    max_similarities = [min([Levenshtein.distance(token1, token2) for token2 in unique_s2]) for token1 in unique_s1]
    # to-do (optional): remove tokens from unique_s2 that are already used
    print('Recipe Tokens:', s1)
    print('FDC Tokens:', s2)
    print('Jaccard Coefficient:', jacc_coefficient)
    print('Max. similarities:', max_similarities)
    max_similarities = np.exp(-0.2*max_similarities)
    return jacc_coefficient*((len(s1.union(s2))+sum(max_similarities))/len(s1))


def request_database(food: str) -> Tuple[list, int]:
    # method to request a certain food from the USDA food nutrition database
    # returns None if the query did not find any hits
    first_nonempty_data_type: str = ''
    relevance = ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded']
    query = {
        'api_key': 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F',
        'dataType': ['Foundation'],  # list of 'Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded'
        'query': food,
        'pageSize': 1,  # maximum number of results shown for the current page
        'pageNumber': 1,  # determines offset of retrieved results: pageNumber*pageSize
        # 'sortBy': ,                   # either dataType.keyword, description, fdcId, publishedDate
        # 'sortOrder':                  # either asc or desc
        # 'brandOwner':                 # only applies to foods of 'Branded' dataType
        # 'requireAllWords':            # default is false
    }

    # send a "dummy" query to see which dataType has entries for that food
    response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
    num_hits_by_data_type: dict = response.json().get('aggregations').get('dataType')

    # take results of dataType which comes first in relevance and yields at least one results
    for key in relevance:
        if num_hits_by_data_type.get(key):
            first_nonempty_data_type = key
            break

    if first_nonempty_data_type == '':
        return [None], 0

    # get up to five results from the most relevant dataType
    page_size = max(num_hits_by_data_type.get(first_nonempty_data_type), 5)
    query.update({'dataType': first_nonempty_data_type,
                  'pageSize': page_size})
    response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
    foods = [{k: food[k] for k in food.keys() & {'dataType', 'lowercaseDescription', 'fdcId', 'foodCategory'}}
             for food in response.json().get('foods')]

    # keep track of how many requests were send to API, as there is a limit on 3600 queries per hour
    num_requests_made = page_size + 1

    return foods, num_requests_made


def predict_food_category(foods: list) -> str:
    # implement method that uses term similarity to determine best search result and take its food category
    # if query matches 100% with result, take that result!!!
    for food in foods:
        updated_description = list(map(tokenize_fdc, tokenize_fdc(food.get('lowercaseDescription'))))
    #    updated_description = # to-do: remove measure units
        food.update({'lowercaseDescription': updated_description})
    counters = {}
    for food in foods:
        counters[food.get('foodCategory')] = counters.get(food.get('foodCategory'), 0) + 1

    # when two categories have the same count, the category first added to counters is returned
    return max(counters, key=counters.get)


def get_food_category(food: str, categorized_foods: dict) -> Tuple[str, int]:
    if food in categorized_foods:
        return categorized_foods.get(food), 0
    else:
        hits, num_requests = request_database(food)
        if hits[0] is None:
            return 'Unidentified', num_requests

        categorized_foods.update({food: predict_food_category(hits)})
        return predict_food_category(hits), num_requests


def categorize_foods(foods: list, categorized_foods: dict = None, queue_request_times: list = None) -> list:
    if categorized_foods is None:
        categorized_foods: dict = {}

    if queue_request_times is None:
        queue_request_times = []
        cumulated_requests: int = 0
    else:
        cumulated_requests = sum(request for request, timestamp in queue_request_times)

    foods = iter(foods)
    categories: list = []

    while True:
        while cumulated_requests > 0:
            if queue_request_times[0][1] < time.time():
                cumulated_requests -= queue_request_times[0][0]
                del queue_request_times[0]
            else:
                break

        if cumulated_requests > 3500:
            print('Maximum number of requests per hour reached')
            print('Program will continue in ', (queue_request_times[0][1] - time.time()) / 60, 'minutes.')
            # implement that the method saves the progress it has made so far in a .csv file
            time.sleep(queue_request_times[0][1] - time.time())

        try:
            food = next(foods)
        except StopIteration:
            break

        category, num_requests = get_food_category(food, categorized_foods)
        queue_request_times.append([num_requests, time.time()])
        cumulated_requests += num_requests
        categories.append(category)

    return categories


def categorize_recipes(preprocessed_ingredients: pd.core.series.Series, categorized_foods: dict = None,
                       queue_request_times: list = None) -> pd.core.series.Series:
    if categorized_foods is None:
        categorized_foods: dict = {}
    if queue_request_times is None:
        queue_request_times = []
    recipe_categories: list = []

    for ingredients in preprocessed_ingredients:
        recipe_categories.append(categorize_foods(foods=ingredients, categorized_foods=categorized_foods,
                                                  queue_request_times=queue_request_times))

    return pd.Series(recipe_categories)


def batch_categorize_and_save(temp_category_save_path: str, batch_size: int, data_path: str,
                              reference_units_path: str, categorized_foods_path: str = '', num_batches: int = 1,
                              skip_batches: int = 0):
    if os.path.isfile(categorized_foods_path):
        temp = pd.read_csv(categorized_foods_path, index_col=0)
        categorized_foods = temp.to_dict("split")
        categorized_foods = dict(zip(categorized_foods["index"], categorized_foods["data"]))
    else:
        categorized_foods: dict = {}

    queue_request_times: list = []

    for batch_num in range(num_batches):
        data = pd.read_csv(data_path, header=0, usecols=['ingredients'], nrows=batch_size,
                           skiprows=range(1, 1 + (batch_size * batch_num) + skip_batches))
        data['ingredients'] = data['ingredients'].apply(literal_eval)
        data['pp_ingredients'] = remove_measure_units(path_reference_units=reference_units_path,
                                                      data=data['ingredients'])
        data['categories'] = categorize_recipes(data['pp_ingredients'], categorized_foods, queue_request_times)
        temp2 = pd.DataFrame.from_dict(categorized_foods, orient="index")
        temp2.to_csv(categorized_foods_path)
        print(f'Batch {batch_num + 1}/{num_batches} processed and saved.')
        if os.path.isfile(temp_category_save_path):
            data['categories'].to_csv(temp_category_save_path, mode='a', header=False)
        else:
            data['categories'].to_csv(temp_category_save_path, mode='a', header=['categories'])
        # work-in-progress: save categories of foods in the original data file as a new column
    pass

# only load useful columns with df = pd.read_csv("filepath", usecols=list_useful_column_names)
# specify data types to take less memory (e.g. for year-numbers use int.16 instead of int.64)
# command: df = pd.read_csv("train.csv", dtype={"column_name": "more_efficient_datatype"})
# "categorical" is a datatype that uses minimal space if the entries in a column only take on
# a few different values dtype={'column_name': 'category'}
# replace missing values when loading data with 1) defining a convert function, e.g.
# def convert(var): if var==np.nan: return "sth you want to have instead, or 0", return val
# pd.read_csv('filepath', converters={'column_name': convert_function})
# for test purposes, only load in a small part of the data with nrows=1000

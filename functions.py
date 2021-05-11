import matplotlib.pyplot
import pandas as pd
import requests
import string
from matplotlib import rcParams  # makes labels not run off the bottom of the graphic
from operator import itemgetter


# cleaning up the data:
# for the purpose of my analysis (finding out if a recipe is vegan/vegetarian/omnivore, I can disregard many
# descriptions of ingredients. Examples would be "fresh" or "mix", like in "fresh parsley", or "pudding mix".

# What I did to the data for the ingredients_one_list:
# Took "ingredients" column and fused the individual entries to a huge list - ingredients_one_list.csv

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


def tokenize(phrase: str):
    # tokenizes ingredients:
    # removes (,) and round brackets, replaces (&) with (and), strips trailing commas and deletes numbers
    if type(phrase) != str:
        print('input is not if type string, but type ', type(phrase))
        print(phrase)
    #tokens = list(filter(lambda a: a != (',' or '(' or ')'), phrase.lower().split()))
    #print('removing commas and brackets works')
    #tokens = ['and' if token == '&' else token for token in tokens]
    #tokens = [token.rstrip(',') for token in tokens]
    #tokens = [token for token in tokens if not token.isdigit()]
    #return tokens


def remove_measure_units_single_recipe(ingredients: list, reference_units: pd.core.series.Series):
    # removes all "measure units" and some other unnecessary descriptions specified in the reference list from
    # a single list of ingredients/recipe
    # reference list in stored in a .csv file (one row of strings)
    print(ingredients)
    print(type(ingredients))
    cleaned_ingredients = list(map(tokenize, ingredients))
    print('tokenizing works')
    print(cleaned_ingredients)
    print(type(cleaned_ingredients))
    print(cleaned_ingredients[0])
    print(type(cleaned_ingredients[0]))
    for tokens in cleaned_ingredients:
        print('selecting tokens works')
        for token in tokens:
            print('selecting token works')
            if token in reference_units.to_numpy().tolist():
                tokens.remove(token)
    cleaned_ingredients = list(map(' '.join, cleaned_ingredients))
    return cleaned_ingredients


def remove_measure_units(path_reference_units: str, data: pd.core.series.Series):
    # removes all "measure units" and some other unnecessary descriptions specified in the reference list from
    # a list of recipes (each having a list of ingredients)
    # reference list in stored in a .csv file (one row of strings)
    reference_units = pd.read_csv(path_reference_units, squeeze=True)
    print(type(reference_units))
    data_without_units = [remove_measure_units_single_recipe(ingredients=ingr, reference_units=reference_units) for ingr in data]
    return data_without_units


def jaccard_coefficient(strings1: list, strings2: list):
    s1 = set(strings1)
    s2 = set(strings2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def similarity(food1: str, food2: str):
    # "tokenization" - takes lowercase strings and splits them into words
    food1_tokens = list(filter(lambda a: a != ',', food1.lower().split()))
    food2_tokens = list(filter(lambda a: a != ',', food2.lower().split()))
    # implement method to compare similarity of food in recipe vs. food gotten by  REST API
    # tags given as input should not have commas
    # Jaccard-Coefficient: proportion of identical tokens
    # Levensthein-Distance: number of coefficients to transform one sting into another
    pass


def request_database(food: str):
    # method to request a certain food from the USDA food nutrition database
    num_requests: int = 0
    page_size: int = 1
    query = {
        'api_key': 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F',
        'dataType': 'Foundation',
        'foodTypes': 'Foundation',
        'query': food,
        'pageSize': page_size
    }
    # account for how many requests are made perhour (3600/hour max!)
    # send a "dummy" query first to see if "Foundation" database has entries for that food
    # if yes, take those. If no, search in "SR Legacy", then "Survey (FNDDS), then "Branded".
    response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)


def get_food_category(food: str, categorized_foods: dict):
    if food in categorized_foods:
        return dict[food]
    else:
        # request database and determine food_category
        pass
    # return food category


def add_categorized_food(food: str, category: str, categorized_foods: dict):
    if food not in categorized_foods:
        categorized_foods[food] = category
    else:
        print(food, ' already exists in dictionary.')


# only load useful columns with df = pd.read_csv("filepath", usecols=list_useful_column_names)
# specify data types to take less memory (e.g. for year-numbers use int.16 instead of int.64)
# command: df = pd.read_csv("train.csv", dtype={"column_name": "more_efficient_datatype"})
# "categorical" is a datatype that uses minimal space if the entries in a column only take on
# a few different values dtype={'column_name': 'category'}
# replace missing values when loading data with 1) defining a convert function, e.g.
# def convert(var): if var==np.nan: return "sth you want to have instead, or 0", return val
# pd.read_csv('filepath', converters={'column_name': convert_function})
# for test purposes, only load in a small part of the data with nrows=1000

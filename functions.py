import matplotlib.pyplot
import numpy as np
import pandas as pd
import requests
from matplotlib import rcParams  # makes labels not run off the bottom of the graphic
import time
from ast import literal_eval
import os.path
import Levenshtein
import csv

# cleaning up the data:
# for the purpose of my analysis (finding out if a recipe is vegan/vegetarian/omnivore, I can disregard many
# descriptions of ingredients. Examples would be "fresh" or "mix", like in "fresh parsley", or "pudding mix".

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


def find_occurrence(data, word: str) -> set:
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
    # tokens = [sub_token for token in tokens for sub_token in token.split()]
    # this would split 'grade a' into 'grade' and 'a'
    tokens = [token for token in tokens if not token.isdigit()]
    return tokens


def remove_noise_terms_single_ingredient(ingredient: str, noise_terms: list, use_fdc_tokenizer=False) -> str:
    # removes terms that are not necessary to determine if ingredient is vegan/vegetarian (noise terms)
    # Example of noise terms: 'fresh', 'canned', 'frozen', 'tablespoon', 'sugar-free')
    # noise terms are specified in external .csv file
    if use_fdc_tokenizer:
        tokenizer = tokenize_fdc
    else:
        tokenizer = tokenize
    tokens: list = tokenizer(ingredient)
    tokens = [token for token in tokens if token not in set(noise_terms)]
    return ' '.join(tokens)


def remove_noise_terms_multiple_ingredients(ingredients: list, noise_terms: list):
    # for explanation, see function 'remove_noise_terms_single_ingredient'
    # applies function remove_noise_terms_single_ingredient to multiple ingredients
    return [remove_noise_terms_single_ingredient(ingredient=ingredient, noise_terms=noise_terms) for ingredient in
            ingredients]


def remove_noise_terms(noise_terms: list, data: pd.core.series.Series):
    # for explanation, see function 'remove_measure_units_single_ingredient'
    # applies function remove_noise_terms_multiple_ingredients to an entire series of recipes
    return [remove_noise_terms_multiple_ingredients(ingredients=recipe, noise_terms=noise_terms) for recipe in data]


def similarity(food_recipe: str, food_fdc: str, noise_terms: list) -> float:
    # custom measure for similarity of two phrases (similarity is always a value between 0 and 1)
    # uses Levensthein distance and devalues when number of tokens is unequal (inspired by Jaccard-Coefficient)
    # How it works:
    # 1) tokenize ingredients
    # 2) take all tokens from ingredient in recipe and assign them a value:
    # 2.a) assign value 1 to tokens that also appear in the search result from FDC (FoodData Central database)
    # 2.b) assign exp(-0.2*max(Levensthein_Distance(token, tokens from search result)) to tokens not appearing in result
    # from the database. Factor of 0.2 is chosen arbitrarily, I just needed a function that has value 1 at x=0 and
    # is strictly decreasing, while asymptotically reaching 0
    # 3) average the value of all tokens in ingredient
    # 4) multiply average with exp(-0.1*|num_tokens_recipe - num_tokens_FDC|) to 'punish' difference in number of tokens
    recipe_tokens: list = tokenize(food_recipe)
    fdc_tokens = remove_noise_terms_single_ingredient(ingredient=food_fdc, noise_terms=noise_terms,
                                                      use_fdc_tokenizer=True).split()
    s1: set = set(recipe_tokens)
    s2: set = set(fdc_tokens)
    if len(s1) == 0:
        print('from similarity: length 0 tokens:', food_recipe)
    unique_s1: set = s1.difference(s2)
    unique_s2: set = s2.difference(s1)
    length_difference = abs(len(s1) - len(s2))
    # if I don't check if sets are empty, min-function below will return ValueError

    # Levensthein-Distance: number of operations to transform one string into another
    try:
        min_levensthein_distances = np.array([min([Levenshtein.distance(token1, token2) for token2 in unique_s2])
                                              for token1 in unique_s1])
    except ValueError:
        if (len(unique_s1) == 0) or (len(unique_s2) == 0):
            return np.exp(-0.1 * length_difference)
        else:
            raise ValueError('Unknown reason for error, please investigate...')
    # to-do (optional): remove tokens from unique_s2 that are already used
    min_levensthein_distances = np.exp(-0.2 * min_levensthein_distances)
    return np.exp(-0.1 * length_difference) * ((len(s1.intersection(s2)) + sum(min_levensthein_distances)) / len(s1))


def request_database(food: str) -> list:
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
    print(f'Remaining requests: {response.headers["X-RateLimit-Remaining"]}')
    while int(response.headers["X-RateLimit-Remaining"]) < 30:
        print('Close to reaching rate limit, sleeping for 60 seconds')
        time.sleep(60)
        response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
    try:
        num_hits_by_data_type: dict = response.json().get('aggregations').get('dataType')
    except AttributeError:
        return [None]

    # take results of dataType which comes first in relevance and yields at least one results
    for data_type in relevance:
        if num_hits_by_data_type.get(data_type):
            first_nonempty_data_type = data_type
            break

    if first_nonempty_data_type == '':
        return [None]

    # get up to five results from the most relevant dataType
    page_size = min(num_hits_by_data_type.get(first_nonempty_data_type), 5)
    query.update({'dataType': first_nonempty_data_type,
                  'pageSize': page_size})
    response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
    foods = [{k: food[k] for k in food.keys() & {'dataType', 'lowercaseDescription', 'fdcId', 'foodCategory'}}
             for food in response.json().get('foods')]

    # keep track of how many requests were send to API, as there is a limit on 3600 queries per hour

    return foods


def predict_food_category_of_request(food_recipe: str, fdc_foods: list, noise_terms: list) -> str:
    # method that uses similarity to determine quality of search result and food category
    counters = {}
    for food in fdc_foods:
        counters[food.get('foodCategory')] = counters.get(food.get('foodCategory'), 0) + similarity(
            food_recipe=food_recipe, food_fdc=food.get('lowercaseDescription'), noise_terms=noise_terms)

    # when two categories have the same count, the category first added to counters is returned
    return max(counters, key=counters.get)


def categorize_single_ingredient(food: str, categorized_foods: dict, noise_terms: list) -> str:
    # searches in list of already categorized foods and returns category if found, otherwise send request to FDC
    # database and predicts category given the search results from the database
    if food == '':
        return 'Unidentified'
    if food in categorized_foods:
        return categorized_foods.get(food)
    else:
        hits = request_database(food)
        if hits[0] is None:
            categorized_foods.update({food: 'Unidentified'})
            return 'Unidentified'

        predicted_category = predict_food_category_of_request(food_recipe=food, fdc_foods=hits, noise_terms=noise_terms)
        categorized_foods.update({food: predicted_category})
        return predicted_category


def categorize_multiple_ingredients(foods: list, noise_terms: list, categorized_foods: dict = None) -> list:
    # applies function categorize_single_ingredient to a list of ingredients
    if categorized_foods is None:
        categorized_foods: dict = {}

    foods = iter(foods)
    categories: list = []

    while True:
        try:
            food = next(foods)
        except StopIteration:
            break

        category = categorize_single_ingredient(food, categorized_foods, noise_terms=noise_terms)
        categories.append(category)

    return categories


def categorize_recipes(preprocessed_ingredients: pd.core.series.Series, noise_terms: list,
                       categorized_foods: dict = None) -> pd.core.series.Series:
    # applies function categorize_multiple_ingredients to an entire series of recipes
    if categorized_foods is None:
        categorized_foods: dict = {}
    recipe_categories: list = []

    for ingredients in preprocessed_ingredients:
        recipe_categories.append(categorize_multiple_ingredients(foods=ingredients, noise_terms=noise_terms,
                                                                 categorized_foods=categorized_foods))

    return pd.Series(recipe_categories)


def append_csv(data_frame: pd.core.frame.DataFrame, file_path: str, sep=",") -> None:
    if not os.path.isfile(file_path):
        data_frame.to_csv(file_path, mode='a', index=False, sep=sep)
    else:
        data_frame.to_csv(file_path, mode='a', index=False, sep=sep, header=False)


def batch_categorize_and_save(temp_category_save_path: str, batch_size: int, data_path: str,
                              path_noise_term_file: str, categorized_foods_path: str = '', num_batches: int = 1,
                              skip_batches: int = 0) -> None:
    # function to categorize ingredients in batches, due to API constraints on maximum number of requests per hour
    if os.path.isfile(categorized_foods_path):
        categorized_foods = np.load(categorized_foods_path, allow_pickle='TRUE').item()
    else:
        categorized_foods: dict = {}

    noise_terms = pd.read_csv(path_noise_term_file, squeeze=True).to_numpy().tolist()

    for batch_num in range(num_batches):
        data = pd.read_csv(data_path, header=0, usecols=['ingredients'], nrows=batch_size,
                           skiprows=range(1, 1 + (batch_size * batch_num) + skip_batches * batch_size))
        data['ingredients'] = data['ingredients'].apply(literal_eval)
        data['pp_ingredients'] = remove_noise_terms(noise_terms, data=data['ingredients'])
        data['categories'] = categorize_recipes(preprocessed_ingredients=data['pp_ingredients'],
                                                noise_terms=noise_terms, categorized_foods=categorized_foods)
        np.save(categorized_foods_path, categorized_foods)
        print(f'Batch {batch_num + 1}/{num_batches} processed and saved.')
        append_csv(data['categories'], temp_category_save_path)


def add_classes(pp_ingredients: list, categorized_foods: dict) -> list:
    # for a recipe (= list of ingredients), return a list of categories for those ingredients
    return [categorized_foods[ingredient] if ingredient in categorized_foods else 'Missing' for ingredient in
            pp_ingredients]


def is_vegetarian(categories: list) -> int:
    # identifies if a recipe (=list of ingredients) is vegetarian
    non_vegetarian = ['Beef Products', 'Fish and Seafood', 'Lamb, Veal, and Game Products', 'Poultry Products',
                      'Pork Products', 'Sausages and Luncheon Meats']
    for category in categories:
        if category in non_vegetarian:
            return 0
    return 1


def is_vegan(categories: list) -> int:
    # identifies if a recipe (=list of ingredients) is vegan
    non_vegan = ['Beef Products', 'Fish and Seafood', 'Lamb, Veal, and Game Products', 'Poultry Products',
                 'Pork Products', 'Sausages and Luncheon Meats', 'Dairy and Egg Products', 'Non vegan sweets',
                 'Vegetarian substitutes', 'Wine', 'Non-vegan Liquor']
    for category in categories:
        if category in non_vegan:
            return 0
    return 1


def tagged_vegetarian(tags: list) -> int:
    # identifies if a recipe was tagged as vegetarian by the user
    for tag in tags:
        if tag == 'vegetarian':
            return 1
    return 0


def load_npy_dict(file_path: str = '../../IRTM/recipe data/categorized_foods.npy') -> dict:
    # return a dictionary that was stored in a .npy file
    return np.load(file_path, allow_pickle='TRUE').item()


def write_npy_dict_to_csv(save_location: str = '../../IRTM/recipe data/categorized_foods.csv',
                          npy_path: str = '../../IRTM/recipe data/categorized_foods.npy') -> None:
    # converts and saves dict stored as .npy to .csv
    npy_dict: dict = load_npy_dict(npy_path)
    with open(save_location, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in npy_dict.items():
            writer.writerow([key, value])


def read_dict_from_csv(csv_path: str = '../../IRTM/recipe data/categorized_foods.csv') -> dict:
    # return dict that was saved in a .csv
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        return dict(reader)


def false_vegetarian_tag_by_user(data: pd.core.frame.DataFrame, categorized_foods: dict) -> pd.core.frame.DataFrame:
    # returns a DataFrame of ingredients and categories, which were tagged as vegetarian by the user, but not show
    # up as vegetarian according to my analysis
    work_data = data
    work_data['categories'] = data['pp_ingredients'].apply(add_classes, categorized_foods=categorized_foods)
    data['vegetarian_tagged'] = data['tags'].apply(tagged_vegetarian)
    data['is_vegetarian'] = data['categories'].apply(is_vegetarian)
    tagged = data[data['vegetarian_tagged'] > 0]
    tagged = tagged[tagged['is_vegetarian'] < 1]
    print('Number of recipes falsely tagged:', tagged.shape[0])
    return tagged[['ingredients', 'categories']]


def ingredients_to_string(ingredients: list):
    return ', '.join(ingredients)

# only load useful columns with df = pd.read_csv("filepath", usecols=list_useful_column_names)
# specify data types to take less memory (e.g. for year-numbers use int.16 instead of int.64)
# command: df = pd.read_csv("train.csv", dtype={"column_name": "more_efficient_datatype"})
# "categorical" is a datatype that uses minimal space if the entries in a column only take on
# a few different values dtype={'column_name': 'category'}
# replace missing values when loading data with 1) defining a convert function, e.g.
# def convert(var): if var==np.nan: return "sth you want to have instead, or 0", return val
# pd.read_csv('filepath', converters={'column_name': convert_function})
# for test purposes, only load in a small part of the data with nrows=1000

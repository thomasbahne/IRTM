import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from functions import remove_noise_terms, plot_value_counts, batch_categorize_and_save, add_classes, is_vegetarian,\
    is_vegan, counts_per_year, cohens_kappa
from bert_classifier import prepare_data, split_data
import logging

data_folder_path = '../../IRTM/recipe data/'

### preprocessing data ###

# batch_categorize_and_save(temp_category_save_path=data_folder_path+'temp_category_save.csv', batch_size=100,
#                           data_path=data_folder_pat+'RAW_recipes_copy.csv',
#                           path_noise_term_file=data_folder_path + 'noise_terms.csv', num_batches=1,
#                           categorized_foods_path=data_folder_path + 'categorized_foods.npy')
#
# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0, index_col=0,
#                    parse_dates=['submitted'], infer_datetime_format=True)
# categorization_dict = pd.read_csv(data_folder_path + 'categorized_foods_corrected.csv', header=None, index_col=0,
#                                    squeeze=True, sep=';').to_dict()
# noise_terms = pd.read_csv(data_folder_path + 'noise_terms.csv', squeeze=True).to_numpy().tolist()
# data['ingredients'] = data['ingredients'].apply(literal_eval)
# data['pp_ingredients'] = remove_noise_terms(noise_terms=noise_terms, data=data['ingredients'])
# data['categories'] = data['pp_ingredients'].apply(add_classes, categorized_foods=categorization_dict)
# data['is_vegetarian'] = data['categories'].apply(is_vegetarian)
# data['is_vegan'] = data['categories'].apply(is_vegan)
# data.to_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')


### get sample for other judges for kappa distance ###

# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0)
# data.columns = ['index', 'submitted', 'ingredients', 'pp_ingredients', 'categories', 'vegetarian', 'vegan']
# simplified = data.drop(axis=1, labels=['index', 'submitted', 'pp_ingredients', 'categories', 'vegetarian',
#                                        'vegan'])
# sample = simplified.sample(n=100, random_state=2205)
# sample.to_csv('../../IRTM/recipe data/kappa_samples_annotated.csv')


### calculate kappa distances ###

judge_data = [pd.read_csv(data_folder_path + 'kappa_samples_annotated.csv', header=0),
              pd.read_csv(data_folder_path + 'kappa_sample_charlotte.csv', header=0, sep=';'),
              pd.read_csv(data_folder_path + 'kappa_sample_margit.csv', header=0, sep=';'),
              pd.read_csv(data_folder_path + 'kappa_sample_wolfgang.csv', header=0, sep=';')]

kappa_dist_vegetarian = np.empty(shape=(4, 4))
for i in range(4):
    for j in range(4):
        kappa_dist_vegetarian[i, j] = cohens_kappa(judge_data[i]['vegetarian'], judge_data[j]['vegetarian'])
print('Vegetarian:')
print(kappa_dist_vegetarian)
print()

kappa_dist_vegan = np.empty(shape=(4, 4))
for i in range(4):
    for j in range(4):
        kappa_dist_vegan[i, j] = cohens_kappa(judge_data[i]['vegan'], judge_data[j]['vegan'])
print('Vegan:')
print(kappa_dist_vegan)

### create counts per year ###

data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0,
                   parse_dates=['submitted'], infer_datetime_format=True)
counts = counts_per_year(data)
# counts.to_csv('../../IRTM/recipe data/counts_by_year.csv')

y = np.vstack([counts['vegan'], counts['vegetarian'], counts['other']])
labels = ["Vegan ", "Vegetarian, but not vegan", "Neither vegan nor vegetarian"]
fig, ax = plt.subplots()
ax.stackplot(counts['year'].tolist(), y, labels=labels, baseline='wiggle')
ax.legend(loc='upper left')
plt.show()
# plt.stackplot(counts['year'], counts.drop(axis=1, labels=['year']), baseline='zero')

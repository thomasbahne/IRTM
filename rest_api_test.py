##### Please be aware that this code cannot simply be run as is. I went through many iterations and changes during #####
##### the implementation and I did not write a final script that one could run without making any changes. #####


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from functions import remove_noise_terms, plot_value_counts, batch_categorize_and_save, add_classes, is_vegetarian, \
    is_vegan, counts_per_year, cohens_kappa, proportions_per_year, precision, recall, f1_score
from bert_classifier import prepare_data, split_data, predict_series, load_trained_model
import logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from datetime import datetime

data_folder_path = '../../IRTM/recipe data/'

##### preprocessing data #####

# noise_terms = pd.read_csv(data_folder_path + 'noise_terms.csv', squeeze=True).to_numpy().tolist()

### Generating preliminary categories for every ingredient by contacting the FoodData Central Database API ###

# batch_categorize_and_save(temp_category_save_path=data_folder_path+'temp_category_save.csv', batch_size=100,
#                           data_path=data_folder_pat+'RAW_recipes_copy.csv',
#                           path_noise_term_file=data_folder_path + 'noise_terms.csv', num_batches=1,
#                           categorized_foods_path=data_folder_path + 'categorized_foods.npy')

# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0, index_col=0,
#                    parse_dates=['submitted'], infer_datetime_format=True)
# categorization_dict = pd.read_csv(data_folder_path + 'categorized_foods_corrected.csv', header=None, index_col=0,
#                                    squeeze=True, sep=';').to_dict()
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

# judge_data = [pd.read_csv(data_folder_path + 'kappa_samples_annotated.csv', header=0),
#               pd.read_csv(data_folder_path + 'kappa_sample_charlotte.csv', header=0, sep=';'),
#               pd.read_csv(data_folder_path + 'kappa_sample_margit.csv', header=0, sep=';'),
#               pd.read_csv(data_folder_path + 'kappa_sample_wolfgang.csv', header=0, sep=';')]
#
# kappa_dist_vegetarian = np.empty(shape=(4, 4))
# for i in range(4):
#     for j in range(4):
#         kappa_dist_vegetarian[i, j] = cohens_kappa(judge_data[i]['vegetarian'], judge_data[j]['vegetarian'])
# print('Vegetarian:')
# print(kappa_dist_vegetarian)
# print()
#
# kappa_dist_vegan = np.empty(shape=(4, 4))
# for i in range(4):
#     for j in range(4):
#         kappa_dist_vegan[i, j] = cohens_kappa(judge_data[i]['vegan'], judge_data[j]['vegan'])
# print('Vegan:')
# print(kappa_dist_vegan)

### create counts and proportions of categories per year ###

# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0,
#                    parse_dates=['submitted'], infer_datetime_format=True)
# counts = counts_per_year(data)
# proportions = proportions_per_year(counts)
# counts.to_csv('../../IRTM/recipe data/counts_by_year.csv')
# proportions.to_csv('../../IRTM/recipe data/proportions_by_year.csv')
#
# count_plot_data = {'vegan': counts['vegan'],
#                    'Vegetarian, but not vegan': counts['vegetarian'],
#                    'Neither vegan nor vegetarian': counts['other']}
# years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
# fig_count, ax_count = plt.subplots()
# ax_count.stackplot(counts['year'].tolist(), count_plot_data.values(), labels=count_plot_data.keys())
# plt.xticks(ticks=years)
# ax_count.legend(loc='upper left')
# ax_count.set_title('Counts of recipe uploads by category')
# ax_count.set_xlabel('Year')
# ax_count.set_ylabel('Number of uploads')
# plt.show()
#
# prop_plot_data = {'vegan': proportions['vegan'],
#                   'Vegetarian, but not vegan': proportions['vegetarian'],
#                   'Neither vegan nor vegetarian': proportions['other']}
# fig_prop, ax_prop = plt.subplots()
# ax_prop.stackplot(proportions['year'].tolist(), prop_plot_data.values(), labels=prop_plot_data.keys())
# plt.xticks(ticks=years)
# ax_prop.legend(loc='upper left')
# ax_prop.set_title('Proportion of recipe uploads by category')
# ax_prop.set_xlabel('Year')
# ax_prop.set_ylabel('Proportion of uploads')
# plt.show()

### load saved BERT model to classify all recipes ###

# path_vegetarian_bert = '../../IRTM/networks/small_bert_vegetarian'
# path_vegan_bert = '../../IRTM/networks/small_bert_vegan'
# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0, index_col=0)
#
# vegan_model = load_trained_model(path_vegan_bert, data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')
# vegetarian_model = load_trained_model(path_vegetarian_bert, data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')
# data['vegan_bert'] = predict_series(vegan_model, data['ingredients'], vegan=True)
# data['vegetarian_bert'] = predict_series(vegetarian_model, data['ingredients'])
# correct_vegan = data[data['is_vegan'] == data['vegan_bert']]
# correct_vegetarian = data[data['is_vegetarian'] == data['vegetarian_bert']]
# print('Accuracy vegan:', correct_vegan.shape[0]/data.shape[0])
# print('Accuracy vegetarian:', correct_vegetarian.shape[0]/data.shape[0])
# data.to_csv(data_folder_path + 'fully_preprocessed_data/results_bert.csv')

### calculate performance metrics of classification ###

# data_classified = pd.read_csv(data_folder_path + 'fully_preprocessed_data/results_bert.csv', header=0, index_col=0)
# precision_vegan = precision(reference=data_classified['is_vegan'], classifier=data_classified['vegan_bert'])
# precision_vegetarian = precision(reference=data_classified['is_vegetarian'],
#                                  classifier=data_classified['vegetarian_bert'])
# recall_vegan = recall(reference=data_classified['is_vegan'], classifier=data_classified['vegan_bert'])
# recall_vegetarian = recall(reference=data_classified['is_vegetarian'], classifier=data_classified['vegetarian_bert'])
# f1_vegan = f1_score(reference=data_classified['is_vegan'], classifier=data_classified['vegan_bert'])
# f1_vegetarian = f1_score(reference=data_classified['is_vegetarian'], classifier=data_classified['vegetarian_bert'])
# print(f'Performance of the vegan classifier: \n Precision: {precision_vegan} \n Recall: {recall_vegan} \n F1_score: '
#       f'{f1_vegan}')
# print(f'Performance of the vegetarian classifier: \n Precision: {precision_vegetarian} \n Recall: {recall_vegetarian} '
#       f'\n F1_score: {f1_vegetarian}')

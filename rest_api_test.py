import pandas as pd
from ast import literal_eval
from functions import remove_noise_terms, plot_value_counts, batch_categorize_and_save, add_classes, is_vegetarian,\
    is_vegan, counts_per_year
from bert_classifier import prepare_data, split_data
import logging

data_folder_path = '../../IRTM/recipe data/'

### preprocessing data ###

# batch_categorize_and_save(temp_category_save_path=data_folder_path+'temp_category_save.csv', batch_size=100,
#                           data_path=data_folder_pat+'RAW_recipes_copy.csv',
#                           path_noise_term_file=data_folder_path + 'noise_terms.csv', num_batches=1,
#                           categorized_foods_path=data_folder_path + 'categorized_foods.npy')
#
# categorization_dict = pd.read_csv(data_folder_path + 'categorized_foods_corrected.csv', header=None, index_col=0,
#                                   squeeze=True, sep=';').to_dict()
# noise_terms = pd.read_csv(data_folder_path + 'noise_terms.csv', squeeze=True).to_numpy().tolist()
# data = pd.read_csv(data_folder_path + 'RAW_recipes.csv', header=0, usecols=['submitted', 'ingredients'],
#                    parse_dates=['submitted'], infer_datetime_format=True)
# data['ingredients'] = data['ingredients'].apply(literal_eval)
# data['pp_ingredients'] = remove_noise_terms(noise_terms=noise_terms, data=data['ingredients'])
# data['categories'] = data['pp_ingredients'].apply(add_classes)
# data['is_vegetarian'] = data['categories'].apply(is_vegetarian)
# data['is_vegan'] = data['categories'].apply(is_vegan)
# data.to_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')


### get sample for other judges for kappa distance ###

# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0)
# data.columns = ['index', 'submitted', 'ingredients', 'pp_ingredients', 'categories', 'is_vegetarian', 'is_vegan']
# simplified = data.drop(axis=1, labels=['index', 'submitted', 'pp_ingredients', 'categories', 'is_vegetarian',
#                                        'is_vegan'])
# sample = simplified.sample(n=100, random_state=2205)
# sample.to_csv('../../IRTM/recipe data/kappa_samples_annotated.csv')


### calculate kappa distances ###

kappa_judge0 = pd.read_csv(data_folder_path + 'kappa_samples_annotated.csv', header=0)
kappa_judge1 = pd.read_csv(data_folder_path + 'kappa_sample_charlotte.csv', header=0)
kappa_judge2 = pd.read_csv(data_folder_path + 'kappa_sample_margit.csv', header=0)
kappa_judge3 = pd.read_csv(data_folder_path + 'kappa_sample_wolfgang.csv', header=0)

### create counts per year ###

# data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0,
#                    parse_dates=['submitted'], infer_datetime_format=True)
# counts = counts_per_year(data)
# counts.to_csv('../../IRTM/recipe data/counts_by_year.csv')

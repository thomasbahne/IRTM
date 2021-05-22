import pandas as pd
from ast import literal_eval
from functions import remove_noise_terms, plot_value_counts, batch_categorize_and_save, add_classes, is_vegetarian,\
    is_vegan
from bert_classifier import prepare_data, balance_train_data, split_data
from sklearn.model_selection import train_test_split

data_folder_path = '../../IRTM/recipe data/'
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

prep_data = prepare_data(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')
d = split_data(prep_data)
# stratified_split = train_test_split(prep_data, test_size=0.2, random_state=2205)
# x_train = stratified_split[0]['text']
# y_train = stratified_split[0]['labels']
# x_test = stratified_split[1]['text']
# y_test = ['labels']
x_train = d[0]['text']
y_train = d[0]['labels']
x_test = d[1]['text']
y_test = d[1]['labels']
print(x_train, y_train.value_counts())

# data['steps'] = data['steps'].apply(literal_eval)
# data['tags'] = data['tags'].apply(literal_eval)
# data['instructions'] = data['steps'].apply('. '.join)
# data['instructions'] = data['instructions'].str.replace(' , ', ', ')
# batch_categorize_and_save(temp_category_save_path=data_folder_path+'temp_category_save.csv', batch_size=100,
#                           data_path=data_folder_pat+'RAW_recipes_copy.csv',
#                           path_noise_term_file=data_folder_path + 'noise_terms.csv', num_batches=1,
#                           categorized_foods_path=data_folder_path + 'categorized_foods.npy')

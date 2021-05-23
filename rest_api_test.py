import pandas as pd
from ast import literal_eval
from functions import remove_noise_terms, plot_value_counts, batch_categorize_and_save, add_classes, is_vegetarian,\
    is_vegan
from bert_classifier import prepare_data, split_data, get_trained_model
import logging

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

# if __name__ == '__main__':
#
#     logging.basicConfig(level=logging.INFO)
#     transformers_logger = logging.getLogger("transformers")
#     transformers_logger.setLevel(logging.WARNING)
#
#     prep_data = prepare_data(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv')
#     train_data, eval_data, test_data = split_data(prep_data)
#     model = get_trained_model(training_data=train_data, evaluation_data=eval_data)
#     result, model_outputs, wrong_predictions = model.eval_model(test_data)
#     print(result)
#     print(wrong_predictions)
#     predictions, raw_outputs = model.predict(['milk, salt, macaroni, cheese, fresh coarse ground black pepper, butter, onions, garlic cloves'])
#     print(predictions)

# data['steps'] = data['steps'].apply(literal_eval)
# data['tags'] = data['tags'].apply(literal_eval)
# data['instructions'] = data['steps'].apply('. '.join)
# data['instructions'] = data['instructions'].str.replace(' , ', ', ')
# batch_categorize_and_save(temp_category_save_path=data_folder_path+'temp_category_save.csv', batch_size=100,
#                           data_path=data_folder_pat+'RAW_recipes_copy.csv',
#                           path_noise_term_file=data_folder_path + 'noise_terms.csv', num_batches=1,
#                           categorized_foods_path=data_folder_path + 'categorized_foods.npy')

data = pd.read_csv(data_folder_path + 'fully_preprocessed_data/preprocessed_recipes.csv', header=0)
data.columns = ['index', 'submitted', 'ingredients', 'pp_ingredients', 'categories', 'is_vegetarian', 'is_vegan']
simplified = data.drop(axis=1, labels=['index', 'submitted', 'pp_ingredients', 'categories', 'is_vegetarian', 'is_vegan'])
print(simplified.head())
sample = simplified.sample(n=100, random_state=2205)
print(sample.head)
sample.to_csv('../../IRTM/recipe data/kappa_sample.csv')

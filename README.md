## IRTM

Repository for my IRTM practical assignment

- Task: Categorize recipes into categories omnivore, vegetarian, vegan by their list of ingredients or their instructions
- Data: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

Function definitions related to everything but the BERT implementations can be found in functions.py. All other functions are declared in either bert_classifier.py or tf_bert.py. The file tf_bert.py also contains the implementatition of the BERT models (instructions for the implementation taken from https://www.tensorflow.org/tutorials/text/classify_text_with_bert and altered to fit my needs).

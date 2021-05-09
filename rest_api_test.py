import requests
import pprint
import json

api_key = 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F'
query = {
    'api_key': 'SoQlD3ha3vNDGM9ReqhdRBEj2j2EcYWaoIJipX5F',
    #'dataType': 'Foundation',
    #'foodTypes': 'Foundation',
    'query': 'pepper',
    'pageSize': 1
}
query['query'] = 'bread'
response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=query)
pprint.pprint(response.json())
#f_list = response.json()['foods']
#print(f_list[0]['foodCategory'])
#for key in response.json():
    #print(key, ":", response.json()[key])
# print(json.dumps(response.json(), indent=2))

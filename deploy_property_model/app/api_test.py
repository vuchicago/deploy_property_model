#%%
uvicorn main:app --reload

import requests

url = 'http://127.0.0.1:8000/predict_property'
data = {'Description': "John Baethke"}

response = requests.post(url, json=data)
print(response.text)


#%%
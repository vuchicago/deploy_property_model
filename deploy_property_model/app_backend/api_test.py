#%%
uvicorn main:app --reload


#%%
import requests
#%%

url = 'http://127.0.0.1:8000/predict_property_credit'
data = {'Description': "John Baethke was here"}

response = requests.post(url, json=data)
print(response.text)


#%%

data = {'Description': "I like shopping at Menards"}

response = requests.post(url, json=data)
print(response.text)

# %%
#%%
data = {'Description': "Keon lives here"}

response = requests.post(url, json=data)
print(response.text)
# %%
#%%
data = {'Description': "Tom and Kat bugs us a lot"}

response = requests.post(url, json=data)
print(response.text)

# %%
data = {'Description': "John Baethke"}

response = requests.post(url, json=data)
print(response.text)
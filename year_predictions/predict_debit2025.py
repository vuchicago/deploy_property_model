#%%
import os
import pandas as pd
import numpy as np
import requests
import json

dir = '/Users/vuchicago/Python/deploy_property_model/model_pipeline'
dir_data = os.path.join(dir, 'data/train')
print(dir)
os.chdir(dir)

#%%
from create_data import ChaseExpenseRead, FileType, TypeCC
debit_csv_container = ChaseExpenseRead(FileType.csv, TypeCC.debit)
debit_xlsx_container = ChaseExpenseRead(FileType.xlsx, TypeCC.debit)
credit_csv_container = ChaseExpenseRead(FileType.csv, TypeCC.credit)
credit_xlsx_container = ChaseExpenseRead(FileType.xlsx, TypeCC.credit)

#%%
os.chdir(dir_data)
##normally it's Havu taxes 
debit_file = 'Havu_debit_2025.csv'

# %%
havu_debit = debit_csv_container.read(debit_file)
havu_debit['Descriptions_all'] = havu_debit['Descriptions_all'].fillna('').astype(str)
havu_debit.head()
# %%
###call api http://localhost:8000 to get prediction on debit maximum probabilty
def get_prediction_property_debit(Description):
    url = 'http://localhost:8000/predict_property_debit'
    headers = {"Content-Type": "application/json"}
    data = {"Description": Description}  # Adjust the key to match the expected input
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['property'][2:-2]
    else:
        response.raise_for_status()

# Example usage

# %%
havu_debit['Property_pred'] = havu_debit.apply(lambda x: get_prediction_property_debit(x['Descriptions_all']), axis=1)

# %%
##label predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

debit2024='Havu_debit2024_fixed.xlsx'
df=debit_xlsx_container.read(debit2024)
df_model = df[['Descriptions_all', 'Amount', 'Label_pred']].copy()
df_model = df_model.dropna(subset=['Descriptions_all', 'Amount', 'Label_pred'])
df_model['Descriptions_all'] = df_model['Descriptions_all'].astype(str)
df_prop_debit = df_model[['Descriptions_all', 'Amount']]

y = df_model['Label_pred']


#%%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_prop_debit[['Descriptions_all','Amount']], y, test_size=0.2, random_state=1000)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['Descriptions_all'])
X_test_vec = vectorizer.transform(X_test['Descriptions_all'])

#%%
model_label_debit= LogisticRegression()
model_label_debit.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model_label_debit.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


model_label_debit = LogisticRegression()
vectorizer=TfidfVectorizer()
df_train_vec=vectorizer.fit_transform(df_prop_debit['Descriptions_all'])
X_pred_vec = vectorizer.transform(havu_debit['Descriptions_all'])
model_label_debit.fit(df_train_vec, y)
labels=model_label_debit.predict(X_pred_vec) ##Property predictions from debit

#%%

havu_debit['Label_pred']=[label for label in labels]
havu_debit.head()
# %%
havu_debit.to_excel('Havu_debit2025_fixed.xlsx', index=False)
# %%

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
credit_file = 'Havu_credit2025.CSV'

# %%
havu_credit = credit_csv_container.read(credit_file)
havu_credit['Descriptions_all']=ChaseExpenseRead.convert_credit(havu_credit)
havu_credit.head()
# %%
###call api http://localhost:8000 to get prediction on debit maximum probabilty
def get_prediction_property_debit(Description):
    url = 'http://localhost:8000/predict_property_credit'
    headers = {"Content-Type": "application/json"}
    data = {"Description": Description}  # Adjust the key to match the expected input
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['property'][2:-2]
    else:
        response.raise_for_status()

# Example usage

# %%
havu_credit['Property_pred'] = havu_credit.apply(lambda x: get_prediction_property_debit(x['Descriptions_all']), axis=1)
havu_credit.head()
# %%
##label predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

credit_old='Havu_credit2023_fixed.xlsx'
df=credit_xlsx_container.read(credit_old)
df_prop_credit = df[['Descriptions_all','Amount']]
col_prop=['Descriptions_all','Amount','Label_pred']
df_prop_credit = df[['Descriptions_all','Amount']]
y = df['Label_pred']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_prop_credit[['Descriptions_all','Amount']], y, test_size=0.2, random_state=1000)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['Descriptions_all'])
X_test_vec = vectorizer.transform(X_test['Descriptions_all'])

#%%
model_label_credit= LogisticRegression()
model_label_credit.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model_label_credit.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


model_label_credit = LogisticRegression()
vectorizer=TfidfVectorizer()
df_train_vec=vectorizer.fit_transform(df_prop_credit['Descriptions_all'])
X_pred_vec = vectorizer.transform(havu_credit['Descriptions_all'])
model_label_credit.fit(df_train_vec, y)
labels=model_label_credit.predict(X_pred_vec) ##Property predictions from debit

#%%

havu_credit['Label_pred']=[label for label in labels]

# %%
havu_credit.to_excel('havu_credit2025_fixed.xlsx', index=False)
# %%

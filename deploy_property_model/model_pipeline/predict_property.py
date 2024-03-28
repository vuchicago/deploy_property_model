#%%
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb



#%%
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
from create_data import chase_debit_read

#%%
data_path=os.path.join(dir,'data')

dir_output=os.path.join(data_path,'train/intermediate')
os.chdir(dir_output)

#%%
X_train=pd.read_parquet('X_train_prop.parquet')
y_train=pd.read_parquet('y_train_prop.parquet')
X_val=pd.read_parquet('X_val_prop.parquet')
y_val=pd.read_parquet('y_val_prop.parquet')

#%%
os.chdir(dir)
X_pred=chase_debit_read('2023_debit.CSV')

X_train_full=pd.concat([X_train,X_val],axis=0)
y_train_full=pd.concat([y_train,y_val],axis=0)

#%%
# Train the Logistic Regression model
model_property_credit = LogisticRegression()
model_property_credit.fit(X_train_full, y_train_full)

# Make predictions on the test set
y_pred = model_property_credit.predict(X_val)



#%%
# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

#%%

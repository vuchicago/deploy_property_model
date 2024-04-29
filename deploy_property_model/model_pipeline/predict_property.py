#%%
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
from scipy.sparse import hstack


#%%
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
from create_data import ChaseExpenseRead, FileType,TypeCC
#%%


debit_csv_container=ChaseExpenseRead(FileType.csv,TypeCC.debit)
debit_xlsx_container=ChaseExpenseRead(FileType.xlsx,TypeCC.debit)
credit_csv_container=ChaseExpenseRead(FileType.csv,TypeCC.credit)
credit_xlsx_container=ChaseExpenseRead(FileType.xlsx,TypeCC.credit)
#%%

debit2022='havu_debit2022_fixed.xlsx'
debit2023='2023_debit.CSV'

havu_debit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Debit')
havu_debit2021['Descriptions_all']=havu_debit2021['Description']+' '+havu_debit2021['Details']+' '+havu_debit2021['Type']
havu_debit2022=debit_xlsx_container.read(debit2022)
#havu_debit2022=pd.read_excel('havu_debit2022_fixed.xlsx')
#%%

#%%
#havu_debit2021=debit_xlsx_container.read(debit2021)
# #havu_debit2022=debit_xlsx_container.read(debit2022)
havu_debit2023=debit_csv_container.read(debit2023)

credit2022='havu_credit_2022_fixed.xlsx'
credit2023='2023_credit.CSV'
havu_credit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Credit')
#havu_credit2021['Descriptions_all']=havu_credit2021['Category'] + ' ' + havu_credit2021['Description']+ ' ' + havu_credit2021['Type']
havu_credit2021['Descriptions_all']=ChaseExpenseRead.convert_credit(havu_credit2021)
havu_credit2022=credit_xlsx_container.read(credit2022)
#%%
havu_credit2023=credit_csv_container.read(credit2023)
havu_credit2023


#%%
        
havu_debit2021['Property'].loc[havu_debit2021['Property']=='laramie']='Laramie'
havu_debit2021['Property'].loc[havu_debit2021['Property']=='Anthony']='Laramie'


col_keep=['Descriptions_all','Amount','Property','Label']
df=pd.concat([havu_debit2021[col_keep],havu_debit2022[col_keep]],axis=0)
df.dropna(inplace=True)

#%%

df_prop_debit = df[['Descriptions_all','Amount']]
y = df['Property']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_prop_debit[['Descriptions_all','Amount']], y, test_size=0.2, random_state=1000)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['Descriptions_all'])
X_test_vec = vectorizer.transform(X_test['Descriptions_all'])
#X_train_vec=hstack([X_train_vec, X_train['Amount'].values.reshape(-1, 1)])
#X_test_vec=hstack([X_test_vec, X_test['Amount'].values.reshape(-1, 1)])

#X_pred = hstack([X_pred_vec, df_prop['Amount'].values.reshape(-1, 1)])
#df_train=pd.DataFrame(X_train_vec.todense())
#df_test=pd.DataFrame(X_test_vec.todense())
#%%
model_property_debit= LogisticRegression()
model_property_debit.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model_property_debit.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#%%
###Fit model on entire dataset
model_property_debit = LogisticRegression()
vectorizer_debit=TfidfVectorizer()
df_train_vec=vectorizer_debit.fit_transform(df_prop_debit['Descriptions_all'])
X_pred_vec = vectorizer_debit.transform(havu_debit2023['Descriptions_all'])
model_property_debit.fit(df_train_vec, y)
df_property_debit2023_pred=model_property_debit.predict(X_pred_vec) ##Property predictions from debit



########################Property Predictions for Credit
###############################################################
#%%
col_prop=['Descriptions_all','Amount','Property']
df_prop_cc=pd.concat([havu_credit2021[col_prop],havu_credit2022[col_prop]])
X_train, X_test, y_train, y_test = train_test_split(df_prop_cc[['Descriptions_all','Amount']], df_prop_cc['Property'], test_size=0.2, random_state=19)

model_property_credit = LogisticRegression()
vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train['Descriptions_all'])
X_test_vec=vectorizer.transform(X_test['Descriptions_all'])


#X_train_vec=hstack([X_train_vec, X_train['Amount'].values.reshape(-1, 1)])
#X_test_vec=hstack([X_test_vec, X_test['Amount'].values.reshape(-1, 1)])
model_property_credit.fit(X_train_vec,y_train)

y_pred=model_property_credit.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#%%

###Fit model on entire dataset
model_property_credit = LogisticRegression()
vectorizer_credit=TfidfVectorizer()
df_train_vec=vectorizer_credit.fit_transform(df_prop_cc['Descriptions_all'])
y=df_prop_cc['Property']
X_pred_vec = vectorizer_credit.transform(havu_credit2023['Descriptions_all'])

model_property_credit.fit(df_train_vec, y)
df_property_credit2023_pred=model_property_credit.predict(X_pred_vec) ##Property predictions from debit


#%%
pd.DataFrame(df_property_credit2023_pred).to_csv('df_prop_pred_credit.csv',index=False)
pd.DataFrame(df_property_debit2023_pred).to_csv('df_prop_pred_debit.csv',index=False)

pickle.dump(model_property_debit, open("property_pred_debit.pickle", "wb"))
pickle.dump(model_property_credit, open("property_pred_credit.pickle", "wb"))
pickle.dump(vectorizer_debit, open("vectorizer_prop_debit.pickle", "wb"))
pickle.dump(vectorizer_credit, open("vectorizer_prop_credit.pickle", "wb"))

#%%
from pydantic import BaseModel
from typing import Any

class TextInput(BaseModel):
        text_input: str


#%%
class PropertyPredict(BaseModel):
        vectorizer: Any #Model.logistic  # Specify the type based on your actual vectorizer class, using Any for now as a placeholder
        model: Any       # Similarly, specify the type according to your model class
        
        def predict(self,text_input:TextInput):
                input=self.vectorizer.transform([text_input.text_input])
                output=self.model.predict(input)
                return output

#%%





#%%
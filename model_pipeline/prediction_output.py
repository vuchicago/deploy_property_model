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

dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
from create_data import ChaseExpenseRead, FileType,TypeCC
#%%


debit_csv_container=ChaseExpenseRead(FileType.csv,TypeCC.debit)
debit_xlsx_container=ChaseExpenseRead(FileType.xlsx,TypeCC.debit)
credit_csv_container=ChaseExpenseRead(FileType.csv,TypeCC.credit)
credit_xlsx_container=ChaseExpenseRead(FileType.xlsx,TypeCC.credit)

debit2023='2023_debit.CSV'
credit2023='2023_credit.CSV'
havu_debit2023=debit_csv_container.read(debit2023)
havu_credit2023=credit_csv_container.read(credit2023)

#%%
with open("label_pred_credit.pickle", "rb+") as pickle_file:
        label_pred_credit = pickle.load(pickle_file)
with open("label_pred_debit.pickle", "rb+") as pickle_file:
        label_pred_debit = pickle.load(pickle_file)
with open("property_pred_credit.pickle", "rb+") as pickle_file:
        property_pred_credit = pickle.load(pickle_file)
with open("property_pred_debit.pickle", "rb+") as pickle_file:
        property_pred_debit = pickle.load(pickle_file)
# %%
df_property_credit_pred=pd.read_csv('df_prop_pred_credit.csv',names=['Property_pred'],skiprows=1)
df_property_debit_pred=pd.read_csv('df_prop_pred_debit.csv',names=['Property_pred'],skiprows=1)
df_label_credit_pred=pd.read_csv('df_label_pred_credit.csv',names=['Label_pred'],skiprows=1)
df_label_debit_pred=pd.read_csv('df_label_pred_debit.csv',names=['Label_pred'],skiprows=1)
# %%

df_credit_pred=pd.concat([df_property_credit_pred,df_label_credit_pred],axis=1)
df_debit_pred=pd.concat([df_property_debit_pred,df_label_debit_pred],axis=1)
#%%
df_credit2023_fixed=pd.merge(havu_credit2023,df_credit_pred,left_index=True,right_index=True)
df_debit2023_fixed=pd.merge(havu_debit2023,df_debit_pred,left_index=True,right_index=True)

#%%
df_credit2023_fixed.to_excel('Havu_credit2023_fixed.xlsx',index=False)
df_debit2023_fixed.to_excel('Havu_debit2023_fixed.xlsx',index=False)
#%%
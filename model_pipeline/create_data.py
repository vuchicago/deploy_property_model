#%%
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pyarrow as pa
from enum import Enum
import pickle





from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
#%%
dir = os.path.dirname(os.path.realpath(__file__))
data_path=os.path.join(dir,'data')
data_train_path=os.path.join(data_path,'train')
os.chdir(data_train_path)
#%%
havu_debit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Debit')

havu_credit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Credit')

havu_debit2022=pd.read_csv('2022_debit.csv')
havu_credit2022=pd.read_csv('2022 Havu CC.CSV')

print(havu_debit2021['Posting Date'].min(),havu_debit2021['Posting Date'].max())
#%%

havu_credit2021.fillna({'Category':"Unknown"}, inplace=True)
havu_credit2021['Descriptions_all']=havu_credit2021['Category'] + ' ' + havu_credit2021['Description']+ ' ' + havu_credit2021['Type']
havu_credit2022['Descriptions_all'] = havu_credit2022['Category'] +' ' +  havu_credit2022['Description'] + ' ' +  havu_credit2022['Type']

vectorizer = TfidfVectorizer()
#%%
class FileType(Enum): #inherit Enum to make tuple fixed type
    csv = 'csv'
    parquet = 'parquet'
    pickle = 'pickle'
    shp = 'shp'
    json = 'json'
    xlsx ='xlsx'

class TypeCC(Enum):
        credit='credit'
        debit='debit'
        
class ChaseExpenseRead:
        def __init__(self, file_type:FileType,money_type:TypeCC):
               self.type=file_type
               self.money=money_type
        def read(self,file_input):
                self.df=pd.DataFrame([0])
                if self.type==FileType.csv and (self.money==TypeCC.credit):
                        self.df=pd.read_csv(file_input,index_col=False)
                        self.df[['Description','Category','Type']]=self.df[['Description','Category','Type']].replace(np.nan,"")
                        self.df['Descriptions_all']=self.df['Description'] + ' ' + self.df['Category']+ ' ' + self.df['Type']
                elif self.type==FileType.csv and (self.money==TypeCC.debit):
                        self.df=pd.read_csv(file_input,index_col=False)
                        self.df[['Description','Details','Type']]=self.df[['Description','Details','Type']].replace(np.nan,"")
                        self.df['Descriptions_all']=self.df['Description']+' '+self.df['Details']+' '+self.df['Type']
                elif self.type==FileType.xlsx and (self.money==TypeCC.credit):
                        self.df=pd.read_excel(file_input)
                        self.df[['Description','Category','Type']]=self.df[['Description','Category','Type']].replace(np.nan,"")
                        self.df['Descriptions_all']=self.df['Description'] + ' ' + self.df['Category']+ ' ' + self.df['Type']
                elif self.type==FileType.xlsx and (self.money==TypeCC.debit):
                        self.df=pd.read_excel(file_input)
                        self.df[['Description','Details','Type']]=self.df[['Description','Details','Type']].replace(np.nan,"")
                        self.df['Descriptions_all']=self.df['Description']+' '+self.df['Details']+' '+self.df['Type']
                self.df['Amount']=pd.to_numeric(self.df['Amount'])
                
                return self.df
        @classmethod
        def convert_credit(cls, df_input:pd.DataFrame) -> pd.DataFrame:
                df_input[['Description','Category','Type']]=df_input[['Description','Category','Type']].replace(np.nan,"")
                return df_input['Description'] +df_input['Category']+df_input['Type']
        def convert_debit(cls, df_input:pd.DataFrame) -> pd.DataFrame:
                df_input[['Description','Details','Type']]=df_input[['Description','Details','Type']].replace(np.nan,"")
                return df_input['Description'] +df_input['Details']+df_input['Type']
        

#%%                        
### Add in one vectorizer for all
def create_vectorizer(df_input, **kwargs: pd.DataFrame) -> TfidfVectorizer:
        pass

        
        
        

#%%
def save_data_and_models():
    X_train_prop, y_train_prop, X_val_prop, y_val_prop, vectorizer_prop = df_property_label()
    X_train_label, y_train_label, X_val_label, y_val_label, vectorizer_label = df_expense_label()
    dir_output = os.path.join(data_path, 'train/intermediate')
    os.makedirs(dir_output, exist_ok=True)  # Ensure the directory exists
    os.chdir(dir_output)
    X_train_prop.to_parquet('X_train_prop.parquet')
    X_val_prop.to_parquet('X_val_prop.parquet')
    y_train_prop.to_parquet('y_train_prop.parquet')
    y_val_prop.to_parquet('y_val_prop.parquet')
    X_train_label.to_parquet('X_train_label.parquet')
    X_val_label.to_parquet('X_val_label.parquet')
    y_train_label.to_parquet('y_train_label.parquet')
    y_val_label.to_parquet('y_val_label.parquet')
    pickle.dump(vectorizer_prop, open("vectorizer_prop.pickle", "wb"))
    pickle.dump(vectorizer_label, open("vectorizer_label.pickle", "wb"))

if __name__=='__main__':
        save_data_and_models()

# %%
###Output the datasets to the next step


# %%

#%%
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pyarrow as pa
from enum import Enum





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
class chase_debit_read:
        def __init__(self, file_type:FileType):
               self.type=file_type
               
        def read(self,file_input):
                if self.type==FileType.csv:
                        self.df=pd.read_csv(file_input,index_col=False)
                elif self.type==FileType.xlsx:
                        self.df=pd.read_excel(file_input)
                self.df['Amount']=pd.to_numeric(self.df['Amount'])
                self.df['Descriptions_all']=self.df['Description']+' '+self.df['Details']+' '+self.df['Type']
                return self.df

#%%                        
df_file=chase_debit_read(FileType.csv)         
df=df_file.read('2023_debit.CSV')     
               


#%%
# Assume the 'text' column has the text data and 'label' column has the property labels
##
def df_property_label():
        X = havu_credit2021[['Descriptions_all','Amount']]
        y = havu_credit2021['Property']

        # Split the data into training and test sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11)

        # Fit the vectorizer on the training set
        vectorizer.fit(X_train['Descriptions_all'])

        # Transform both the test set and the prediction set using the same vectorizer
        X_train_vec = vectorizer.transform(X_train['Descriptions_all'])
        X_val_vec = vectorizer.transform(X_val['Descriptions_all'])

        # Ensure the 'Amount' column is of numeric data type
        X_train['Amount'] = X_train['Amount'].astype('float64')
        X_val['Amount'] = X_val['Amount'].astype('float64')

        # Stack the 'Amount' column with the text features
        X_train = hstack([X_train_vec,X_train['Amount'].values.reshape(-1, 1)])
        X_val = hstack([X_val_vec,X_val['Amount'].values.reshape(-1, 1)])
        
        havu_credit2022['Category'].fillna("Unknown", inplace=True)

        # Create a new DataFrame 'df' with 'Descriptions_all' and 'Amount' columns and drop NaN values
        df = havu_credit2022[['Descriptions_all', 'Amount']]
        df.dropna(inplace=True)

        X_pred_vec = vectorizer.transform(df['Descriptions_all'])

        # Use 'Amount' column from 'df' after dropping NaN values
        X_pred = hstack([X_pred_vec, df['Amount'].values.reshape(-1, 1)])
        csr_matrix = X_train.tocsr()
        # Now convert to DataFrame
        df_train = pd.DataFrame(X_train.todense()) ##sparse matrix.  Needs to convert to dense matrix to dataframe it
        df_val = pd.DataFrame(X_val.todense())
        df_test=pd.DataFrame(X_pred.todense())
        return df_train,pd.DataFrame(y_train), df_val, pd.DataFrame(y_val),vectorizer()

#%%

# %%


def df_expense_label():
        havu_debit2021['Property'][havu_debit2021['Property']=='laramie']='Laramie'
        havu_debit2021['Property'][havu_debit2021['Property']=='Anthony']='Laramie'
        havu_debit2021['Amount']=pd.to_numeric(havu_debit2021['Amount'])

        # Load your CSV file
        havu_debit2021['Descriptions_all']=havu_debit2021['Description']+' '+havu_debit2021['Details']+' '+havu_debit2021['Type']
        havu_debit2022['Descriptions_all']=havu_debit2022['Description']+' '+havu_debit2022['Details']+' '+havu_debit2022['Type']

        df = havu_credit2022[['Descriptions_all', 'Amount']]
        df.dropna(inplace=True)

        X = havu_debit2021['Descriptions_all']
        y = havu_debit2021['Label']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        X_pred_vec = vectorizer.transform(df['Descriptions_all'])
        X_pred = hstack([X_pred_vec, df['Amount'].values.reshape(-1, 1)])
        df_train=pd.DataFrame(X_train_vec.todense())
        df_test=pd.DataFrame(X_test_vec.todense())
        
        return df_train , pd.DataFrame(y_train), df_test, pd.DataFrame(y_test)
#%%
if __name__== '__main__':
        X_train_prop, y_train_prop, X_val_prop, y_val_prop=df_property_label()
        X_train_label,y_train_label,X_val_label,y_val_label=df_expense_label()
        dir_output=os.path.join(data_path,'train/intermediate')
        os.chdir(dir_output)
        X_train_prop.to_parquet('X_train_prop.parquet')
        X_val_prop.to_parquet('X_val_prop.parquet')
        y_train_prop.to_parquet('y_train_prop.parquet')
        y_val_prop.to_parquet('y_val_prop.parquet')
        X_train_label.to_parquet('X_train_label.parquet')
        X_val_label.to_parquet('X_val_label.parquet')
        y_train_label.to_parquet('y_train_label.parquet')
        y_val_label.to_parquet('y_val_label.parquet')



# %%
###Output the datasets to the next step


# %%

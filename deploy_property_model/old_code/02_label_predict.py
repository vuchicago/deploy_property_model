# Load your CSV file
#%%
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

#%%

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

#%%
dir=os.path.join(os.getcwd(),'data')
os.chdir(dir)

havu_debit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Debit')

havu_credit2021=pd.read_excel('HaVu Taxes 2021.xlsx',sheet_name='Credit')

havu_debit2022=pd.read_csv('2022_debit.csv')
havu_credit2022=pd.read_csv('2022 Havu CC.CSV')

print(havu_debit2021['Posting Date'].min(),havu_debit2021['Posting Date'].max())


havu_credit2021['Category'].fillna("Unknown", inplace=True)
havu_credit2021['Descriptions_all']=havu_credit2021['Category']+havu_credit2021['Description']+havu_credit2021['Type']
havu_credit2022['Descriptions_all'] = havu_credit2022['Category'] + havu_credit2022['Description'] + havu_credit2022['Type']

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
#%%
# Assume the 'text' column has the text data and 'label' column has the property labels
X = havu_credit2021[['Descriptions_all','Amount']]
y = havu_credit2021['Property']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Fit the vectorizer on the training set
vectorizer.fit(X_train['Descriptions_all'])

# Transform both the test set and the prediction set using the same vectorizer
X_train_vec = vectorizer.transform(X_train['Descriptions_all'])
X_test_vec = vectorizer.transform(X_test['Descriptions_all'])

# Ensure the 'Amount' column is of numeric data type
X_train['Amount'] = X_train['Amount'].astype('float64')
X_test['Amount'] = X_test['Amount'].astype('float64')

# Stack the 'Amount' column with the text features
X_train = hstack([X_train_vec,X_train['Amount'].values.reshape(-1, 1)])
X_test = hstack([X_test_vec,X_test['Amount'].values.reshape(-1, 1)])

# Train the Logistic Regression model
model_property_credit = LogisticRegression()
model_property_credit.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_property_credit.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on 2022 set
havu_credit2022['Category'].fillna("Unknown", inplace=True)

# Create a new DataFrame 'df' with 'Descriptions_all' and 'Amount' columns and drop NaN values
df = havu_credit2022[['Descriptions_all', 'Amount']]
df.dropna(inplace=True)

X_pred_vec = vectorizer.transform(df['Descriptions_all'])

# Use 'Amount' column from 'df' after dropping NaN values
X_pred = hstack([X_pred_vec, df['Amount'].values.reshape(-1, 1)])

y_pred_property = model_property_credit.predict(X_pred)
property_pred = pd.Series(y_pred_property, name='Property')
property_pred.sample(10)

# %%

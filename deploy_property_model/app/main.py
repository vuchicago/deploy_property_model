#%%
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from property_predict import TextInput, PropertyPredict
#%%
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from property_predict import prediction_prop_debit,prediction_prop_credit

#%%


app = FastAPI()

class Description(BaseModel):
    Description: str


@app.get("/")
async def root():
    return {"message": "main page"}

@app.post("/predict_property_credit")
async def predict_property_cc(description: Description):
    result,output_prob = prediction_prop_credit(description.Description)
    return {"property": str(result), "proba":str(output_prob.max())}

@app.post("/predict_property_debit")
async def predict_property_debit(description: Description):
    result,output_prob = prediction_prop_debit(description.Description)
    return {"property": str(result), "proba":str(output_prob.max())}

# %%

predict_property_cc("hello keon")
# %%

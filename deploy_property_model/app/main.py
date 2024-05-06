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
    return {"message": "HaVu Investment Group Property Predictions Page"}

@app.post("/predict_property_credit")
async def predict_property_cc(description: Description):
    result,output_prob = prediction_prop_credit(description.Description)
    return {"property": str(result), "proba":str(output_prob.max())}

@app.post("/predict_property_debit")
async def predict_property_debit(description: Description):
    result,output_prob = prediction_prop_debit(description.Description)
    return {"property": str(result), "proba":str(output_prob.max())}

# %%

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
# %%

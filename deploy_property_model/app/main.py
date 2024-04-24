#%%
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from property_predict import TextInput, PropertyPredict,prediction


#%%


app = FastAPI()

class Description(BaseModel):
    Description: str


@app.get("/")
async def root():
    return {"message": "main page"}
@app.get("/hello_world")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_property")
async def predict_property_cc(description: Description):
    result = prediction(description.Description)
    return {"message": result}




#%%
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from property_predict import TextInput, PropertyPredict
from typing import Annotated, List
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import SessionLocal, engine
import models 
from fastapi.middleware.cors import CORSMiddleware
from database import engine
#%%
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from property_predict import prediction_prop_debit,prediction_prop_credit

#%%



app = FastAPI()

class Description(BaseModel):
    description: str  # Lowercase field name

class TransactionModel(BaseModel):
    id: int
    description: str
    prediction: str
    proba: float

    class Config:
        orm_mode = True  # Adjusted for Pydantic v2 if needed

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Depends(get_db)

models.Base.metadata.create_all(bind=engine)  # Ensure this is called appropriately, usually outside an endpoint

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_property_credit", response_model=TransactionModel)
async def predict_property_cc(description: Description, db: Session = db_dependency):
    # Example function to predict and return a result
    result, output_prob = prediction_prop_credit(description.description)
    
    # Create a new transaction record
    db_property = models.TransactionProperty(
        Description=description.description,
        Prediction=result,
        Proba=output_prob
    )
    db.add(db_property)
    db.commit()
    db.refresh(db_property)
    
    return {"property": str(result), "proba": str(output_prob.max()), "transaction": db_property}



@app.post("/predict_property_debit")
async def predict_property_debit(description: Description):
    result,output_prob = prediction_prop_debit(description.Description)
    return {"property": str(result), "proba":str(output_prob.max())}

# %%

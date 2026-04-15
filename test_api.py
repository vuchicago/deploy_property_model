##test api on port 8000
#%%
import requests
from fastapi.testclient import TestClient
from main import app
import pandas as pd
import os
from property_predict import prediction_prop_debit, prediction_prop_credit

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

client = TestClient(app)

def test_prediction_debit():
    response = client.post("/predict/debit", json={"data": "your_test_data_here"})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_prediction_credit():
    response = client.post("/predict/credit", json={"data": "hello keon"})
    assert response.status_code == 200
    assert "prediction" in response.json()

if __name__ == "__main__":
    test_prediction_debit()
    test_prediction_credit()
    print("All tests passed!")




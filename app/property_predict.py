#%%
import os
import pickle
from pydantic import BaseModel
from typing import Any
os.chdir(r'/Users/vuchicago/Python/deploy_property_model/model_pipeline/data/train')

#%%
with open("vectorizer_prop.pickle", "rb") as f:
    vectorizer_prop = pickle.load(f)

with open("property_pred_debit.pickle", "rb") as f:
    model_property_debit = pickle.load(f)

with open("property_pred_credit.pickle", "rb") as f:
    model_property_credit = pickle.load(f)

#%%

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

def prediction(Description):
        predictor = PropertyPredict(vectorizer=vectorizer_prop, model=model_property_credit)
        input_data = TextInput(text_input=Description)
        result = predictor.predict(input_data)
        return str(result)
        


        
        


# %%

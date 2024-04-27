#%%
import os
import pickle
from pydantic import BaseModel
from typing import Any
os.chdir(r'/Users/vuchicago/Python/deploy_property_model/model_pipeline/data/train')

#%%
with open("vectorizer_prop_credit.pickle", "rb") as f:
    vectorizer_prop_credit = pickle.load(f)
with open("vectorizer_prop_debit.pickle", "rb") as f:
    vectorizer_prop_debit = pickle.load(f)

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
                output_prob=max(self.model.predict_proba(input))
                return output,output_prob



def prediction_prop_debit(Description):
        predictor = PropertyPredict(vectorizer=vectorizer_prop_debit, model=model_property_debit)
        input_data = TextInput(text_input=Description)
        result,output_prob = predictor.predict(input_data)
        return result, output_prob

def prediction_prop_credit(Description):
        predictor = PropertyPredict(vectorizer=vectorizer_prop_credit, model=model_property_credit)
        input_data = TextInput(text_input=Description)
        result,output_prob = predictor.predict(input_data)
        return result,output_prob




        
        


# %%
prediction_prop_credit('Laramie')
# %%

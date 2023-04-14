import os
import pickle
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from azure.storage.blob import BlobClient
from sklearn.preprocessing import RobustScaler
from azure.identity import DefaultAzureCredential
from sklearn.ensemble import RandomForestClassifier

router = APIRouter()
token_credential = DefaultAzureCredential()

#Get Environment Variables
STORAGE_ACCOUNT_URL=os.getenv("STORAGE_ACCOUNT_URL")
STORAGE_ACCOUNT_CONTAINER_NAME=os.getenv("STORAGE_ACCOUNT_CONTAINER_NAME")

class StrokeParameters(BaseModel):

    have_heart_disease: str = Field(title="Heart Disease?",
    desceription="Does patient have disease?", default="yes")

    ever_smoked: str = Field(default="formerly smoked", title="smoker",
    description="Does patitients smoke")

    have_hypertention: str = Field(default="yes", title="Hypertention",
    description="Does patients have hypertention")

    glucose: float = Field(default=78.90, title="Average glucose level",
    description="Patient glucose level")

    bmi: float = Field(default=56.10, title="Body maths Index",
    description="Patient body maths index")

@router.post("/stroke")
async def predictStroke(stroke: StrokeParameters = Body(...)):

    if stroke.ever_smoked == "formerly smoked" or stroke.ever_smoked == "never smoked" or stroke.ever_smoked == "smokes":

        if stroke.have_hypertention == "yes" or stroke.have_hypertention == "no":

            if stroke.have_heart_disease == "yes" or stroke.have_heart_disease == "no":
                
                if stroke.ever_smoked == "formerly smoked":
                    formaly_smoke = 1.0
                    never_smoke = 0.0
                    unknown = 0.0
                    smokes = 0.0

                elif stroke.ever_smoked == "never smoked":
                    formaly_smoke = 0.0
                    never_smoke = 1.0
                    unknown = 0.0
                    smokes = 0.0
            
                elif stroke.ever_smoked == "smokes":
                    formaly_smoke = 0.0
                    never_smoke = 0.0
                    unknown = 0.0
                    smokes = 1.0
            
                else:
                    formaly_smoke = 0.0
                    never_smoke = 0.0
                    unknown = 1.0
                    smokes = 0.0

                if stroke.have_hypertention == "yes":
                    have_hypertention = 1
                else:
                    have_hypertention = 0


                if stroke.have_heart_disease == "yes":
                    have_heart_disease = 1.0
                else:
                    have_heart_disease = 0.0

                # Scale Bmi and Glucose Level
                robust_scaler = RobustScaler()

                # Fit and transform on dataset
                bmi_glucose = robust_scaler.fit_transform([[stroke.bmi, stroke.glucose]])

                bmi = bmi_glucose[0][0]
                glucose = bmi_glucose[0][1]

                input = [
                    unknown, 
                    formaly_smoke, 
                    never_smoke, 
                    smokes, 
                    glucose, 
                    have_hypertention, 
                    have_heart_disease, 
                    bmi
                ]

                blob_client = BlobClient(STORAGE_ACCOUNT_URL,container_name=STORAGE_ACCOUNT_CONTAINER_NAME, blob_name="model.pkl", credential=token_credential)

                with open("./model.pkl", "wb") as my_blob:
                    blob_data = blob_client.download_blob()
                    blob_data.readinto(my_blob)

                saved_model = pickle.load(open("model.pkl","rb"))

                predicted = saved_model.predict([input])
                os.remove("model.pkl")
    
                if predicted[0] == 1:
                    return {"Predicted": "Stroke Patient"}

                else:
                     return {"Predicted": "Not A Stroke Patient"}
            else:
                
                return {"allowed values for heart disease are": ["yes", "no"]}
        else:
            return {"allowed values for have_hypertention are": ["yes", "no"]}
    else:
        return {"allowed values for smoke are": ["formerly smoked", "never smoked", "smokes"]}
    
    
    
    

        
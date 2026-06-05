# main.py
import os

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from .Dataset import symptoms, predict_specialists, specialists, SPECIALIST_TABLETS

# 2️⃣ Initialize FastAPI
app = FastAPI()

# 3️⃣ Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3️⃣ Load model at startup
model = joblib.load(os.path.join(BASE_DIR, "Model", "specialist_rf_model.pkl"))
print("Model loaded successfully!")

class SymptomInput(BaseModel):
    symptoms: list[str]


# 4️⃣ Define prediction endpoint
@app.post("/predict")
def predict(data: SymptomInput):
    predicted_specialists, confidence_scores = predict_specialists(
        input_symptoms=data.symptoms,
        model=model,
        feature_names=symptoms,
        specialist_names=specialists,
        threshold=0.2
    )
    
    recommended_tablets = []
    for spec in predicted_specialists:
        recommended_tablets.extend(SPECIALIST_TABLETS.get(spec, []))
    recommended_tablets = list(set(recommended_tablets))


    return {"prediction": predicted_specialists, "scores": confidence_scores,"tablets": recommended_tablets}


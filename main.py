from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Intrusion Detection System API")

# Enable CORS so your frontend can make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Load models and scaler
logistic_model = joblib.load(BASE_DIR / "models" / "logistic_intrusion_model.pkl")
decision_tree_model = joblib.load(BASE_DIR / "models" / "decision_tree_intrusion_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict/logistic")
def predict_logistic(features: list):
    try:
        data = scaler.transform([features])
        prediction = logistic_model.predict(data)
        return {
            "model": "logistic_regression",
            "attack_detected": int(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/tree")
def predict_tree(features: list):
    try:
        data = scaler.transform([features])
        prediction = decision_tree_model.predict(data)
        return {
            "model": "decision_tree",
            "attack_detected": int(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}

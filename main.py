from fastapi import FastAPI
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Intrusion Detection System API")

BASE_DIR = Path(__file__).resolve().parent

# Load models and scaler (CORRECT PATHS)
logistic_model = joblib.load(BASE_DIR / "models" / "logistic_intrusion_model.pkl")
decision_tree_model = joblib.load(BASE_DIR / "models" / "decision_tree_intrusion_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict/logistic")
def predict_logistic(features: list):
    data = scaler.transform([features])
    prediction = logistic_model.predict(data)
    return {
        "model": "logistic_regression",
        "attack_detected": int(prediction[0])
    }

@app.post("/predict/tree")
def predict_tree(features: list):
    data = scaler.transform([features])
    prediction = decision_tree_model.predict(data)
    return {
        "model": "decision_tree",
        "attack_detected": int(prediction[0])
    }

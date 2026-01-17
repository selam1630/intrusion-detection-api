from fastapi import FastAPI, Body
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Intrusion Detection System API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
logistic_model = joblib.load(BASE_DIR / "models" / "logistic_intrusion_model.pkl")
decision_tree_model = joblib.load(BASE_DIR / "models" / "decision_tree_intrusion_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict/logistic")
def predict_logistic(features: List[float] = Body(...)):
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
def predict_tree(features: List[float] = Body(...)):
    try:
        data = scaler.transform([features])
        # Obtain predictions from both models and handle failures gracefully
        tree_pred = None
        logistic_pred = None
        try:
            tree_pred = int(decision_tree_model.predict(data)[0])
        except Exception:
            tree_pred = None

        try:
            logistic_pred = int(logistic_model.predict(data)[0])
        except Exception:
            logistic_pred = None

        if tree_pred is None and logistic_pred is None:
            return {"error": "Both models failed to produce a prediction."}

        # Prefer logistic prediction when both disagree (temporary fallback)
        if logistic_pred is not None and tree_pred is not None:
            final_pred = logistic_pred if logistic_pred != tree_pred else tree_pred
        else:
            final_pred = logistic_pred if logistic_pred is not None else tree_pred

        return {
            "model": "decision_tree",
            "tree_prediction": tree_pred,
            "logistic_prediction": logistic_pred,
            "attack_detected": int(final_pred),
            "note": "If tree and logistic disagree, logistic prediction is preferred as a temporary fallback."
        }
    except Exception as e:
        return {"error": str(e)}

# app/main.py
from fastapi import FastAPI
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

app = FastAPI()
VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}

@app.get("/predict")
def predict():
    if VERSION == "v1.1.0":
        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_true = y
        acc = accuracy_score(y_pred, y_true)
        return {"prediction": f'Prediction success, accuracy = {acc}', "version": VERSION}
    return {"prediction": "ok", "version": VERSION}
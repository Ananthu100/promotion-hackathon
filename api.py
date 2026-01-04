from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("model/model.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return {"promotion_probability": float(prob)}

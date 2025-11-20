from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

scaler = joblib.load("scaler.pkl")
model = joblib.load("XGB_model_final.pkl")

def predict(row_dict):
    x = pd.DataFrame([row_dict])
    
    x_scaled = scaler.transform(x)

    prob = model.predict_proba(x_scaled)[0][1]
    pred = int(prob >= 0.5)

    return pred, prob

@app.post("/predict")
async def endpoint(form_data: dict):
    print(form_data)

    x = predict(form_data)
    print(x)

    return {"prediction": x[0], "probability": round(x[1].item()*100, 2), "cluster": None}


@app.post("/cluster")
async def cluster(form_data: dict):
    print(form_data)
    # Here you would add your clustering logic using the parameters
    # For demonstration, we will just return the received parameters
    return {"received_params": form_data}
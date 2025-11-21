from fastapi import FastAPI
import pandas as pd
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scaler = joblib.load("scaler.pkl")
booster = xgb.Booster()
booster.load_model("XGB_model_final.json")

clusterScaler = joblib.load("scaler(clus).pkl")
clusterModel = joblib.load("kmeans_model.pkl")
pca = joblib.load("pca.pkl")




def predictCluster(row_dict):
    x = pd.DataFrame([row_dict])

    # Scale
    x_scaled = clusterScaler.transform(x)

    x_scaled_PCA = pca.transform(x_scaled)

    pred = clusterModel.predict(x_scaled_PCA)
    
    return pred



def predict(row_dict):
    x = pd.DataFrame([row_dict])
    x_scaled = scaler.transform(x)

    dmatrix = xgb.DMatrix(x_scaled)
    prob = float(booster.predict(dmatrix)[0])
    pred = int(prob >= 0.5)

    return pred, prob




@app.post("/predict")
async def endpoint(form_data: dict):
    print(form_data)

    attrition = predict(form_data)

    clusterData = form_data.copy()
    clusterData.pop("Department_Human Resources")
    clusterData.pop("Department_Research & Development")
    clusterData.pop("Department_Sales")
    cluster = predictCluster(clusterData)

    return {"prediction": attrition[0], "probability": round(attrition[1]*100, 2), "cluster": int(cluster)}

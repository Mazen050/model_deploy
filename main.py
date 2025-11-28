from fastapi import FastAPI
import shap
import numpy as np
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

explainer = shap.TreeExplainer(booster)

clusterScaler = joblib.load("scaler(clus).pkl")
clusterModel = joblib.load("kmeans_model.pkl")
pca = joblib.load("pca.pkl")



def predictCluster(row_dict):
    x = pd.DataFrame([row_dict])

    x_scaled = clusterScaler.transform(x)

    x_scaled_PCA = pca.transform(x_scaled)

    pred = clusterModel.predict(x_scaled_PCA)
    
    return pred


def get_top3_features(employee_data: pd.DataFrame,attrition: int):
    # Compute SHAP values
    shap_vals = explainer.shap_values(employee_data)[0]

    if attrition == 1:
        df = pd.DataFrame({
            "feature": employee_data.columns,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals)
        }).sort_values(by="shap_value", ascending=False)
    else:
        df = pd.DataFrame({
            "feature": employee_data.columns,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals)
        }).sort_values(by="shap_value", ascending=True)


    # Pick top 3
    top3 = df.head(3)

    # Convert into readable explanation
    readable_reasons = []
    for _, row in top3.iterrows():
        feature = row["feature"]
        value = employee_data.iloc[0][feature]
        impact = row["shap_value"]

        if impact > 0:
            direction = "increases"
        else:
            direction = "decreases"

        readable_reasons.append(
            f"{feature.replace('_',' ').title()} {direction} chance of leaving."
        )

    return {
        "top3_table":  top3.to_dict(orient="records"),
        "reasons": readable_reasons
    }




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

    topFeats = get_top3_features(pd.DataFrame([form_data]), attrition[0])

    cluster = None
    if attrition[0] == 1:
        clusterData = form_data.copy()
        clusterData.pop("Department_Human Resources")
        clusterData.pop("Department_Research & Development")
        clusterData.pop("Department_Sales")
        cluster = predictCluster(clusterData)
        cluster = int(cluster)

    return {
            "prediction": attrition[0],
            "probability": round(attrition[1]*100, 2),
            "cluster": cluster,
            "top_features": dict(topFeats)
            }

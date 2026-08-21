from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Cloud SDK: boto3 (AWS)
import boto3
import joblib
import os

app = FastAPI()

# Doc ten bucket tu bien moi truong (duoc dat trong systemd service)
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


def download_model():
    """Tai file model.pkl tu S3 ve may khi server khoi dong."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Downloaded s3://{S3_BUCKET}/{S3_MODEL_KEY} -> {MODEL_PATH}")


# Goi ham nay khi module duoc import (chay khi server khoi dong)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server. GitHub Actions dung endpoint nay de xac nhan deploy thanh cong."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.

    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    Dau ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    pred = int(model.predict([req.features])[0])

    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

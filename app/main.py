from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict a student's final grade using a trained ML model.",
    version="1.0.0",
)


class StudentData(BaseModel):
    school: str
    sex: str
    age: int = Field(ge=15, le=22)
    address: str
    famsize: str
    Pstatus: str
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0)
    G1: int = Field(ge=0, le=20)
    G2: int = Field(ge=0, le=20)


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(student: StudentData):
    try:
        input_df = pd.DataFrame([student.model_dump()])
        prediction = model.predict(input_df)[0]

        return {
            "predicted_final_grade": round(float(prediction), 2)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

from features import make_features

app = FastAPI()

# Load the trained model artifact that was created by train_model.py.
# It includes both the sklearn pipeline and the tuned threshold.
artifact = joblib.load("model.joblib")
model = artifact["model"]
THRESHOLD = artifact["threshold"]


# This represents one study in the request body.
# study_date is optional because the model only uses descriptions right now.
class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: Optional[str] = None


# This represents one case with a current study and a list of prior studies.
class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: Study
    prior_studies: List[Study]


# Full input schema for the /predict endpoint.
# Some fields are included only so the API accepts the challenge JSON format cleanly.
class RequestBody(BaseModel):
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    split: Optional[str] = None
    truth_count: Optional[int] = None
    truth: Optional[list] = None
    cases: List[Case]


@app.post("/predict")
def predict(payload: RequestBody):
    predictions = []

    rows = []
    metadata = []

    # Build one model input row for every prior study in every case.
    # Metadata is saved separately so we can attach predictions back to the right ids.
    for case in payload.cases:
        current = case.current_study

        for prior in case.prior_studies:
            rows.append(make_features(
                current.study_description,
                prior.study_description
            ))
            metadata.append((case.case_id, prior.study_id))

    X = pd.DataFrame(rows)

    # Convert model probabilities into true/false predictions using the tuned threshold.
    probs = model.predict_proba(X)[:, 1]
    model_preds = probs >= THRESHOLD

    # Format results in the exact output style expected by the challenge.
    for (case_id, study_id), pred in zip(metadata, model_preds):
        predictions.append({
            "case_id": case_id,
            "study_id": study_id,
            "predicted_is_relevant": bool(pred)
        })

    return {"predictions": predictions}


@app.get("/")
def root():
    return {"status": "ok"}
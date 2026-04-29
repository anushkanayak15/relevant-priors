from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re
import pandas as pd
import joblib

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


# Make the same features here that were used during training.
# This has to stay consistent with train_model.py or the model input will not match.
def make_model_features(current: Study, prior: Study) -> dict:
    cur = normalize(current.study_description)
    prior_desc = normalize(prior.study_description)

    cur_words = set(cur.split())
    prior_words = set(prior_desc.split())

    # Basic word overlap features help compare how similar the two study descriptions are.
    overlap = len(cur_words & prior_words)
    union = len(cur_words | prior_words)
    jaccard = overlap / union if union else 0

    cur_mod = get_modality(cur)
    prior_mod = get_modality(prior_desc)

    cur_region = " ".join(sorted(get_regions(cur))) or "unknown"
    prior_region = " ".join(sorted(get_regions(prior_desc))) or "unknown"

    return {
        # Combined pair text is used by the TF-IDF part of the trained pipeline.
        "text": f"current: {cur} prior: {prior_desc} pair: {cur} [SEP] {prior_desc}",
        "cur_text": cur,
        "prior_text": prior_desc,
        "cur_mod": cur_mod,
        "prior_mod": prior_mod,
        "cur_region": cur_region,
        "prior_region": prior_region,
        "same_modality": int(cur_mod == prior_mod),
        "same_region": int(bool(set(cur_region.split()) & set(prior_region.split()))),
        "exact_match": int(cur == prior_desc),
        "word_overlap": overlap,
        "jaccard": jaccard,
        "cur_len": len(cur_words),
        "prior_len": len(prior_words),
    }


# Simple normalization so capitalization and punctuation do not create fake differences.
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


# Pull out the likely imaging type from the description.
# This is rule-based, so it is rough, but it gives the model useful signal.
def get_modality(desc: str):
    desc = normalize(desc)
    if "mri" in desc or "mr " in desc:
        return "mri"
    if "ct" in desc:
        return "ct"
    if "xray" in desc or "x ray" in desc or "radiograph" in desc:
        return "xray"
    if "ultrasound" in desc or "us " in desc:
        return "us"
    if "mammo" in desc:
        return "mammo"
    if "pet" in desc:
        return "pet"
    return "other"


# Find likely body regions using keyword matching.
# A description can have more than one region, so this returns a set.
def get_regions(desc: str):
    desc = normalize(desc)
    regions = set()
    region_keywords = {

        "brain": [
            "brain", "head", "stroke", "intracranial"
        ],

        "chest": [
            "chest",
            "thorax",
            "lung",
            "pulmonary",
            "rib",
            "ribs",
            "cxr",
            "pa lat"
        ],

        "abdomen": [
            "abdomen",
            "abdominal",
            "liver",
            "pancreas",
            "kidney",
            "kidneys",
            "renal",
            "bladder"
        ],

        "pelvis": [
            "pelvis",
            "pelvic"
        ],

        "spine": [
            "spine",
            "cervical",
            "thoracic",
            "lumbar"
        ],

        "breast": [
            "breast",
            "mam",
            "mammo",
            "mammography",
            "tomo",
            "cad",
            "digital screener",
            "screen 3d",
            "us breast",
            "ultrasound bilat",
            "ultrasound lt diag target",
            "standard screening combo",
            "screening combo"
        ],

        "heart": [
            "cardiac",
            "heart",
            "coronary",
            "myo perf",
            "myocardial",
            "spect",
            "calc screening",
            "coronary calc",
            "echo",
            "echocardiogram",
            "transesophageal",
            "definity"
        ],

        "extremity": [
            "knee",
            "shoulder",
            "hip",
            "ankle",
            "foot",
            "hand",
            "wrist",
            "elbow",
            "tibia",
            "fibula",
            "finger",
            "venous",
            "doppler",
            "leg",
            "legs",
            "femur"
        ],

        "neck": [
            "neck",
            "carotid",
            "soft tissue neck"
        ]
    }

    for region, keywords in region_keywords.items():
        if any(k in desc for k in keywords):
            regions.add(region)

    return regions


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
            rows.append(make_model_features(current, prior))
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

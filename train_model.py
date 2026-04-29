import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Basic cleanup so descriptions are easier to compare.
# I am keeping only lowercase letters, numbers, and spaces so punctuation does not affect matching.
def normalize(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


# Roughly identify the imaging modality from keywords in the study description.
# This is not perfect, but it gives the model an extra helpful feature beyond raw text.
def modality(desc):
    if "mri" in desc or "mr " in desc:
        return "mri"
    if "ct" in desc:
        return "ct"
    if "xr" in desc or "xray" in desc or "x ray" in desc:
        return "xray"
    if "ultrasound" in desc or "us " in desc:
        return "us"
    if "mam" in desc or "breast" in desc:
        return "mammo"
    if "pet" in desc:
        return "pet"
    return "other"


# Roughly identify body region from common radiology words.
# A study can match more than one region, so I join all the regions that are found.
def region(desc):
    groups = {
        "brain": ["brain", "head", "stroke", "intracranial"],
        "chest": ["chest", "lung", "thorax", "pulmonary", "rib", "ribs"],
        "abdomen": ["abdomen", "abdominal", "liver", "kidney", "renal", "bladder"],
        "pelvis": ["pelvis", "pelvic"],
        "spine": ["spine", "cervical", "thoracic", "lumbar"],
        "breast": ["breast", "mam", "mammo", "mammography", "tomo"],
        "heart": ["heart", "cardiac", "coronary", "echo", "myo", "spect"],
        "extremity": ["knee", "ankle", "foot", "hand", "wrist", "leg", "femur"],
        "neck": ["neck", "carotid"],
    }

    found = []
    for r, words in groups.items():
        if any(w in desc for w in words):
            found.append(r)

    return " ".join(found) if found else "unknown"


# Build one row of features for a current/prior study pair.
# The model gets both text-based features and simple manual comparison features.
def make_features(current_desc, prior_desc):
    cur = normalize(current_desc)
    prior = normalize(prior_desc)

    cur_words = set(cur.split())
    prior_words = set(prior.split())

    # Jaccard overlap is a simple way to measure how many words the two descriptions share.
    overlap = len(cur_words & prior_words)
    union = len(cur_words | prior_words)
    jaccard = overlap / union if union else 0

    cur_mod = modality(cur)
    prior_mod = modality(prior)

    cur_region = region(cur)
    prior_region = region(prior)

    return {
        # This combined text lets TF-IDF learn words/phrases across the pair together.
        "text": f"current: {cur} prior: {prior} pair: {cur} [SEP] {prior}",
        "cur_text": cur,
        "prior_text": prior,
        "cur_mod": cur_mod,
        "prior_mod": prior_mod,
        "cur_region": cur_region,
        "prior_region": prior_region,
        "same_modality": int(cur_mod == prior_mod),
        "same_region": int(bool(set(cur_region.split()) & set(prior_region.split()))),
        "exact_match": int(cur == prior),
        "word_overlap": overlap,
        "jaccard": jaccard,
        "cur_len": len(cur_words),
        "prior_len": len(prior_words),
    }


# Load the public training data.
with open("relevant_priors_public.json", "r") as f:
    data = json.load(f)

# Store labels by case_id and study_id so each prior study can be matched to its true answer.
labels = {
    (x["case_id"], x["study_id"]): x["is_relevant_to_current"]
    for x in data["truth"]
}

rows = []

# Convert each current/prior pair into one training row.
for case in data["cases"]:
    case_id = case["case_id"]
    current_desc = case["current_study"]["study_description"]

    for prior in case["prior_studies"]:
        study_id = prior["study_id"]
        key = (case_id, study_id)

        # Some priors may not have labels, so skip those.
        if key not in labels:
            continue

        feats = make_features(current_desc, prior["study_description"])
        feats["label"] = labels[key]
        rows.append(feats)

# Put all rows into a dataframe for sklearn.
df = pd.DataFrame(rows)

# These are the exact columns the model expects during training and prediction.
feature_cols = [
    "text",
    "cur_text",
    "prior_text",
    "cur_mod",
    "prior_mod",
    "cur_region",
    "prior_region",
    "same_modality",
    "same_region",
    "exact_match",
    "word_overlap",
    "jaccard",
    "cur_len",
    "prior_len",
]

X = df[feature_cols]
y = df["label"]

# Split into train and validation so we can tune the probability threshold.
# Stratify keeps the positive/negative label balance similar in both splits.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF handles the text features, and StandardScaler handles the numeric features.
# I used larger max_features because radiology descriptions can have useful phrase patterns.
preprocess = ColumnTransformer(
    transformers=[
        ("pair_text", TfidfVectorizer(ngram_range=(1, 5), min_df=2, max_features=250000), "text"),
        ("cur_text", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=50000), "cur_text"),
        ("prior_text", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=50000), "prior_text"),
        ("cur_mod", TfidfVectorizer(), "cur_mod"),
        ("prior_mod", TfidfVectorizer(), "prior_mod"),
        ("cur_region", TfidfVectorizer(), "cur_region"),
        ("prior_region", TfidfVectorizer(), "prior_region"),
        ("num", StandardScaler(), [
            "same_modality",
            "same_region",
            "exact_match",
            "word_overlap",
            "jaccard",
            "cur_len",
            "prior_len",
        ]),
    ]
)

# Logistic regression is simple but works well here because TF-IDF creates strong sparse features.
model = Pipeline([
    ("features", preprocess),
    ("clf", LogisticRegression(
        max_iter=7000,
        C=12.0,
        class_weight=None,
        solver="liblinear"
    ))
])

# Train the pipeline end-to-end.
model.fit(X_train, y_train)

# Get predicted probabilities on validation data, then choose the best cutoff.
probs = model.predict_proba(X_val)[:, 1]

best_acc = 0
best_threshold = 0.5

# Try different thresholds instead of assuming 0.5 is best.
for threshold in np.arange(0.20, 0.81, 0.01):
    preds = probs >= threshold
    acc = accuracy_score(y_val, preds)

    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print("Best validation accuracy:", best_acc)
print("Best threshold:", best_threshold)

# Save both the model and threshold so the API can use the same trained setup.
artifact = {
    "model": model,
    "threshold": float(best_threshold),
    "feature_cols": feature_cols,
}

joblib.dump(artifact, "model.joblib")
print("Saved improved model.joblib")

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import make_features, FEATURE_COLS
# Training hyperparameters are centralized in config.py
# so experiments can be changed without editing model code.
from config import (
    C,
    MAX_ITER,
    RANDOM_STATE,
    TEST_SIZE,
    PAIR_MAX_FEATURES,
    CUR_MAX_FEATURES,
    PRIOR_MAX_FEATURES
)

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
    patient_id = case.get("patient_id", case_id)
    current_desc = case["current_study"]["study_description"]

    for prior in case["prior_studies"]:
        study_id = prior["study_id"]
        key = (case_id, study_id)

        # Some priors may not have labels, so skip those.
        if key not in labels:
            continue

        feats = make_features(current_desc, prior["study_description"])
        feats["case_id"] = case_id
        feats["patient_id"] = patient_id
        feats["label"] = labels[key]

        rows.append(feats)

# Put all rows into a dataframe for sklearn.
df = pd.DataFrame(rows)

# These are the exact columns the model expects during training and prediction.
feature_cols = FEATURE_COLS

X = df[feature_cols]
y = df["label"]

# Split into train and validation by case_id.
# This is better than random pair-level splitting because it prevents priors from the same case
# from appearing in both training and validation.
groups = df["case_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(splitter.split(df, df["label"], groups=groups))

X_train = df.iloc[train_idx][feature_cols]
X_val = df.iloc[val_idx][feature_cols]
y_train = df.iloc[train_idx]["label"]
y_val = df.iloc[val_idx]["label"]

# TF-IDF handles the text features, and StandardScaler handles the numeric features.
# I used larger max_features because radiology descriptions can have useful phrase patterns.
preprocess = ColumnTransformer(
    transformers=[
        ("pair_text", TfidfVectorizer(ngram_range=(1, 5), min_df=2, max_features=PAIR_MAX_FEATURES), "text"),
        ("cur_text", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=CUR_MAX_FEATURES), "cur_text"),
        ("prior_text", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=PRIOR_MAX_FEATURES), "prior_text"),
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
        max_iter=MAX_ITER,
        C=C,
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
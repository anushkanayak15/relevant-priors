from features import make_features, FEATURE_COLS
from app import RequestBody
from fastapi.testclient import TestClient
from app import app


def test_make_features_has_all_expected_columns():
    features = make_features(
        "MRI BRAIN WITHOUT CONTRAST",
        "CT HEAD WITHOUT CONTRAST"
    )

    for col in FEATURE_COLS:
        assert col in features


def test_feature_values_are_reasonable():
    features = make_features(
        "MRI BRAIN WITHOUT CONTRAST",
        "CT HEAD WITHOUT CONTRAST"
    )

    assert features["cur_mod"] == "mri"
    assert features["prior_mod"] == "ct"
    assert features["same_region"] == 1
    assert features["exact_match"] == 0
    assert features["word_overlap"] > 0


def test_request_schema_accepts_challenge_format():
    sample = {
        "challenge_id": "relevant-priors-v1",
        "schema_version": 1,
        "generated_at": "2026-04-16T12:00:00.000Z",
        "cases": [
            {
                "case_id": "1001016",
                "patient_id": "606707",
                "patient_name": "Example, Patient",
                "current_study": {
                    "study_id": "3100042",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2026-03-08"
                },
                "prior_studies": [
                    {
                        "study_id": "2453245",
                        "study_description": "CT HEAD WITHOUT CONTRAST",
                        "study_date": "2021-03-08"
                    }
                ]
            }
        ]
    }

    parsed = RequestBody(**sample)

    assert len(parsed.cases) == 1
    assert parsed.cases[0].case_id == "1001016"
    assert parsed.cases[0].prior_studies[0].study_id == "2453245"




def test_predict_returns_one_prediction_per_prior():
    client = TestClient(app)

    sample = {
        "cases": [
            {
                "case_id": "case-1",
                "current_study": {
                    "study_id": "current-1",
                    "study_description": "MRI BRAIN WITHOUT CONTRAST"
                },
                "prior_studies": [
                    {
                        "study_id": "prior-1",
                        "study_description": "CT HEAD WITHOUT CONTRAST"
                    },
                    {
                        "study_id": "prior-2",
                        "study_description": "XR CHEST 2 VIEW"
                    }
                ]
            }
        ]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_output_schema():
    client = TestClient(app)

    sample = {
        "cases": [
            {
                "case_id": "case-1",
                "current_study": {
                    "study_id": "current-1",
                    "study_description": "CT CHEST WITH CONTRAST"
                },
                "prior_studies": [
                    {
                        "study_id": "prior-1",
                        "study_description": "CT CHEST WITHOUT CONTRAST"
                    }
                ]
            }
        ]
    }

    response = client.post("/predict", json=sample)
    pred = response.json()["predictions"][0]

    assert set(pred.keys()) == {
        "case_id",
        "study_id",
        "predicted_is_relevant"
    }

    assert pred["case_id"] == "case-1"
    assert pred["study_id"] == "prior-1"
    assert isinstance(pred["predicted_is_relevant"], bool)
import json
import requests

with open("relevant_priors_public.json", "r") as f:
    data = json.load(f)

labels = {}
for item in data["truth"]:
    labels[(item["case_id"], item["study_id"])] = item["is_relevant_to_current"]

case_lookup = {}
for case in data["cases"]:
    case_lookup[case["case_id"]] = {
        "current": case["current_study"]["study_description"],
        "priors": {
            prior["study_id"]: prior["study_description"]
            for prior in case["prior_studies"]
        }
    }

response = requests.post("http://localhost:8000/predict", json=data)

if response.status_code != 200:
    print("Status code:", response.status_code)
    print("Error response:", response.text)
    exit()

predictions = response.json()["predictions"]

correct = 0
predicted_keys = set()

for pred in predictions:
    key = (pred["case_id"], pred["study_id"])
    predicted_keys.add(key)

    if key in labels and pred["predicted_is_relevant"] == labels[key]:
        correct += 1

missing_predictions = len(set(labels.keys()) - predicted_keys)
total = len(labels)
accuracy = correct / total

print("\nSample mistakes:\n")

shown = 0
for pred in predictions:
    key = (pred["case_id"], pred["study_id"])

    if key in labels and pred["predicted_is_relevant"] != labels[key]:
        case_id, study_id = key

        print("Case:", key)
        print("Current:", case_lookup[case_id]["current"])
        print("Prior:", case_lookup[case_id]["priors"][study_id])
        print("Predicted:", pred["predicted_is_relevant"])
        print("Actual:", labels[key])
        print("-" * 60)

        shown += 1
        if shown == 20:
            break

print(f"Accuracy: {accuracy:.4f}")
print(f"Correct: {correct}")
print(f"Total Truth Labels: {total}")
print(f"Missing Predictions (incorrect): {missing_predictions}")
print(f"Predictions Returned: {len(predictions)}")
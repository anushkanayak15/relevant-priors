# Experiments

## Problem Framing

The task is to predict whether a prior radiology examination should be shown to a radiologist while reading a current examination. I treated this as a binary classification problem over current/prior study pairs.

Each prediction uses the current study description and one prior study description, then returns whether the prior is relevant.

---

## Experiment Results

| Approach | Features | Validation / Evaluation | Accuracy |
|---|---|---|---|
| Rule-based baseline | Modality and anatomy overlap heuristics | Public eval script | ~91% |
| Logistic Regression + TF-IDF | Current/prior description pair text | Validation split | ~95% |
| Final model | TF-IDF + modality + anatomy + lexical overlap + threshold tuning | Grouped validation by case | 94.77% |
| Final local public evaluation | Final trained model through API | Full public eval JSON | 96.40% |

Final model results:

- Best grouped validation accuracy: 94.77%
- Best threshold: 0.71
- Full public evaluation accuracy: 96.40%
- Correct predictions: 26,621 / 27,614
- Missing predictions: 0

---

## Final Model

The final approach uses a supervised logistic regression classifier trained on engineered features from each current/prior study pair.

The model uses:

- TF-IDF n-gram features over the combined current/prior text pair
- Separate TF-IDF features for current study description
- Separate TF-IDF features for prior study description
- modality features
- anatomical region features
- modality match
- region overlap
- exact description match
- word overlap
- Jaccard similarity
- description length features

I tuned the probability threshold on the validation split instead of assuming a default 0.5 cutoff. The best threshold was 0.71.

---

## What Worked

The largest improvement came from moving from hand-written rules to a supervised learning approach.

The rule-based model was helpful as a baseline, but it struggled with edge cases. The logistic regression model learned more flexible patterns from the public labeled data while staying fast enough for endpoint evaluation.

The most useful features were:

- current/prior TF-IDF pair text
- anatomy overlap
- modality match
- exact description match
- word overlap and Jaccard similarity

I also moved feature engineering into a shared `features.py` file so that training and inference use the same logic. This reduces training-serving drift.

---

## What Failed

The pure rule-based approach plateaued around 91% accuracy. It produced false positives when two exams shared broad anatomy but were not actually useful comparisons. It also produced false negatives for cross-modality comparisons that were clinically relevant.

Adding too many hand-crafted overrides sometimes hurt generalization. For example, broad rules around chest imaging, PET/CT, breast imaging, and vascular studies fixed some examples but created new errors elsewhere.

A pair-level random split also risked inflating validation accuracy because similar studies from the same case could appear in both training and validation. I improved this by using grouped validation by case.

---

## Error Analysis

Remaining mistakes usually fell into these categories:

### Ambiguous cross-modality comparisons

Some prior exams are only relevant depending on clinical context, such as:

- PET/CT vs CT chest
- echocardiography vs chest imaging
- CT abdomen/pelvis vs ultrasound abdomen

Study descriptions alone do not always contain enough information to decide these perfectly.

### Generic study descriptions

Some descriptions are very short or vague, such as:

- “Abdomen”
- “Venous”
- “Breast”
- “Pelvic”

These are difficult because the model has limited context beyond the study description.

### Identical descriptions with different labels

Some pairs had the same or nearly identical study descriptions but different relevance labels. This suggests that factors beyond text, such as date, clinical indication, or accession context, may influence relevance.

### Specialized radiology workflow cases

The model still struggles with some specialized categories, including:

- breast laterality
- vascular studies
- oncology staging studies
- extremity versus vascular ultrasound comparisons

---

## Workflow Considerations

For radiologist workflow support, false negatives are usually more harmful than false positives because hiding a useful prior can remove important comparison context.

However, too many false positives can clutter the reading workflow and slow interpretation. Because of this, the classification threshold should depend on the intended product goal:

- use a lower threshold if recall is more important
- use a higher threshold if reducing clutter is more important

In a real deployment, I would evaluate this with radiologists using:

- relevance review studies
- reading-time measurements
- radiologist satisfaction feedback
- comparison against current PACS prior-selection workflows

If used as an assistive recommendation tool, I would favor higher recall so likely useful priors are surfaced for radiologist judgment. If used as an automatic filtering system that hides priors, I would favor higher precision to avoid suppressing clinically useful comparisons.

---

## Next Improvements

With more time, I would explore:

- domain-specific radiology embeddings
- gradient boosting or ensemble models
- richer anatomy ontology mapping
- patient-level grouped validation
- calibration for different precision/recall operating points
- adding study date gaps as a feature
- clinician-in-the-loop review for ambiguous cases
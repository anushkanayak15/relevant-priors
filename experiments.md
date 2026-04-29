# Experiments

## Baseline
I first implemented a rule-based relevance classifier using modality and anatomical overlap heuristics. This achieved roughly 91% public-split accuracy.

## What Worked
The largest improvement came from moving from rules to a supervised machine learning approach.

I trained a logistic regression classifier using:

- TF-IDF features over current/prior study description pairs
- modality match features
- anatomy/region overlap features
- exact description match
- lexical overlap and Jaccard similarity
- threshold tuning on a validation split

Best validation result:

- Accuracy: 95.38%
- Optimal threshold: 0.46

## What Failed
Pure rule-based methods plateaued and produced many false positives and false negatives in edge cases such as cross-modality comparisons and highly domain-specific priors.

Adding too many hand-crafted overrides often hurt generalization.

## Next Improvements
With more time I would explore:

- domain-specific radiology embeddings
- gradient boosting or ensemble models
- ontology-based anatomy grouping
- stacking rule-based priors with learned classifiers
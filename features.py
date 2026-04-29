import re

# These are the exact feature columns expected by both training and inference.
# Keeping them in one place avoids feature drift.
FEATURE_COLS = [
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

# Basic text cleanup.
# Lowercase everything and remove punctuation so descriptions compare consistently.
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())

# Roughly infer imaging modality from keywords in the description.
# This gives the model structured signal beyond raw text.
def get_modality(desc: str) -> str:
    desc = normalize(desc)

    if "mri" in desc or "mr " in desc:
        return "mri"

    if "ct" in desc:
        return "ct"

    if "xr" in desc or "xray" in desc or "x ray" in desc or "radiograph" in desc:
        return "xray"

    if "ultrasound" in desc or " us " in f" {desc} ":
        return "us"

    if "mam" in desc or "breast" in desc:
        return "mammo"

    if "pet" in desc:
        return "pet"

    if "echo" in desc or "echocardiogram" in desc:
        return "echo"

    if "nm" in desc or "spect" in desc or "myo perf" in desc:
        return "nuclear"

    return "other"


# Identify likely body regions from common radiology words.
# A study can belong to multiple regions, so this returns a set.
def get_regions(desc: str) -> set:
    desc = normalize(desc)
    regions = set()

    region_keywords = {

        "brain": [
            "brain",
            "head",
            "stroke",
            "intracranial",
            "cranial"
        ],

        "chest": [
            "chest",
            "thorax",
            "lung",
            "pulmonary",
            "rib",
            "ribs",
            "cxr"
        ],

        "abdomen": [
            "abdomen",
            "abdominal",
            "liver",
            "pancreas",
            "kidney",
            "kidneys",
            "renal",
            "bladder",
            "gallbladder",
            "spleen",
            "bowel"
        ],

        "pelvis": [
            "pelvis",
            "pelvic",
            "uterus",
            "ovary",
            "prostate"
        ],

        "spine": [
            "spine",
            "cervical",
            "thoracic",
            "lumbar",
            "sacral"
        ],

        "breast": [
            "breast",
            "mam",
            "mammo",
            "mammography",
            "tomo",
            "cad"
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
            "transesophageal"
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
            "leg",
            "legs",
            "femur"
        ],

        "neck": [
            "neck",
            "carotid",
            "soft tissue neck",
            "thyroid"
        ],

        "face_sinus": [
            "orbit",
            "eye",
            "facial",
            "face",
            "sinus",
            "temporal",
            "iac"
        ],
    }

    # Add every region whose keywords appear in the description.
    for region, keywords in region_keywords.items():
        if any(keyword in desc for keyword in keywords):
            regions.add(region)

    return regions


# Build one full feature row for a current/prior study pair.
# This is used both during training and at inference.
def make_features(current_desc: str, prior_desc: str) -> dict:

    cur = normalize(current_desc)
    prior = normalize(prior_desc)

    cur_words = set(cur.split())
    prior_words = set(prior.split())

    # Lexical overlap features
    overlap = len(cur_words & prior_words)
    union = len(cur_words | prior_words)

    # Jaccard similarity gives a normalized overlap score.
    jaccard = overlap / union if union else 0.0

    cur_mod = get_modality(cur)
    prior_mod = get_modality(prior)

    cur_regions = get_regions(cur)
    prior_regions = get_regions(prior)

    # Convert region sets into strings for TF-IDF inputs.
    cur_region = " ".join(sorted(cur_regions)) or "unknown"
    prior_region = " ".join(sorted(prior_regions)) or "unknown"

    return {

        # Pair text lets TF-IDF learn relationships across both descriptions jointly.
        "text": f"current: {cur} prior: {prior} pair: {cur} [SEP] {prior}",

        # Separate descriptions are also fed independently.
        "cur_text": cur,
        "prior_text": prior,

        # Structured categorical features
        "cur_mod": cur_mod,
        "prior_mod": prior_mod,
        "cur_region": cur_region,
        "prior_region": prior_region,

        # Binary comparison features
        "same_modality": int(cur_mod == prior_mod),
        "same_region": int(bool(cur_regions & prior_regions)),
        "exact_match": int(cur == prior),

        # Numeric overlap features
        "word_overlap": overlap,
        "jaccard": jaccard,
        "cur_len": len(cur_words),
        "prior_len": len(prior_words),
    }
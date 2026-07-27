from __future__ import annotations

import re

DISPLAY_OVERRIDES = {
    "AGE_AT_STROKE": "Age at stroke",
    "acute_kidney_failure": "Acute kidney failure",
    "acute_respiratory_failure": "Acute respiratory failure",
    "chronic_kidney_disease": "Chronic kidney disease",
    "congestive_heart_failure": "Congestive heart failure",
    "coronary_artery_disease": "Coronary artery disease",
    "diabetes_mellitus": "Diabetes mellitus",
    "DISCHARGE_GROUP": "Discharge destination",
    "DISCHARGE_STATUS": "Discharge status",
    "enc_duration": "Encounter duration",
    "prolonged_los": "Prolonged length of stay",
    "systolic_blood_pressure": "Systolic blood pressure",
    "diastolic_blood_pressure": "Diastolic blood pressure",
}

ACRONYMS = {
    "adi": "ADI",
    "alt": "ALT",
    "ast": "AST",
    "bmi": "BMI",
    "bun": "BUN",
    "ckd": "CKD",
    "dbp": "DBP",
    "egfr": "eGFR",
    "hct": "Hct",
    "hgb": "Hgb",
    "ig": "IG",
    "rbc": "RBC",
    "ruca": "RUCA",
    "sbp": "SBP",
    "svi": "SVI",
    "wbc": "WBC",
}


def display_name(value: object) -> str:
    """Convert machine-oriented variable names to manuscript display labels."""
    text = str(value)
    if text in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[text]

    cleaned = re.sub(r"[_\-]+", " ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return text

    words = []
    for token in cleaned.split(" "):
        key = token.lower().strip("()")
        replacement = ACRONYMS.get(key)
        words.append(replacement if replacement is not None else token.lower())

    label = " ".join(words)
    return label[0].upper() + label[1:] if label else text

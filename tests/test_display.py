import pandas as pd

from tpcluster.display import display_name


def test_display_name_uses_overrides():
    assert display_name("AGE_AT_STROKE") == "Age at stroke"
    assert display_name("acute_kidney_failure") == "Acute kidney failure"
    assert display_name("enc_duration") == "Encounter duration"


def test_display_name_preserves_common_acronyms():
    assert display_name("wbc_count") == "WBC count"
    assert display_name("ruca_category") == "RUCA category"


def test_display_name_handles_generic_machine_names():
    values = pd.Series(["serum_albumin", "lymphocyte_percentage"])
    assert values.map(display_name).tolist() == [
        "Serum albumin",
        "Lymphocyte percentage",
    ]

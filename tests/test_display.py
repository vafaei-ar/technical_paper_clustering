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


def test_display_name_polishes_manuscript_context_labels():
    assert display_name("Pat pref language spoken") == "Preferred language"
    assert display_name("SVI overal rank22") == "Social Vulnerability Index overall rank"
    assert display_name("SVI ses rank22") == "SVI socioeconomic status rank"
    assert display_name("SVI race rank22") == "SVI racial and ethnic minority status rank"
    assert display_name("ADI nat rank22") == "Area Deprivation Index national rank"
    assert display_name("Pct urban") == "Percent urban"

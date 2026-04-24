"""UCI Adult dataset loader.

The dataset is expected locally in `data/adult.csv`. If it is not present, we
synthesise a compatible dataset on the fly so the platform still runs end-to-end
without network access. The synthetic data mimics the column structure of the
real UCI Adult dataset, so the training pipeline is identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def _synthesise_adult(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic UCI-Adult-like dataset for offline / demo use."""
    rng = np.random.default_rng(seed)
    workclass = rng.choice(
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"],
        n_rows,
        p=[0.7, 0.08, 0.04, 0.05, 0.08, 0.05],
    )
    education = rng.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "Assoc-voc", "11th"],
        n_rows,
        p=[0.32, 0.22, 0.2, 0.08, 0.03, 0.1, 0.05],
    )
    education_num_map = {
        "11th": 7, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
        "Bachelors": 13, "Masters": 14, "Doctorate": 16,
    }
    education_num = np.array([education_num_map[e] for e in education])
    age = rng.integers(18, 75, size=n_rows)
    hours = rng.integers(20, 70, size=n_rows)
    capital_gain = np.where(rng.random(n_rows) < 0.08, rng.integers(1000, 40000, n_rows), 0)
    capital_loss = np.where(rng.random(n_rows) < 0.04, rng.integers(500, 4000, n_rows), 0)
    sex = rng.choice(["Male", "Female"], n_rows, p=[0.67, 0.33])
    marital = rng.choice(
        ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
        n_rows, p=[0.47, 0.33, 0.13, 0.03, 0.04],
    )
    occupation = rng.choice(
        ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair",
         "Adm-clerical", "Machine-op-inspct", "Transport-moving", "Other-service"],
        n_rows,
    )
    relationship = rng.choice(
        ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"],
        n_rows,
    )
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        n_rows, p=[0.85, 0.1, 0.03, 0.01, 0.01],
    )
    native = rng.choice(["United-States", "Mexico", "Philippines", "Germany", "India", "Other"],
                       n_rows, p=[0.9, 0.03, 0.01, 0.01, 0.01, 0.04])
    fnlwgt = rng.integers(20_000, 500_000, size=n_rows)

    # Signal: older, better-educated, longer hours, capital gains, male, married => higher income
    score = (
        0.02 * (age - 30)
        + 0.25 * (education_num - 9)
        + 0.03 * (hours - 40)
        + 0.00004 * capital_gain
        - 0.00005 * capital_loss
        + 0.4 * (sex == "Male")
        + 0.6 * (marital == "Married-civ-spouse")
        + rng.normal(0, 1.0, n_rows)
    )
    prob_high = 1.0 / (1.0 + np.exp(-(score - score.mean())))
    income = np.where(rng.random(n_rows) < prob_high, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": fnlwgt, "education": education,
        "education_num": education_num, "marital_status": marital, "occupation": occupation,
        "relationship": relationship, "race": race, "sex": sex,
        "capital_gain": capital_gain, "capital_loss": capital_loss,
        "hours_per_week": hours, "native_country": native, "income": income,
    })
    return df[ADULT_COLUMNS]


def load_adult(path: str | Path, auto_synthesise: bool = True) -> pd.DataFrame:
    """Load the UCI Adult dataset from a CSV file, or synthesise if missing."""
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        # Strip whitespace in string cells (real UCI file has leading spaces)
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].astype(str).str.strip()
        # Normalise income column
        df["income"] = df["income"].replace({">50K.": ">50K", "<=50K.": "<=50K"})
        return df

    if not auto_synthesise:
        raise FileNotFoundError(f"Adult dataset not found at {path}")

    p.parent.mkdir(parents=True, exist_ok=True)
    df = _synthesise_adult()
    df.to_csv(p, index=False)
    return df


def split_xy(df: pd.DataFrame, target: str = "income") -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X (features) and y (binary target)."""
    y = (df[target] == ">50K").astype(int)
    x = df.drop(columns=[target])
    return x, y

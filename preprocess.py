# preprocess.py
import pandas as pd
import numpy as np
import re
import os
import nltk
nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize

def simple_text_features(text):
    if pd.isnull(text): return {"notes_len":0, "infection_kw":0, "pain_kw":0}
    tokens = word_tokenize(text.lower())
    notes_len = len(tokens)
    infection_kw = sum(1 for t in tokens if t in ["infection","culture","fever"])
    pain_kw = sum(1 for t in tokens if t in ["pain","dizziness","chest","breath"])
    return {"notes_len": notes_len, "infection_kw": infection_kw, "pain_kw": pain_kw}

def preprocess(in_path="data/synthetic_healthcare.csv", out_path="data/processed_healthcare.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(in_path)
    # Drop duplicates if any
    df = df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    # Fill missing numeric values with median
    num_cols = ["age","heart_rate","systolic_bp","diastolic_bp","glucose","bmi","previous_admissions","length_of_stay"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    # Create simple features from clinical_notes
    features = df["clinical_notes"].apply(simple_text_features).apply(pd.Series)
    df = pd.concat([df, features], axis=1)
    # A simple risk feature: combine age, prev admits, glucose
    df["risk_index"] = (df["age"]/100) + (df["previous_admissions"]*0.2) + (df["glucose"]/400)
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")

if __name__ == "__main__":
    preprocess()

# generate_data.py
import random
import pandas as pd
import numpy as np
from faker import Faker
import os

fake = Faker()

def random_notes(i):
    # simple synthetic clinical notes
    sentences = [
        "Patient complains of chest pain and shortness of breath.",
        "Follow-up after surgery, wound healing is satisfactory.",
        "High blood pressure observed, medication adjusted.",
        "Diabetic patient with fluctuating sugar levels.",
        "Patient reports dizziness and mild fever.",
        "No acute distress; continue current treatment.",
        "Reports improved mobility after physiotherapy.",
        "Possible infection; culture recommended."
    ]
    # add some variability
    return " ".join(random.choices(sentences, k=random.randint(1,3)))

def create_synthetic(n=2000, out_path="data/synthetic_healthcare.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for i in range(n):
        age = random.randint(1, 95)
        gender = random.choice(["M", "F"])
        heart_rate = int(np.clip(random.gauss(75, 12), 40, 140))
        systolic = int(np.clip(random.gauss(125, 20), 80, 220))
        diastolic = int(np.clip(random.gauss(78, 12), 40, 140))
        glucose = int(np.clip(random.gauss(110, 40), 40, 400))
        bmi = round(np.clip(random.gauss(26, 6), 12, 50), 1)
        num_prev_adm = random.choices([0,1,2,3], weights=[0.6,0.25,0.1,0.05])[0]
        length_of_stay = int(np.clip(random.gauss(4, 3), 1, 30))
        # readmission indicator: make it dependent on some factors
        risk_score = 0
        risk_score += 0.02 * (age - 50) if age > 50 else 0
        risk_score += 0.1 * (num_prev_adm)
        risk_score += 0.01 * (bmi - 25) if bmi > 25 else 0
        risk_score += 0.005 * (glucose - 100) if glucose > 120 else 0
        prob_readmit = min(max(0.05 + risk_score, 0.01), 0.9)
        readmitted = np.random.rand() < prob_readmit
        notes = random_notes(i)
        patient_id = f"P{100000 + i}"
        rows.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "glucose": glucose,
            "bmi": bmi,
            "previous_admissions": num_prev_adm,
            "length_of_stay": length_of_stay,
            "clinical_notes": notes,
            "readmitted_30d": int(readmitted)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to {out_path}")

if __name__ == "__main__":
    create_synthetic(n=2500)

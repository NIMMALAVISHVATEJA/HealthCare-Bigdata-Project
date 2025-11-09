# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train(in_path="data/processed_healthcare.csv", model_path="models/readmission_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = pd.read_csv(in_path)
    # features and target
    features = ["age","heart_rate","systolic_bp","diastolic_bp","glucose","bmi",
                "previous_admissions","length_of_stay","notes_len","infection_kw","pain_kw","risk_index"]
    X = df[features]
    y = df["readmitted_30d"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds, digits=3))
    print("ROC AUC:", round(roc_auc_score(y_test, probs), 3))
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fpdf import FPDF
from preprocess import simple_text_features

MODEL_PATH = "models/readmission_model.pkl"
DATA_PATH = "data/processed_healthcare.csv"

st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")

st.title("🏥 Healthcare Analytics — Big Data Project")
st.markdown("### Real-Time Dashboard with CSV Upload & PDF Report")

@st.cache_data
def load_data(path=DATA_PATH):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning("⚠️ Default data file not found. Please upload a CSV below.")
        return pd.DataFrame()

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except:
        st.error("❌ Model not found. Please run `train_model.py` first.")
        return None

data = load_data()
model = load_model()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Charts", "Predict", "Upload & Export"])

# ---------------- Overview ----------------
if page == "Overview":
    st.header("📋 Dataset Overview")
    if data.empty:
        st.warning("No data found. Please go to 'Upload & Export' and upload your CSV file.")
    else:
        st.dataframe(data.head(10))
        st.subheader("Basic Statistics")
        st.write(data.describe().T)

# ---------------- Charts ----------------
if page == "Charts" and not data.empty:
    st.header("📊 Visual Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(data, x="age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(data, y="length_of_stay", title="Length of Stay Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.histogram(data, x="risk_index", title="Risk Index Distribution")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.bar(
            data.groupby("previous_admissions")["readmitted_30d"].mean().reset_index(),
            x="previous_admissions",
            y="readmitted_30d",
            title="Readmission Rate by Previous Admissions",
        )
        st.plotly_chart(fig4, use_container_width=True)

# ---------------- Predict ----------------
if page == "Predict":
    st.header("🧠 Predict 30-Day Readmission")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)
    systolic = st.number_input("Systolic BP", min_value=80, max_value=220, value=120)
    diastolic = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=110)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    prev_adm = st.number_input("Previous Admissions", min_value=0, max_value=10, value=0)
    los = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=3)
    notes = st.text_area("Clinical Notes", "Patient complains of chest pain and dizziness.")

    text_feats = simple_text_features(notes)
    risk_index = (age / 100) + (prev_adm * 0.2) + (glucose / 400)

    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "heart_rate": heart_rate,
                "systolic_bp": systolic,
                "diastolic_bp": diastolic,
                "glucose": glucose,
                "bmi": bmi,
                "previous_admissions": prev_adm,
                "length_of_stay": los,
                "notes_len": text_feats["notes_len"],
                "infection_kw": text_feats["infection_kw"],
                "pain_kw": text_feats["pain_kw"],
                "risk_index": risk_index,
            }
        ]
    )

    if model and st.button("Predict Readmission"):
        prob = model.predict_proba(input_data)[:, 1][0]
        pred = "Yes" if prob >= 0.5 else "No"
        st.success(f"Prediction: {pred} (Probability: {prob:.2f})")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Healthcare Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Prediction Result: {pred}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}, Glucose: {glucose}, BMI: {bmi}", ln=True)
        pdf.cell(200, 10, txt=f"Clinical Notes: {notes}", ln=True)
        report_path = "data/prediction_report.pdf"
        pdf.output(report_path)
        st.download_button("📄 Download PDF Report", open(report_path, "rb"), file_name="prediction_report.pdf")

# ---------------- Upload & Export ----------------
if page == "Upload & Export":
    st.header("📁 Upload Healthcare CSV File")
    uploaded = st.file_uploader("Upload your healthcare CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())
        df.to_csv("data/uploaded_healthcare.csv", index=False)
        st.info("Saved to data/uploaded_healthcare.csv")

    st.header("📤 Export Current Data as CSV")
    if not data.empty:
        st.download_button("⬇️ Download Processed CSV", data.to_csv(index=False).encode("utf-8"),
                           file_name="processed_healthcare.csv", mime="text/csv")

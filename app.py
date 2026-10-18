"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Loads the Support Vector Classifier trained in
Heart_Disease_Prediction_System.ipynb (heart_disease_svc_trained_model.pkl)
and recreates the exact same LabelEncoders (fit on the same fixed category
lists), so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_trained_model.pkl"

GENDER = ["Male", "Female"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA = ["Yes", "No"]
THAL = ["Normal", "Fixed Defect", "Reversible Defect"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """Same encoders as Step 5.1 of the notebook (codes = alphabetical order)."""
    return {
        "Gender": LabelEncoder().fit(GENDER),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
        "Thal": LabelEncoder().fit(THAL),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter a patient's details below. A **Support Vector Classifier** trained on the
    **UCI Heart Disease (Cleveland)** dataset predicts whether heart disease is present.

    *Test accuracy: **75%** (61 held-out patients).*
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", GENDER)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
with col2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA, index=1)
    thal = st.selectbox("Thal (Thallium Stress Test)", THAL)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Gender": [gender],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Thal": [thal],
    })
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE: YES")
    else:
        st.success("### ✅ HEART DISEASE: NO")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for Assignment 2 (End-to-End ML System). For educational purposes only - "
    "not a medical diagnosis tool."
)

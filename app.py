import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Risk Predictor", layout="centered")

st.title("🎯 Job Readiness Risk Predictor")

st.write("Answer the questions below:")

# ---------------- INPUT ----------------
inputs = []

for i in range(8):
    val = st.slider(f"Question {i+1}", 1, 5, 3)
    inputs.append(val)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    data = np.array([inputs])
    data = scaler.transform(data)

    pred = model.predict(data)[0]
    probs = model.predict_proba(data)[0]

    labels = {0:"Low Risk Capacity", 1:"Medium Risk Capacity", 2:"High Risk Capacity"}

    st.subheader(f"Prediction: {labels[pred]}")

    st.write("Confidence:")
    st.write(probs)

    st.bar_chart(probs)

    # ---------------- RECOMMENDATION ----------------
    if pred == 2:
        st.success("You are well prepared for real-world situations ✅")

    elif pred == 1:
        st.warning("You need improvement in some areas ⚠️")

    else:
        st.error("You need serious preparation and skill development ❌")

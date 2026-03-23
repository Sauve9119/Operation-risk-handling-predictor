import streamlit as st
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Readiness Predictor", layout="wide")
@st.cache_data
def load_data():
     df = pd.read_csv("How Ready Are You for Real-World Job Situations_  (Responses) - Form Responses 1.csv")
     df = df.iloc[:,2:]
     return df
df = load_data()
for col in df.columns:
    unique_vals = df[col].unique()
    mapping = {val: i+1 for i, val in enumerate(unique_vals)}
    df[col] = df[col].map(mapping)
     
# load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# # correlation heatmap
# corr_matrix = df.corr()

#  # Plot the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, 
#             annot=True,      # Show numerical values
#             cmap='coolwarm', # Color scheme
#             vmin=-1, vmax=1, # Scale for correlation
#             fmt=".2f",       # Format to 2 decimal places
#             linewidths=0.5)  # Add lines between cells

# plt.title('Correlation Heatmap')
# plt.show()

# ---------------- LOAD MODEL ----------------
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# st.set_page_config(page_title="Risk Predictor", layout="centered")

# st.title("🎯 Job Readiness Risk Predictor")

# st.write("Answer the questions below:")

# # ---------------- INPUT ----------------
# inputs = []

# for i in range(8):
#     val = st.slider(f"Question {i+1}", 1, 5, 3)
#     inputs.append(val)

# # ---------------- PREDICTION ----------------
# if st.button("Predict"):

#     data = np.array([inputs])
#     data = scaler.transform(data)

#     pred = model.predict(data)[0]
#     probs = model.predict_proba(data)[0]

#     labels = {0:"Low Risk Capacity", 1:"Medium Risk Capacity", 2:"High Risk Capacity"}

#     st.subheader(f"Prediction: {labels[pred]}")

#     st.write("Confidence:")
#     st.write(probs)

#     st.bar_chart(probs)

#     # ---------------- RECOMMENDATION ----------------
#     if pred == 2:
#         st.success("You are well prepared for real-world situations ✅")

#     elif pred == 1:
#         st.warning("You need improvement in some areas ⚠️")

#     else:
#         st.error("You need serious preparation and skill development ❌")
# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Analysis", "🤖 Prediction"])

# ===================== PAGE 1 =====================
if page == "📊 Data Analysis":

    st.title("📊 Data Analysis Dashboard")

    # ---- DATA ----
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---- HEATMAP ----
    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    # ---- DISTRIBUTION ----
    st.subheader("Response Distribution")

    fig2 = df.hist(figsize=(10,6))
    st.pyplot(fig2[0][0].figure)

# ===================== PAGE 2 =====================
elif page == "🤖 Prediction":

    st.title("🎯 Job Readiness Risk Predictor")

    st.write("Rate yourself on following questions (1–5):")

    inputs = []

    col1, col2 = st.columns(2)

    with col1:
        for i in range(4):
            val = st.slider(f"Question {i+1}", 1, 5, 3)
            inputs.append(val)

    with col2:
        for i in range(4, 8):
            val = st.slider(f"Question {i+1}", 1, 5, 3)
            inputs.append(val)

    # ---- PREDICT ----
    if st.button("Predict Risk"):

        data = np.array([inputs])
        data = scaler.transform(data)

        pred = model.predict(data)[0]
        probs = model.predict_proba(data)[0]

        labels = {0:"Low Risk", 1:"Medium Risk", 2:"High Risk"}

        st.subheader(f"Prediction: {labels[pred]}")

        # ---- PROBABILITY GRAPH ----
        st.subheader("Confidence")

        prob_df = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Probability": probs
        })

        st.bar_chart(prob_df.set_index("Risk Level"))

        # ---- RECOMMENDATION ----
        st.subheader("Recommendation")

        if pred == 0:
            st.success("You are well prepared for job situations ✅")

        elif pred == 1:
            st.warning("You need improvement in some areas ⚠️")
            st.markdown("""
            - Improve technical skills  
            - Practice problem solving  
            - Work on real-world projects  
            """)

        else:
            st.error("You need serious preparation ❌")
            st.markdown("""
            - Focus on basics  
            - Improve communication  
            - Build strong projects  
            - Practice interviews  
            """)

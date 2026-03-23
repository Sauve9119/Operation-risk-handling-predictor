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

def main():
    # Initialize session state variables if they don't exist
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'le' not in st.session_state:
        st.session_state.le = None
    if 'ct' not in st.session_state:
        st.session_state.ct = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    if page == "Data Exploration":
        data_exploration()
    elif page == "Model Training":
        model_training()
    elif page == "Predictions":
        make_predictions()
    
    # Add footer to the bottom-left corner of the sidebar
    st.sidebar.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                text-align: left;
                padding: 10px;
                font-size: 12px;
                color: #888;
            }
        </style>
        <div class="footer">
             <p>If this guess is wrong, blame the dataset not me.</p>
             <p>Made with 💻 by <a href="https://github.com/aashu-0" target="_blank" style="color: #007acc; text-decoration: none;">@aashu-0</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )
     
# load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Analysis", "🤖 Prediction"])

# ===================== PAGE 1 =====================
if page == "📊 Data Analysis":
   # ---- DATA ----
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
     
 # Display data information
    st.subheader("Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
    with col2:
        buffer = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count().values,
            'Data Type': df.dtypes.values
        })
        st.dataframe(buffer)
st.subheader("📊 Data Visualizations")

viz_type = st.selectbox(
    "Select Visualization",
    [
        "Question-wise Average Score",
        "Response Distribution",
        "Correlation Heatmap",
        "Overall Score Distribution",
        "Boxplot Analysis"
    ]
)

# ---------------- 1. Average Score ----------------
if viz_type == "Question-wise Average Score":

    avg_scores = df.mean()

    fig, ax = plt.subplots()
    avg_scores.plot(kind='bar', ax=ax)

    ax.set_title("Average Score per Question")
    ax.set_xlabel("Questions")
    ax.set_ylabel("Average Score")

    st.pyplot(fig)

# ---------------- 2. Distribution ----------------
elif viz_type == "Response Distribution":

    fig = df.hist(figsize=(10,6))
    st.pyplot(fig[0][0].figure)

# ---------------- 3. Heatmap ----------------
elif viz_type == "Correlation Heatmap":

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# ---------------- 4. Overall Score ----------------
elif viz_type == "Overall Score Distribution":

    df["Total_Score"] = df.sum(axis=1)

    fig, ax = plt.subplots()
    ax.hist(df["Total_Score"], bins=10)

    ax.set_title("Total Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

# ---------------- 5. Boxplot ----------------
elif viz_type == "Boxplot Analysis":

    fig, ax = plt.subplots()
    sns.boxplot(data=df, ax=ax)

    ax.set_title("Boxplot of All Questions")

    st.pyplot(fig)

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

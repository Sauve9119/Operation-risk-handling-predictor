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
                       <p>Made with 💻 by <a href="https://github.com/Sauve9119" target="_blank" style="color: #007acc; text-decoration: none;">@Rachit-gupta</a></p>
                  </div>
                  """,
                  unsafe_allow_html=True
     )
for col in df.columns:
    unique_vals = df[col].unique()
    mapping = {val: i+1 for i, val in enumerate(unique_vals)}
    df[col] = df[col].map(mapping)

inputs = []

     
# load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [" Data Analysis", "Model Training", " Prediction"])

# ===================== PAGE 1 =====================
if page == " Data Analysis":
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

# # ---------------- 3. Heatmap ----------------
#     elif viz_type == "Correlation Heatmap":

         # corr = df.corr()
     
         # fig, ax = plt.subplots(figsize=(6,5))
         # sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
     
         # st.pyplot(fig)

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

    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
     
     #   summary
    st.subheader("Summary Statistics of Numerical Columns")
    st.dataframe(df.describe())
         
elif page == "Model Training":

    st.header("Model Training")
     
         # ---------------- FEATURES ----------------
    X = df.iloc[:,0:8]
    X_scaled = scaler.transform(X)
     
         # ---------------- TARGET ----------------
         # GMM clustering से labels बनाओ (same as training)
    gmm = joblib.load("gmm.pkl")
     
    y = gmm.predict(X_scaled)
   

    # -------- SHOW DISTRIBUTION --------
    st.subheader("Risk Distribution (Low / Medium / High)")
     # Mapping dictionary banayein
    mapping = {1: 0, 3: 1, 4: 2}
    df['Risk'] = pd.Series(y).map(mapping)
     # Ab chart show karein
    st.bar_chart(df['Risk'].value_counts())

    # -------- SPLIT --------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # -------- SCALE --------
    scaler = joblib.load("scaler.pkl")
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    st.success("Preprocessing Done ✅")

    # -------- MODEL SELECT --------
    model_option = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "SVM", "KNN" , "Random Forest"]
    )

    # -------- TRAIN --------
    if st.button("Train Model"):

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        if model_option == "Logistic Regression":
            model = LogisticRegression(C = 0.5)

        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=3)

        elif model_option == "SVM":
            model = SVC(kernel='rbf',
                         C= 10 ,
                         gamma = 0.5,
                         probability = True,
                         random_state = 42
                         )
        elif model_option == "Random Forest":
             model = RandomForestClassifier()

        else:
            model = KNeighborsClassifier(n_neighbors=3)

        model.fit(X_train, y_train)

        st.session_state.trained_model = model

        # -------- PREDICT --------
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

        accuracy = accuracy_score(y_test, y_pred)
        unique_classes = np.unique(y_test)

        report = classification_report(y_test, y_pred,
                                            labels=unique_classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
                
                # Save metrics
        st.session_state.model_metrics = {
                    "accuracy": accuracy,
                    "report": report,
                    "cm": cm,
                    "class_names": class_names
                }
                
        st.success(f"Model training completed with accuracy: {accuracy:.2f}")

        # -------- CONFUSION MATRIX --------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)

        st.pyplot(fig)


# ===================== PAGE 3 =====================
     
elif page == " Prediction":

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
     
             if pred == 2:
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

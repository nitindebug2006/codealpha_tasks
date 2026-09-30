import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

df = pd.read_csv("data/heart_disease_clean.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient medical details to predict the possibility of heart disease.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=140)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

with col2:
    restecg = st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=2)

st.divider()

if st.button("Predict Disease", type="primary"):

    patient = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=X.columns)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    if prediction == 1:
        st.error("Disease detected")
    else:
        st.success("No disease detected")

    st.metric(
        "Disease Probability",
        f"{probability * 100:.2f}%"
    )

st.caption("This application is for educational/project purposes and is not a medical diagnosis.")
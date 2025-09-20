import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import time
#from streamlit_lottie import st_lottie
import requests

# ------------------- Helper to load Lottie Animations -------------------
#def load_lottie_url(url: str):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()

# ✅ Working Lottie Animations
#heartbeat_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_tutvdkg0.json")
#success_animation   = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
#danger_animation    = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")

# -------------------

# Load model, scaler, and expected columns
# -------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------
# Page config and styling
# -------------------
st.set_page_config(page_title="Heart Stroke Prediction", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fad0c4);
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #D50000;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #333333;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        color:black;          /* <-- Add this line */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        margin-top: 40px;
}

    .stButton>button {
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #ee0979, #ff6a00);
    }
    .result {
        padding: 18px;
        border-radius: 12px;
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>❤️ Heart Stroke Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Developed by Tanmay Deshmukh</div>", unsafe_allow_html=True)



# -------------------
# Main layout: inputs + heartbeat graph
# -------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🩺 Provide Your Health Details")
    left, right = st.columns(2)

    with left:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    with right:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("🔍 Predict Risk", use_container_width=True):

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            
            st.markdown('<div class="result" style="background-color:#ff4d4d; color:white;">⚠️ High Risk of Heart Disease</div>', unsafe_allow_html=True)
        else:
            
            st.markdown('<div class="result" style="background-color:#2ecc71; color:white;">✅ Low Risk of Heart Disease</div>', unsafe_allow_html=True)

# -------------------
# Right column: continuous heartbeat with glow
with col2:
    st.subheader("❤️ Live Heartbeat Monitor")
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://media.tenor.com/0XjJbBPcNocAAAAM/heartbeat-pulse.gif" 
                 width="100%" style="border-radius:15px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------
# Developer info card
# -------------------
st.markdown(
    """
    <div class='card'>
        <h3>👨‍💻 Developer Info</h3>
        <p><b>Name:</b> Tanmay Nilkanth Deshmukh</p>
        <p><b>Email:</b> tndeshmukh11@gmail.com</p>
        <p><b>GitHub:</b> <a href='https://github.com/Deshvan11' target='_blank'>github.com/Deshvan11</a></p>
        <p><b>Portfolio:</b> <a href='https://deshvan11.github.io' target='_blank'>deshvan11.github.io</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

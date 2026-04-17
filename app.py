import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path
# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Employee Prediction System",
    page_icon="📊",
    layout="wide"
)

# =========================
# Load Models
# =========================

BASE_DIR = Path(__file__).resolve().parent

model = pickle.load(open(BASE_DIR / "random_forest_salary.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "scaler_features.pkl", "rb"))
target_scaler = pickle.load(open(BASE_DIR / "scaler_target.pkl", "rb"))
# =========================
# Custom UI Styling
# =========================
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: bold;
    color: #00C9A7;
}
.subtitle {
    color: #BBBBBB;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background-color: #1E1E2F;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.3);
}
.result {
    padding: 25px;
    border-radius: 15px;
    background-color: #111827;
    text-align: center;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background-color: #00C9A7;
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<p class="title">💰 Salary Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning based prediction</p>', unsafe_allow_html=True)

st.divider()

# =========================
# Layout
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUT SECTION
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Enter Details")

    age = st.slider("Age", 18, 65, 25)

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    education_map = {
        "High School": 0,
        "Diploma": 1,
        "Bachelor": 2,
        "Master": 3,
        "PhD": 4,
        "Other": 5
    }

    education_label = st.selectbox("Education Level", list(education_map.keys()))
    education = education_map[education_label]

    experience = st.slider("Years of Experience", 0, 40, 2)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# OUTPUT SECTION
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if st.button("🚀 Predict Salary"):

        # Encode gender (same as training)
        gender_encoded = 1 if gender == "Male" else 0

        # Create input (RAW VALUES)
        input_data = np.array([[age, gender_encoded, education, experience]])

        # SCALE INPUT (MOST IMPORTANT STEP)
        input_scaled = scaler.fit_transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
        salary = int(prediction[0][0])

        # Display nicely
        st.markdown(f"""
        <div class="result">
            <h1 style="color:#00C9A7;">₹ {salary:,}</h1>
            <p>Predicted Salary</p>
        </div>
        """, unsafe_allow_html=True)

        # Extra Info
        st.info(f"""
        📌 Input Summary:
        - Age: {age}
        - Gender: {gender}
        - Education: {education_label}
        - Experience: {experience} years
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.divider()
st.caption("Built with ❤️ using Streamlit")

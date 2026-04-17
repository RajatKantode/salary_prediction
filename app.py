import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Employee Salary Prediction System",
    page_icon="📊",
    layout="wide"
)

# =========================
# Load Models
# =========================
BASE_DIR = Path(__file__).resolve().parent

try:
    with open(BASE_DIR / "random_forest_salary.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "scaler_target.pkl", "rb") as f:
        target_scaler = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

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

        try:
            # Encode gender
            gender_encoded = 1 if gender == "Male" else 0

            # Raw input array
            input_array = np.array([[age, gender_encoded, education, experience]])

            # ⚠️ Column-wise scaling (same as your training)
            input_scaled = np.zeros_like(input_array, dtype=float)

            for i in range(input_array.shape[1]):
                sc = StandardScaler()
                input_scaled[:, i] = sc.fit_transform(
                    input_array[:, i].reshape(-1, 1)
                ).flatten()

            # Predict
            prediction = model.predict(input_scaled)

            # Inverse scaling of target
            prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))

            salary = int(prediction[0][0])

            # Display result
            st.markdown(f"""
            <div class="result">
                <h1 style="color:#00C9A7;">₹ {salary:,}</h1>
                <p>Predicted Annual Salary</p>
            </div>
            """, unsafe_allow_html=True)

            # Input Summary
            st.info(f"""
            📌 Input Summary:
            - Age: {age}
            - Gender: {gender}
            - Education: {education_label}
            - Experience: {experience} years
            """)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.divider()
st.caption("Built with ❤️ using Streamlit")

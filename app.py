import streamlit as st
import numpy as np
import pickle

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
model = pickle.load(open("rf_model.pkl", "rb"))
feature_scaler = pickle.load(open("feature_scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

try:
    target_scaler = pickle.load(open("target_scaler.pkl", "rb"))
except:
    target_scaler = None

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #00C9A7;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: #1E1E2F;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #00C9A7;
            color: white;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<p class="title">📊 Employee Prediction System</p>', unsafe_allow_html=True)
st.write("Predict outcomes based on employee details using Machine Learning")

st.divider()

# =========================
# Layout (2 Columns)
# =========================
col1, col2 = st.columns([1, 1])

# =========================
# INPUT SECTION
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Input Details")

    age = st.slider("Age", 18, 65, 25)

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    education = st.selectbox(
        "Education Level",
        ["0 - High School", "1 - Diploma", "2 - Bachelor", "3 - Master", "4 - PhD", "5 - Other"]
    )

    experience = st.slider("Years of Experience", 0, 40, 2)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# OUTPUT SECTION
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if st.button("🔍 Predict"):
        # Encode inputs
        gender_encoded = 1 if gender == "Male" else 0
        education_encoded = int(education.split(" - ")[0])

        input_data = np.array([[age, gender_encoded, education_encoded, experience]])

        # Scale
        input_scaled = feature_scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # Decode label if needed
        try:
            prediction = label_encoder.inverse_transform(prediction)
        except:
            pass

        # Inverse scale if regression
        if target_scaler:
            prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))

        # Display result
        st.success(f"✅ Prediction: {prediction[0]}")

        # Optional probability (if classification)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.write("### 📊 Confidence Scores")
            st.bar_chart(probs)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.divider()
st.caption("Built with ❤️ using Streamlit | Machine Learning Project")
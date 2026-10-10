import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Health Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #101b32 50%, #0b1325 100%);
        color: white;
    }

    /* Remove extra top space */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Main heading */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #4facfe, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #aab5c8;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Info cards */
    .info-card {
        background: rgba(25, 38, 62, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    .info-number {
        font-size: 28px;
        font-weight: 700;
        color: #4facfe;
    }

    .info-label {
        color: #9aa8bd;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Section heading */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
        color: #f1f5f9;
    }

    /* Prediction result */
    .result-box {
        background: linear-gradient(135deg, #172554, #312e81);
        border: 1px solid rgba(129,140,248,0.35);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 30px;
    }

    .result-label {
        color: #b8c4dc;
        font-size: 17px;
    }

    .result-value {
        font-size: 42px;
        font-weight: 800;
        color: #67e8f9;
        margin-top: 8px;
    }

    .result-note {
        color: #aebbd0;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-size: 17px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.35);
    }

    /* Input labels */
    label {
        color: #dbe4f2 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
loaded_model = pickle.load(open("insurance_model.pkl", "rb"))


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🏥 Health Insurance Cost Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based estimation of health insurance costs</div>',
    unsafe_allow_html=True
)


# ---------------- INFO CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">ML</div>
        <div class="info-label">Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">6</div>
        <div class="info-label">Input Features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">LR</div>
        <div class="info-label">Regression Model</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">💰</div>
        <div class="info-label">Cost Prediction</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- INPUT SECTION ----------------
st.markdown(
    '<div class="section-title">👤 Enter Personal Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["Southeast", "Southwest", "Northeast", "Northwest"]
    )


# ---------------- ENCODING ----------------
sex_value = 0 if sex == "Male" else 1

smoker_value = 0 if smoker == "Yes" else 1

region_values = {
    "Southeast": 0,
    "Southwest": 1,
    "Northeast": 2,
    "Northwest": 3
}

region_value = region_values[region]


# ---------------- PREDICTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Insurance Cost"):

    input_data = (
        age,
        sex_value,
        bmi,
        children,
        smoker_value,
        region_value
    )

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    predicted_cost = prediction[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Insurance Cost</div>
        <div class="result-value">${predicted_cost:,.2f}</div>
        <div class="result-note">
            This is an estimated value generated by the machine learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Health Insurance Cost Prediction • Machine Learning Project
</div>
""", unsafe_allow_html=True)
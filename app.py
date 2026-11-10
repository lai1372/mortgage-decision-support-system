"""Streamlit web app for mortgage risk assessment."""
import streamlit as st
import joblib
from pathlib import Path
import numpy as np
from src.decisions import advisory_decision, validate_score

# Configure page
st.set_page_config(
    page_title="Mortgage Risk Assessment",
    page_icon="🏦",
    layout="centered"
)

# Load model
MODEL_FILE = Path(__file__).resolve().parent / 'models' / 'baseline_model.joblib'

@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None


# Title
st.title("🏦 Mortgage Risk Assessment")
st.divider()

# Load model
model = load_model()
if model is None:
    st.error("⚠️ Model not found. Please train the model first by running: `python3 -m src.main`")
    st.stop()

# Simple input fields
credit_score = st.number_input(
    "Credit Score (300-850)",
    min_value=300,
    max_value=850,
    value=680,
    step=1
)

debt_ratio = st.number_input(
    "Debt-to-Income Ratio (0-1)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01
)

missed_payments = st.number_input(
    "Missed Payments (last 24 months)",
    min_value=0,
    max_value=24,
    value=0,
    step=1
)

# Assessment button
if st.button("🔍 Assess Risk", use_container_width=True, type="primary"):
    try:
        # Prepare features
        features = np.array([[credit_score, debt_ratio, missed_payments]])

        # Get prediction
        risk_score = float(model.predict_proba(features)[0][1])

        # Validate score
        risk_score = validate_score(risk_score)

        # Get decision
        decision = advisory_decision(risk_score)

        # Display results
        st.divider()
        st.subheader("Assessment Result")

        # Create columns for result display
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Risk Score", f"{risk_score:.4f}", help="Score between 0 (low risk) and 1 (high risk)")

        # Decision with color coding
        if decision == "low risk":
            with result_col2:
                st.metric("Classification", "🟢 LOW RISK")
            st.success("High confidence of successful repayment - human review required")
        elif decision == "high risk":
            with result_col2:
                st.metric("Classification", "🔴 HIGH RISK")
            st.error("High likelihood of repayment difficulty - human review required")

        else:  # further review
            with result_col2:
                st.metric("Classification", "🟡 FURTHER REVIEW")
            st.warning("Score requires human assessment")


    except ValueError as e:
        st.error(f"❌ Error: Please enter valid values")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")



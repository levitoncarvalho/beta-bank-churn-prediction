import streamlit as st
import pandas as pd
import joblib
import config

st.set_page_config(
    page_title="Beta Bank - Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Beta Bank Customer Churn Prediction")
st.markdown("""
This app uses an optimized **Random Forest** model to predict whether a customer is likely to leave the bank.

Fill in the details below and click **Predict** to see the result.
""")

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'models/best_model.pkl' and 'models/scaler.pkl' exist.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 650, help="Customer's credit score (300-850)")
    geography = st.selectbox("Country", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance (€)", 0.0, value=50000.0, step=1000.0, format="%.2f")
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_crcard = st.checkbox("Has Credit Card?", value=True)
    is_active = st.checkbox("Is Active Member?", value=True)
    estimated_salary = st.number_input("Estimated Annual Salary (€)", 0.0, 100000.0, step=5000.0, format="%.2f")

if st.button("🔍 Predict Churn Risk", use_container_width=True):
    input_dict = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_crcard),
        'IsActiveMember': int(is_active),
        'EstimatedSalary': estimated_salary
    }
    input_df = pd.DataFrame([input_dict])

    # Scale numeric features using config list
    input_df[config.NUMERIC_COLS] = scaler.transform(input_df[config.NUMERIC_COLS])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure columns match those used during training
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]


    # Predict
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ **High churn risk** (exit probability: {proba:.1%})")
        st.markdown("Immediate retention action is recommended (e.g., personalized offers, manager contact).")
    else:
        st.success(f"✅ **Low churn risk** (exit probability: {proba:.1%})")
        st.markdown("Customer is likely to stay with the bank.")

    st.progress(proba)
    st.caption(f"Exact probability: {proba:.4f}")

st.markdown("---")
st.caption("Developed by Leviton Lima Carvalho as a portfolio project | Model trained on Beta Bank historical data")
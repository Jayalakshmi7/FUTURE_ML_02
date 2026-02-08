# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")  # Ensure churn_model.pkl is in same folder

    # Automatically get feature names
    if hasattr(model, "get_booster"):  # XGBoost
        feature_cols = model.get_booster().feature_names
    else:  # scikit-learn
        feature_cols = model.feature_names_in_
    return model, feature_cols

model, feature_cols = load_model()

# -----------------------------
# App title
# -----------------------------
st.set_page_config(page_title="Churn Prediction System", page_icon="📊")
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("""
Predict whether a telecom customer will **churn (leave)** or **stay**.
""")

# -----------------------------
# Customer input form
# -----------------------------
st.header("Customer Details Input")

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges (USD)", min_value=18.0, max_value=120.0, value=70.0, step=0.1)
total_charges = st.number_input("Total Charges (USD)", min_value=18.0, max_value=9000.0, value=1500.0, step=1.0)

contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# -----------------------------
# Prepare input data
# -----------------------------
input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

# 1️⃣ Assign numeric columns
if "tenure" in feature_cols:
    input_df["tenure"] = tenure
if "MonthlyCharges" in feature_cols:
    input_df["MonthlyCharges"] = monthly_charges
if "TotalCharges" in feature_cols:
    input_df["TotalCharges"] = total_charges

# 2️⃣ Assign categorical columns
for cat, val in [
    ("Contract_" + contract_type, 1),
    ("InternetService_" + internet_service, 1),
    ("PaymentMethod_" + payment_method, 1)
]:
    if cat in feature_cols:
        input_df[cat] = val

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # probability of churn

    # Display Prediction and Probability
    col1, col2 = st.columns(2)
    col1.metric("Prediction", "Churn" if pred==1 else "Stay")
    col2.metric("Churn Probability", f"{prob*100:.2f}%")

    # Business Recommendations
    st.subheader("💡 Business Recommendations")
    if pred == 1:
        st.markdown("""
        - Customer is likely to **leave**.  
        - Offer **discounts, loyalty programs, or personalized plans**.  
        - Provide **better support** to increase retention.  
        """)
    else:
        st.markdown("""
        - Customer is likely to **stay**.  
        - Maintain **good relationship and engagement**.  
        - Monitor for any potential dissatisfaction.  
        """)

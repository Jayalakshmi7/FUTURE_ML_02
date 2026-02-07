import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction System")
st.markdown(
    "Predict whether a telecom customer is likely to **churn** or **stay** with the company. "
    "Enter customer details in the sidebar and click **Predict Churn**."
)

# -----------------------------
# LOAD MODEL AND COLUMNS
# -----------------------------
# Load trained model and model columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("churn_columns.pkl")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Customer Details")

# Numeric inputs (₹)
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
monthly_charges = st.sidebar.number_input("Monthly Charges (₹)", min_value=20, max_value=15000, value=3000, step=10)
total_charges = st.sidebar.number_input("Total Charges (₹)", min_value=20, max_value=900000, value=6000, step=50)

# Categorical inputs
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox(
    "Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("All fields are required to get an accurate churn prediction.")

# -----------------------------
# PREPARE INPUT FOR MODEL
# -----------------------------
# Initialize all model columns with 0
input_dict = {col: 0 for col in model_columns}

# Fill numeric features
if "tenure" in model_columns:
    input_dict["tenure"] = tenure
if "MonthlyCharges" in model_columns:
    input_dict["MonthlyCharges"] = monthly_charges
if "TotalCharges" in model_columns:
    input_dict["TotalCharges"] = total_charges

# Fill one-hot encoded categorical features
if f"Contract_{contract}" in model_columns:
    input_dict[f"Contract_{contract}"] = 1
if f"InternetService_{internet}" in model_columns:
    input_dict[f"InternetService_{internet}"] = 1
if f"PaymentMethod_{payment}" in model_columns:
    input_dict[f"PaymentMethod_{payment}"] = 1

input_data = pd.DataFrame([input_dict])

# -----------------------------
# PREDICTION
# -----------------------------
st.header("📊 Churn Prediction")

if st.button("Predict Churn"):
    # Predict churn and probability
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Display prediction and probability side by side
    col1, col2 = st.columns(2)
    col1.metric("Prediction", "Churn ⚠️" if pred == 1 else "Stay ✅")
    col2.metric("Churn Probability", f"{prob*100:.2f}%")

    # Show color-coded probability bar
    st.subheader("Churn Probability")
    st.progress(int(prob*100))

    # -----------------------------
    # BUSINESS RECOMMENDATIONS
    # -----------------------------
    st.header("💡 Business Recommendations")

    recommendations = []
    if pred == 1:  # Customer likely to churn
        # Contract type
        if contract == "Month-to-month":
            recommendations.append("Offer 1-year or 2-year contract discounts to month-to-month customers.")
        # Monthly charges
        if monthly_charges > 3000:
            recommendations.append("Provide personalized discount plans to high monthly charge customers.")
        # Internet service
        if internet == "Fiber optic" and tenure < 6:
            recommendations.append("Introduce loyalty programs or value-added services for Fiber optic customers with low tenure.")
        # Payment method
        if payment in ["Electronic check", "Mailed check"]:
            recommendations.append("Encourage auto-payment methods like bank transfer or credit card to reduce churn risk.")
        # Short tenure
        if tenure < 6:
            recommendations.append("Engage with new customers proactively via calls, emails, or special offers.")

        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.markdown("Customer is stable. Continue regular engagement and maintain service quality.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "Interactive dashboard allows testing multiple customer scenarios quickly and effectively."
)

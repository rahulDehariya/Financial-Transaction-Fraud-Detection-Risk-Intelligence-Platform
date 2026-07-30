import pandas as pd
import streamlit as st

from src.models.load_model import ModelLoader
from src.services.prediction_service import PredictionService

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Financial Transaction Fraud Detection & Risk Intelligence Platform")

model = ModelLoader.load()

service = PredictionService(model)

col1, col2 = st.columns(2)

with col1:

    transaction_id = st.text_input(
        "Transaction ID",
        "TX10001"
    )

    customer_id = st.text_input(
        "Customer ID",
        "C1001"
    )

    transaction_datetime = st.text_input(
        "Transaction Date Time",
        "2026-07-30 20:30:00"
    )

    transaction_amount = st.number_input(
        "Transaction Amount",
        value=4500.0
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "Purchase",
            "Transfer",
            "Withdrawal"
        ]
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "Electronics",
            "Grocery",
            "Travel",
            "Shopping"
        ]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "UPI",
            "Wallet"
        ]
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "Mobile",
            "Desktop",
            "Tablet"
        ]
    )

    account_age = st.number_input(
        "Account Age (Months)",
        value=12
    )

    account_balance = st.number_input(
        "Account Balance",
        value=10000.0
    )

    previous_transactions = st.number_input(
        "Previous Transactions (24h)",
        value=3
    )

    failed_logins = st.number_input(
        "Failed Login Attempts",
        value=0
    )

    international = st.selectbox(
        "International Transaction",
        [0, 1]
    )

    previous_fraud = st.number_input(
        "Previous Fraud Count",
        value=0
    )

if st.button("Predict Fraud", use_container_width=True):

    transaction = {

        "TransactionID": transaction_id,
        "CustomerID": customer_id,
        "TransactionDateTime": transaction_datetime,
        "TransactionAmount": transaction_amount,
        "TransactionType": transaction_type,
        "MerchantCategory": merchant_category,
        "PaymentMethod": payment_method,
        "DeviceType": device_type,
        "AccountAgeMonths": account_age,
        "AccountBalance": account_balance,
        "PreviousTransactions24h": previous_transactions,
        "FailedLoginAttempts": failed_logins,
        "IsInternational": international,
        "PreviousFraudCount": previous_fraud

    }

    result = service.predict_transaction(transaction)

    st.divider()

    st.subheader("Prediction Result")

    prediction = "Fraud" if result["prediction"] else "Legitimate"

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Prediction",
        prediction
    )

    c2.metric(
        "Fraud Probability",
        f'{result["fraud_probability"]*100:.2f}%'
    )

    c3.metric(
        "Risk Score",
        result["risk_score"]
    )

    if result["risk_level"] == "HIGH":

        st.error(
            f"Risk Level : {result['risk_level']}"
        )

    elif result["risk_level"] == "MEDIUM":

        st.warning(
            f"Risk Level : {result['risk_level']}"
        )

    else:

        st.success(
            f"Risk Level : {result['risk_level']}"
        )

    st.info(
        f"Recommendation : {result['recommendation']}"
    )
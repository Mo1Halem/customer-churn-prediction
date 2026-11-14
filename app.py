import pandas as pd
import streamlit as st

from predict import LOW_RISK_CUTOFF, load_model, score

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

YES_NO = ["Yes", "No"]
ADDONS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
ADVICE = {
    "High Risk": ("error", "Likely to leave. Worth a retention offer, for example a discount for an annual contract."),
    "Medium Risk": ("warning", "Some risk. Keep an eye on this customer."),
    "Low Risk": ("success", "Unlikely to leave. No action needed."),
}


@st.cache_resource
def get_model():
    return load_model()


model, threshold, features = get_model()

st.title("Customer Churn Predictor")
st.write("Describe a telecom customer to see how likely they are to cancel their service. The prediction updates in the sidebar as you change the inputs.")

account, services, profile = st.columns(3)

with account:
    st.subheader("Account")
    tenure = st.slider("Months with the company", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = st.selectbox("Paperless billing", YES_NO)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=0.5)

with services:
    st.subheader("Services")
    phone = st.selectbox("Phone service", YES_NO)
    multiple_lines = "No phone service" if phone == "No" else st.selectbox("Multiple lines", YES_NO, index=1)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    if internet == "No":
        addons = {name: "No internet service" for name in ADDONS}
    else:
        addons = {name: st.selectbox(label, YES_NO, index=1, key=name) for name, label in ADDONS.items()}

with profile:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", YES_NO, index=1)
    partner = st.selectbox("Has a partner", YES_NO, index=1)
    dependents = st.selectbox("Has dependents", YES_NO, index=1)

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": int(senior == "Yes"),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": round(tenure * monthly, 2),
}])

result = score(customer, model, threshold, features).iloc[0]
group = str(result["Risk_Group"])
level, advice = ADVICE[group]

with st.sidebar:
    st.header("Prediction")
    st.metric("Churn probability", f"{result['Churn_Probability']:.1%}")
    st.progress(float(result["Churn_Probability"]))
    st.metric("Risk group", group)
    getattr(st, level)(f"**{group}.** {advice}")
    st.caption(
        f"Risk groups: Low below {LOW_RISK_CUTOFF:.0%}, High above {threshold:.0%}. "
        "Total charges are estimated as months x monthly charges. "
        "The model finds patterns in past customers; it does not prove what causes a customer to leave."
    )
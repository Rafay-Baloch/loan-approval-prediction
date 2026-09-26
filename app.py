from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# Load the trained Logistic Regression pipeline
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).with_name(
    "loan_approval_logistic_model.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# The feature names and order must match the training dataset
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area_Rural",
    "Property_Area_Semiurban",
    "Property_Area_Urban"
]


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):
    applicant_income = float(applicant_income)
    coapplicant_income = float(coapplicant_income)
    loan_amount = float(loan_amount)
    loan_term = float(loan_term)

    if (
        applicant_income < 0
        or coapplicant_income < 0
        or loan_amount <= 0
        or loan_term <= 0
    ):
        raise ValueError(
            "Income values cannot be negative, and loan amount "
            "and loan term must be greater than zero."
        )

    # Convert user-friendly inputs into encoded values
    gender_value = 1 if gender == "Male" else 0
    married_value = 1 if married == "Yes" else 0
    education_value = 1 if education == "Graduate" else 0
    self_employed_value = 1 if self_employed == "Yes" else 0
    credit_history_value = 1 if credit_history == "Good" else 0

    property_rural = 1 if property_area == "Rural" else 0
    property_semiurban = 1 if property_area == "Semiurban" else 0
    property_urban = 1 if property_area == "Urban" else 0

    # Prepare the applicant in the training feature format
    applicant_data = pd.DataFrame(
        [{
            "Gender": gender_value,
            "Married": married_value,
            "Dependents": int(dependents),
            "Education": education_value,
            "Self_Employed": self_employed_value,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history_value,
            "Property_Area_Rural": property_rural,
            "Property_Area_Semiurban": property_semiurban,
            "Property_Area_Urban": property_urban
        }],
        columns=FEATURE_COLUMNS
    )

    # Generate class and probability predictions
    predicted_class = int(model.predict(applicant_data)[0])
    probabilities = model.predict_proba(applicant_data)[0]

    rejection_probability = float(probabilities[0]) * 100
    approval_probability = float(probabilities[1]) * 100

    return (
        predicted_class,
        approval_probability,
        rejection_probability
    )


# ---------------------------------------------------------
# Application heading
# ---------------------------------------------------------
st.title("🏦 Loan Approval Prediction Application")

st.write(
    "Enter the applicant's personal and financial information "
    "to receive a loan approval classification and probability."
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.info("Model: Logistic Regression")

with info_col2:
    st.info("Classes: Approved or Rejected")

with info_col3:
    st.info("Decision Threshold: 50%")


# ---------------------------------------------------------
# Applicant information form
# ---------------------------------------------------------
st.subheader("Applicant Information")

with st.form("loan_prediction_form"):
    left_column, right_column = st.columns(2)

    with left_column:
        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            horizontal=True
        )

        married = st.radio(
            "Married",
            ["Yes", "No"],
            horizontal=True
        )

        dependents = st.selectbox(
            "Number of Dependents",
            [0, 1, 2, 3]
        )

        education = st.radio(
            "Education",
            ["Graduate", "Not Graduate"],
            horizontal=True
        )

        self_employed = st.radio(
            "Self Employed",
            ["Yes", "No"],
            index=1,
            horizontal=True
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
            index=2
        )

    with right_column:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1.0,
            value=180.0,
            step=1.0
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
            index=8
        )

        credit_history = st.radio(
            "Credit History",
            ["Good", "Poor"],
            horizontal=True
        )

    predict_button = st.form_submit_button(
        "Predict Loan Approval",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# Display prediction result
# ---------------------------------------------------------
if predict_button:
    try:
        (
            predicted_class,
            approval_probability,
            rejection_probability
        ) = predict_loan(
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        )

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if predicted_class == 1:
                st.success("✅ LOAN APPROVED")
            else:
                st.error("❌ LOAN REJECTED")

        with result_col2:
            st.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )

        with result_col3:
            st.metric(
                "Rejection Probability",
                f"{rejection_probability:.2f}%"
            )

        if predicted_class == 1:
            st.success(
                "The model predicts that this loan application is "
                f"likely to be approved because the approval "
                f"probability is {approval_probability:.2f}%."
            )
        else:
            st.error(
                "The model predicts that this loan application is "
                f"likely to be rejected because the rejection "
                f"probability is {rejection_probability:.2f}%."
            )

        probability_data = pd.DataFrame({
            "Result": ["Approval", "Rejection"],
            "Probability": [
                approval_probability,
                rejection_probability
            ]
        })

        st.bar_chart(
            probability_data.set_index("Result")
        )

    except Exception as error:
        st.error(
            f"An error occurred while generating the prediction: {error}"
        )


# ---------------------------------------------------------
# Example applicant test cases
# ---------------------------------------------------------
st.subheader("Example Applicant Test Cases")

example_cases = pd.DataFrame({
    "Applicant": [
        "Applicant 1",
        "Applicant 2",
        "Applicant 3"
    ],
    "Gender": ["Male", "Female", "Male"],
    "Married": ["Yes", "No", "Yes"],
    "Dependents": [1, 0, 2],
    "Education": [
        "Graduate",
        "Not Graduate",
        "Graduate"
    ],
    "Self Employed": ["No", "Yes", "No"],
    "Applicant Income": [6000, 2000, 5000],
    "Co-applicant Income": [2000, 0, 1500],
    "Loan Amount": [150, 250, 180],
    "Term": [360, 360, 360],
    "Credit History": ["Good", "Poor", "Good"],
    "Property Area": ["Semiurban", "Rural", "Urban"]
})

st.dataframe(
    example_cases,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.warning(
    "Disclaimer: This application is an educational machine "
    "learning project. Its predictions should not be used as the "
    "sole basis for real financial or lending decisions."
)

st.markdown(
    "**Developed by: Abdur Rafay Hassan Baloch**"
)

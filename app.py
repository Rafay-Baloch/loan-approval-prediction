from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


# ---------------------------------------------------------
# Load the trained Logistic Regression pipeline
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).with_name(
    "loan_approval_logistic_model.pkl"
)

model = joblib.load(MODEL_PATH)


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
    try:
        # Validate numerical values
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
            return (
                "INVALID INPUT",
                "Not calculated",
                "Not calculated",
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

        # Prepare the applicant in the same format used during training
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

        if predicted_class == 1:
            prediction = "APPROVED"
            explanation = (
                "The model predicts that this loan application is "
                f"likely to be approved because the approval "
                f"probability is {approval_probability:.2f}%."
            )
        else:
            prediction = "REJECTED"
            explanation = (
                "The model predicts that this loan application is "
                f"likely to be rejected because the rejection "
                f"probability is {rejection_probability:.2f}%."
            )

        return (
            prediction,
            f"{approval_probability:.2f}%",
            f"{rejection_probability:.2f}%",
            explanation
        )

    except Exception as error:
        return (
            "PREDICTION ERROR",
            "Not calculated",
            "Not calculated",
            f"An error occurred while generating the prediction: {error}"
        )


# ---------------------------------------------------------
# Gradio user interface
# ---------------------------------------------------------
with gr.Blocks(
    title="Loan Approval Prediction Application"
) as app:

    gr.Markdown(
        """
        # Loan Approval Prediction Application

        Enter the applicant's personal and financial information
        to receive a loan approval classification and probability.

        **Model:** Logistic Regression  
        **Prediction classes:** Approved or Rejected  
        **Decision threshold:** 50%
        """
    )

    with gr.Row():
        with gr.Column():
            gender = gr.Radio(
                choices=["Male", "Female"],
                value="Male",
                label="Gender"
            )

            married = gr.Radio(
                choices=["Yes", "No"],
                value="Yes",
                label="Married"
            )

            dependents = gr.Dropdown(
                choices=["0", "1", "2", "3"],
                value="0",
                label="Number of Dependents"
            )

            education = gr.Radio(
                choices=["Graduate", "Not Graduate"],
                value="Graduate",
                label="Education"
            )

            self_employed = gr.Radio(
                choices=["Yes", "No"],
                value="No",
                label="Self Employed"
            )

            property_area = gr.Dropdown(
                choices=["Rural", "Semiurban", "Urban"],
                value="Urban",
                label="Property Area"
            )

        with gr.Column():
            applicant_income = gr.Number(
                value=5000,
                label="Applicant Income"
            )

            coapplicant_income = gr.Number(
                value=1500,
                label="Co-applicant Income"
            )

            loan_amount = gr.Number(
                value=180,
                label="Loan Amount"
            )

            loan_term = gr.Dropdown(
                choices=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                value=360,
                label="Loan Amount Term"
            )

            credit_history = gr.Radio(
                choices=["Good", "Poor"],
                value="Good",
                label="Credit History"
            )

    predict_button = gr.Button(
        "Predict Loan Approval",
        variant="primary"
    )

    gr.Markdown("## Prediction Result")

    with gr.Row():
        prediction_output = gr.Textbox(
            label="Loan Prediction",
            interactive=False
        )

        approval_output = gr.Textbox(
            label="Approval Probability",
            interactive=False
        )

        rejection_output = gr.Textbox(
            label="Rejection Probability",
            interactive=False
        )

    explanation_output = gr.Textbox(
        label="Prediction Explanation",
        lines=3,
        interactive=False
    )

    gr.Markdown(
        """
        ### Example Applicant Test Cases

        | Applicant | Gender | Married | Dependents | Education | Self Employed | Applicant Income | Co-applicant Income | Loan Amount | Term | Credit History | Property Area |
        |---|---|---:|---:|---|---|---:|---:|---:|---:|---|---|
        | Applicant 1 | Male | Yes | 1 | Graduate | No | 6000 | 2000 | 150 | 360 | Good | Semiurban |
        | Applicant 2 | Female | No | 0 | Not Graduate | Yes | 2000 | 0 | 250 | 360 | Poor | Rural |
        | Applicant 3 | Male | Yes | 2 | Graduate | No | 5000 | 1500 | 180 | 360 | Good | Urban |
        """
    )

    gr.Markdown(
        """
        ---
        **Disclaimer:** This application is an educational machine
        learning project. Its predictions should not be used as the
        sole basis for real financial or lending decisions.

        **Developed by:** Abdur Rafay Hassan Baloch
        """
    )

    predict_button.click(
        fn=predict_loan,
        inputs=[
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
        ],
        outputs=[
            prediction_output,
            approval_output,
            rejection_output,
            explanation_output
        ]
    )


if __name__ == "__main__":
    app.launch()
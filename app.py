#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("/home/safir/Desktop/py/ML/week8/module29/best_lr_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_LoanStatus(
        Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
        LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, FamilySize
):
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
        LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, FamilySize
    ]],
      columns=[
          'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
          'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'FamilySize'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    status = "Approved" if prediction == 1 else "NOT Approved"
    return f"Loan Status: {status}"

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["No", "Yes"], label="Married"),
    gr.Slider(0, 3, step=1, label="Dependents"),
    gr.Radio(["Graduate", "Not Graduate"], label="Education"),
    gr.Radio(["No", "Yes"], label="Self Employed"),
    gr.Number(label="Applicant Income", value=0),
    gr.Number(label="Coapplicant Income", value=0),
    gr.Number(label="Loan Amount", value=0),
    gr.Dropdown([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], label="Loan Amount Term"),
    gr.Radio([0, 1], label="Credit History"),
    gr.Dropdown(["Urban", "Rural", "Semiurban"], label="Property Area"),
    gr.Number(label="Family Size", value=0)
]

app = gr.Interface(
    fn=predict_LoanStatus,
      inputs=inputs,
        outputs="text", 
        title="Loan Approval Prediction System",
        description="Enter applicant details to predict loan approval status using a trained ML model.")

app.launch(share=True)

# https://f411dbd7b7b00a2230.gradio.live
# public URL (expires after a week)
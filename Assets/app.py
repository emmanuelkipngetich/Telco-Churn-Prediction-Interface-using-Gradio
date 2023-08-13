# Importations
import pandas as pd
import gradio as gr
import os
import pickle


# Creating key List
expected_inputs = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                   'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']


# Function to load machine learning components
def load_components_func(fp):
    # To load the machine learning components saved to re-use in the app
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


# Loading the machine learning components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "ML_Model.pkl")
ml_components_dict = load_components_func(fp=ml_core_fp)

# Unpacking my components
label_encoder = ml_components_dict['label_encoder']
encoder = ml_components_dict['encoder']
imputer = ml_components_dict['imputer']
scaler = ml_components_dict['scaler']
balance = ml_components_dict['imbalance']
model = ml_components_dict['model']


def predict_churn(*args, scaler=scaler, model=model, imputer=imputer, encoder=encoder):

    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Encode the data
    num_col = input_data[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
    cat_col = input_data[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']]
    cat_col = cat_col.astype(str)
    encoded_data = encoder.transform(cat_col)
    encoded_df = pd.concat([num_col, encoded_data], axis=1)

    # Imputing missing values
    imputed_df = imputer.transform(encoded_df)

    # Scaling
   
    scaled_df = scaler.transform(encoded_df)

    # Prediction
    model_output = model.predict_proba(scaled_df)
    #Probability of Churn(Positive class)
    prob_Churn = float(model_output[0][1]) 
    #Probability of staying(Negative Class)
    prob_Stay = 1 - prob_Churn
    return {"Prediction Churn": prob_Churn,
            "Prediction Not Churn": prob_Stay}


# Function to process inputs and return prediction
Gender = gr.Radio(choices=['Male', 'Female'], label="Gender")
Partner = gr.Radio(choices=['Yes', 'No'], label="Partner")
Dependents = gr.Radio(choices=['Yes', 'No'], label="Dependents")
Tenure = gr.Number(label="Tenure")
InternetService = gr.Radio(choices=['DSL', 'Fiber optic', 'No'], label="Internet Service")
PhoneService = gr.Radio(choices=['Yes', 'No'], label="Phone Service")
MultipleLines = gr.Radio(choices=['Yes', 'No'], label="Multiple Lines")
Contract = gr.Radio(choices=['Month-to-month', 'One year', 'Two year'], label="Contract")
MonthlyCharges = gr.Number(label="Monthly Charges")
TotalCharges = gr.Number(label="Total Charges")
PaperlessBilling = gr.Radio(choices=['Yes', 'No'], label='Paperless Billing')
PaymentMethod = gr.Radio(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                'Credit card (automatic)'], label="Payment Method")
OnlineSecurity = gr.Radio(choices=['Yes', 'No'], label="Online Security")
OnlineBackup = gr.Radio(choices=['Yes', 'No', 'None'], label="Online Backup")
DeviceProtection = gr.Radio(choices=['Yes', 'No'], label="Device Protection")
TechSupport = gr.Radio(choices=['Yes', 'No'], label="Tech Support")
StreamingTV = gr.Radio(choices=['Yes', 'No'], label="Streaming TV")
SeniorCitizen = gr.Radio(choices=[0, 1], label='Senior Citizen')
StreamingMovies = gr.Radio(choices=['Yes', 'No'], label="Streaming Movies")

# Output
gr.Interface(inputs=[SeniorCitizen, Tenure, MonthlyCharges, TotalCharges,
                     Gender, Partner, Dependents, PhoneService, MultipleLines,
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                     TechSupport, StreamingTV, StreamingMovies, Contract,
                     PaperlessBilling, PaymentMethod],
             outputs=gr.Label("Awaiting Submission...."),
             fn=predict_churn,
             title=" Teleco Services Customer Churn Prediction",
             description="This model predicts whether a customer will churn or stay with the telecom service based on various input features",
             ).launch(inbrowser=True, show_error=True, share=True)
 
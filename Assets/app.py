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

#Unpacking my components
label_encoder = ml_components_dict['label_encoder']
encoder = ml_components_dict['encoder']
imputer = ml_components_dict['imputer']
scaler = ml_components_dict['imbalance']
model = ml_components_dict['model']

def predict_churn(*args, scaler=scaler, model=model, imputer=imputer, encoder=encoder):
    input_data = pd.DataFrame([args], columns=expected_inputs)

    #Encode the data 
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


# We define our inputs
Gender = gr.Radio(choices=['Male', 'Female'], label="Gender : Gender of the customer")
Partner = gr.Radio(choices=['Yes', 'No'], label="Partner : Whether the customer has a partner.")
Dependents = gr.Radio(choices=['Yes', 'No'], label="Dependents : Whether the customer has dependents.")
Tenure = gr.Number(label="Tenure : The Number of months the customer has been with the company.")
InternetService = gr.Radio(choices=['DSL', 'Fiber optic', 'No'], label="Internet Service : Type of internet service.")
PhoneService = gr.Radio(choices=['Yes', 'No'], label="Phone Service : Whether the customer has phone service.")
MultipleLines = gr.Radio(choices=['Yes', 'No'], label="Multiple Lines : Whether the customer has multiple phone lines.")
Contract = gr.Radio(choices=['Month-to-month', 'One year', 'Two year'], label="Contract : Type of contract the customer has.")
MonthlyCharges = gr.Number(label="Monthly Charges : Amount of monthly charges.")
TotalCharges = gr.Number(label="Total Charges : Total amount charged to the customer.")
PaperlessBilling = gr.Radio(choices=['Yes', 'No'], label='Paperless Billing : Whether the customer uses paperless billing.')
PaymentMethod = gr.Radio(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                'Credit card (automatic)'], label="Payment Method : Payment method used by the customer.")
OnlineSecurity = gr.Radio(choices=['Yes', 'No'], label="Online Security : Whether the customer has online security service.")
OnlineBackup = gr.Radio(choices=['Yes', 'No', 'None'], label="Online Backup : Whether the customer has online backup service.")
DeviceProtection = gr.Radio(choices=['Yes', 'No'], label="Device Protection : Whether the customer has device protection service.")
TechSupport = gr.Radio(choices=['Yes', 'No'], label="Tech Support : Whether the customer has tech support service.")
StreamingTV = gr.Radio(choices=['Yes', 'No'], label="Streaming TV : Whether the customer uses streaming TV service.")
SeniorCitizen = gr.Radio(choices=[0, 1], label='Senior Citizen : Whether the customer is a senior citizen(0 for No and 1 For Yes).')
StreamingMovies = gr.Radio(choices=['Yes', 'No'], label="Streaming Movies : Whether the customer uses streaming movies service.")
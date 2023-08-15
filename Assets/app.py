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

#Loading the machine learning components
def load_components_func(fp):
    #To load the machine learning components to re-use in the app
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
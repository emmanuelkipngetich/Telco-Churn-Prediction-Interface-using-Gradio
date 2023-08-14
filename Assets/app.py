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


# Function to load machine learning components
def load_components_func(fp):
    #To load the machine learning components saved to re-use in the app
    with open(fp,"rb") as f:
        object = pickle.load(f)
    return object

# Loading the machine learning components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH,"Assets","ML_Model.pkl")
ml_components_dict = load_components_func(fp=ml_core_fp)

#Unpacking my components
label_encoder = ml_components_dict['label_encoder']
encoder = ml_components_dict['encoder']
imputer =ml_components_dict['imputer']
scaler = ml_components_dict['scaler']
model = ml_components_dict['model']





components = {
    'label_encoder': label_encoder,
    'encoder': encoder,
    'scaler': scaler,
    'imbalance': smote,
    'grid_search_rfc':grid_search_rf,
    'model': rf_model_tuned_Sn
}
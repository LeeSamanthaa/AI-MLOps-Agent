import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data(file_upload):
    """
    Loads data from an uploaded CSV file into a pandas DataFrame.
    It also sanitizes column names to be compatible with ML models.
    """
    try:
        df = pd.read_csv(file_upload)
        # Sanitize column names: remove special characters and spaces
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

@st.cache_data
def get_sample_datasets():
    """
    Generates and returns a dictionary of sample datasets for classification and regression.
    """
    np.random.seed(42)
    n_samples = 1000
    # Classification Data (Bank Churn)
    data_c = {
        'CustomerID': range(1, n_samples + 1),
        'CreditScore': np.random.randint(400, 850, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 200000, n_samples).round(2),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_samples).round(2)
    }
    df_c = pd.DataFrame(data_c)
    churn_prob = (0.1 + (df_c['Age'] / 100) - (df_c['IsActiveMember'] * 0.15) + (df_c['Geography'] == 'Germany') * 0.1 - (df_c['CreditScore']/10000))
    df_c['Churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    # Regression Data (House Price)
    data_r = {
        'SquareFootage': np.random.randint(800, 4000, n_samples),
        'Bedrooms': np.random.randint(2, 6, n_samples),
        'Bathrooms': np.random.randint(1, 5, n_samples),
        'YearBuilt': np.random.randint(1950, 2020, n_samples)
    }
    df_r = pd.DataFrame(data_r)
    df_r['Price'] = (150 * df_r['SquareFootage'] + 25000 * df_r['Bedrooms'] + 30000 * df_r['Bathrooms'] - 500 * (2025 - df_r['YearBuilt']) + np.random.normal(0, 25000, n_samples)).round(0)
    
    return {
        'Bank Customer Churn (Classification)': df_c,
        'House Price Prediction (Regression)': df_r
    }

class FeatureDetector:
    """A utility class to automatically detect feature types in a DataFrame."""
    @staticmethod
    def detect_features(df, target=None):
        """
        Detects numerical, categorical, and ID features from a DataFrame.
        """
        features = {'numerical': [], 'categorical': [], 'id': []}
        for col in df.columns:
            if col == target:
                continue
            # High cardinality numeric columns are treated as IDs
            if (df[col].nunique() / len(df) > 0.99 and pd.api.types.is_numeric_dtype(df[col])):
                features['id'].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                features['numerical'].append(col)
            else:
                features['categorical'].append(col)
        return features

def generate_fastapi_script(numerical_features, categorical_features, problem_type):
    """
    Generates a Python script for a FastAPI prediction endpoint.
    The script is tailored to the features of the trained model.
    """
    feature_schemas = ",\n    ".join([f"{feat}: float" for feat in numerical_features] + [f"{feat}: str" for feat in categorical_features])
    df_creation_lines = ",\n        ".join([f"'{feat}': [data.{feat}]" for feat in numerical_features + categorical_features])
    
    script = f"""
import uvicorn
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize API
app = FastAPI(
    title="AI MLOps Agent Model API",
    description="API for making predictions with the trained model.",
    version="1.0.0"
)

# 2. Load the trained pipeline
try:
    with open("best_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    pipeline = None # Or handle the error as appropriate

# 3. Define the input data schema
class PredictionFeatures(BaseModel):
    {feature_schemas}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: PredictionFeatures):
    if pipeline is None:
        return {{"error": "Model not loaded. Please ensure best_model.pkl is in the same directory."}}
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame({{
            {df_creation_lines}
        }})

        # Make prediction
        prediction = pipeline.predict(df)
        prediction_value = prediction[0]

        # Convert numpy types to native Python types
        if isinstance(prediction_value, (np.generic)):
            prediction_value = prediction_value.item()

        return {{"prediction": prediction_value}}
    except Exception as e:
        return {{"error": str(e), "details": "An error occurred during prediction."}}

@app.get("/")
def read_root():
    return {{"message": "Welcome to the Model Prediction API. Use the /predict endpoint for predictions."}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    return script

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a UTF-8 encoded CSV file for downloading."""
    return df.to_csv(index=False).encode('utf-8')
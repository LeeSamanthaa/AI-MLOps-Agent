import streamlit as st
import pandas as pd
import numpy as np
import warnings
import json
import hashlib
import logging
import copy
import os
import base64
import re
from io import BytesIO, StringIO
from .transformers import AIFeatureEngineer
from sklearn.metrics import confusion_matrix
import plotly.express as px
import pickle
import zipfile

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings from libraries for a cleaner console output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

class ModelStore:
    """
    Manages the lifecycle of trained model objects.

    Note: This class is NOT thread-safe. It relies on Streamlit's single-threaded
    execution model. Do not use with actual threading (threading.Thread).
    Parallel model training uses joblib which spawns separate processes, so each
    process gets its own ModelStore instance via pickling.
    """
    def __init__(self):
        self._store = {}
        logging.info("ModelStore initialized.")

    def add_model(self, model_id, model_object):
        self._store[model_id] = model_object
        logging.info(f"Model with ID '{model_id}' added to the store.")

    def get_model(self, model_id):
        model = self._store.get(model_id)
        if model:
            logging.info(f"Model with ID '{model_id}' retrieved from the store.")
        else:
            logging.warning(f"Model with ID '{model_id}' not found in the store.")
        return model

    def clear(self):
        self._store = {}
        logging.info("ModelStore cleared.")

def get_default_config():
    """Returns the default configuration dictionary for the app's state."""
    # CHANGES: Added new default config values for Groq API resilience.
    return {
        'problem_type': 'Classification',
        'selected_models': [],
        'groq_api_key': '',
        'groq_model': 'llama-3.1-8b-instant',
        'groq_fallback_model': 'llama-3.1-70b-versatile',
        'api_retry_attempts': 3,
        'api_retry_backoff': 5,
        'primary_metric': 'roc_auc',
        'primary_metric_display': 'ROC AUC',
        'target': None,
        'numerical_features': [],
        'categorical_features': [],
        'test_size': 0.2,
        'cv_folds': 5,
        'imbalance_method': 'None',
        'imputation_strategy': 'median',
        'scaler': 'StandardScaler',
        'handle_outliers': True,
        'use_statistical_feature_selection': False,
        'k_features': 10,
        'bayes_iterations': 25,
        'ensemble_base_models': [],
        'mlflow_uri': './mlruns',
        'mlflow_experiment_name': 'AI_MLOps_Agent_Experiment',
        'quick_test': False,
        'manual_tuning_enabled': False,
        'manual_params': {}
    }

def initialize_session_state(force=False):
    """Initializes the session state with default values, including new workflow state."""
    if 'app_initialized' not in st.session_state or force:
        # Clear existing state before re-initializing
        for key in list(st.session_state.keys()):
            if key in st.session_state:
                del st.session_state[key]
        
        config = get_default_config()
        SESSION_DEFAULTS = {
            'config': config,
            'workflow_step': 1,
            'step_completion': {'ingestion': False, 'eda': False, 'feature_eng': False, 'config': False, 'training': False, 'evaluation': False},
            'data_loaded': False,
            'ai_features_approved': False,
            'feature_recommendations': [],
            'messages': [],
            'run_history': [],
            'parsed_actions': [],
            'df': None,
            'results_summary': None,
            'ai_summary': None,
            'guide_message': None,
            'results_data': None,
            'file_name': None,
            'training_data_stats': None,
            'ai_config_rationale': None,
            'last_api_error': None,
            'view': 'welcome',
            'model_store': ModelStore(),
            'show_eda': False,
            'eda_summary': None,
            'vif_summary': None,
            'shap_fig': None,
            'phase1_complete': False,
            'suggested_base_models': [],
            'phase1_results': None,
            'experiment_history': [],
            'feature_count_before': None,
        }
    
        for key, value in SESSION_DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = copy.deepcopy(value)
        
        for key, value in config.items():
            st.session_state[f'widget_{key}'] = value

        st.session_state.app_initialized = True

def load_data(file_upload):
    """Safely loads data from various file formats with enhanced error handling."""
    MAX_FILE_SIZE_MB = 500
    if file_upload.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit.")
        return None

    try:
        file_name = file_upload.name
        st.session_state.file_name = os.path.splitext(file_name)[0]
        
        file_content = file_upload.getvalue()

        if file_name.endswith('.csv'):
            if file_upload.size > 50 * 1024 * 1024:  # 50MB
                st.warning(f"Large file detected ({file_upload.size / (1024*1024):.1f}MB). Loading first 100,000 rows.")
                df = pd.read_csv(BytesIO(file_content), nrows=100000)
            else:
                df = pd.read_csv(BytesIO(file_content))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_content))
        elif file_name.endswith('.json'):
            df = pd.read_json(StringIO(file_content.decode('utf-8')))
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(file_content))
        else:
            st.error(f"Unsupported file format: {os.path.splitext(file_name)[1]}. Please upload a CSV, Excel, JSON, or Parquet file.")
            return None
        
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')
        return df
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}. Please ensure it is well-formed.")
        logging.error(f"Error reading {file_upload.name}: {e}", exc_info=True)
        return None

@st.cache_data
def profile_data(_df):
    """Generates a data quality report and a score."""
    if _df is None:
        return None, 0

    df = _df.copy()
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    missing_cells = df.isnull().sum().sum()
    completeness_score = max(0, 100 * (1 - missing_cells / total_cells if total_cells > 0 else 1))
    
    duplicate_rows = df.duplicated().sum()
    uniqueness_score = max(0, 100 * (1 - duplicate_rows / n_rows if n_rows > 0 else 1))
    
    overall_score = (completeness_score + uniqueness_score) / 2
    
    report = {
        "Overall Score": f"{overall_score:.1f}/100",
        "Rows": n_rows,
        "Columns": n_cols,
        "Missing Cells": f"{missing_cells} ({missing_cells/total_cells:.2%})" if total_cells > 0 else "0 (0.00%)",
        "Duplicate Rows": f"{duplicate_rows} ({duplicate_rows/n_rows:.2%})" if n_rows > 0 else "0 (0.00%)"
    }
    return report, overall_score

# FIX: Hardened this function against KeyError and other race conditions.
@st.cache_data(show_spinner=False)
def get_processed_df():
    """
    Safely retrieves a copy of the base DataFrame from session state.
    Handles race conditions where session state might be invalid during reruns.
    
    IMPORTANT: After features are approved, they are permanently applied to st.session_state.df,
    so this function should NOT re-apply them.
    """
    # Guard against the 'df' key not existing in session_state at all.
    if 'df' not in st.session_state or st.session_state.df is None:
        return None
    
    try:
        base_df = st.session_state.df.copy()
    except (AttributeError, KeyError) as e:
        # Handles race conditions where st.session_state.df is cleared or becomes invalid during a rerun.
        logging.warning(f"Race condition detected in get_processed_df: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in get_processed_df: {e}", exc_info=True)
        return None
    
    # CRITICAL FIX: Only apply AI features if they haven't been approved yet
    # Once approved, features are permanently in st.session_state.df
    if (not st.session_state.get('ai_features_approved') and 
        st.session_state.get('feature_recommendations')):
        logging.info("Applying AI-engineered features to the dataset (preview mode).")
        try:
            return AIFeatureEngineer(st.session_state.feature_recommendations).transform(base_df)
        except Exception as e:
            logging.error(f"Error applying AI feature engineering: {e}")
            return base_df # Fallback to base_df if transformation fails
    
    # If features already approved, they're in base_df - just return it
    return base_df

@st.cache_data
def get_dataset_sample(_df, n=1000):
    """Returns a sample of the dataframe for faster UI rendering."""
    if _df is None:
        return None
    return _df.head(n)

@st.cache_data
def get_sample_datasets():
    """Generates and returns sample datasets for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    data_c = {'CustomerID': range(1, n_samples + 1), 'CreditScore': np.random.randint(400, 850, n_samples), 'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]), 'Gender': np.random.choice(['Male', 'Female'], n_samples), 'Age': np.random.randint(18, 70, n_samples), 'Tenure': np.random.randint(0, 10, n_samples), 'Balance': np.random.uniform(0, 200000, n_samples).round(2), 'NumOfProducts': np.random.randint(1, 4, n_samples), 'HasCrCard': np.random.choice([0, 1], n_samples), 'IsActiveMember': np.random.choice([0, 1], n_samples), 'EstimatedSalary': np.random.uniform(10000, 200000, n_samples).round(2)}
    df_c = pd.DataFrame(data_c)
    churn_prob = (0.1 + (df_c['Age'] / 100) - (df_c['IsActiveMember'] * 0.15) + (df_c['Geography'] == 'Germany') * 0.1 - (df_c['CreditScore']/10000))
    df_c['Churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    data_r = {'SquareFootage': np.random.randint(800, 4000, n_samples), 'Bedrooms': np.random.randint(2, 6, n_samples), 'Bathrooms': np.random.randint(1, 5, n_samples), 'YearBuilt': np.random.randint(1950, 2020, n_samples)}
    df_r = pd.DataFrame(data_r)
    df_r['Price'] = (150 * df_r['SquareFootage'] + 25000 * df_r['Bedrooms'] + 30000 * df_r['Bathrooms'] - 500 * (2025 - df_r['YearBuilt']) + np.random.normal(0, 25000, n_samples)).round(0)
    return {'Bank Customer Churn (Classification)': df_c, 'House Price Prediction (Regression)': df_r}

class FeatureDetector:
    """Detects feature types (numerical, categorical, id, datetime, text) in a DataFrame."""
    
    @staticmethod
    def is_discrete_code(series):
        """
        Detect if a numeric column is actually a discrete code.
        FIXED: Don't classify interaction features (with decimals) as discrete codes.
        """
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        # If column name suggests it's an engineered feature, keep it as numerical
        col_name = series.name if hasattr(series, 'name') else ""
        if '_x_' in str(col_name) or '_pow' in str(col_name) or '_log' in str(col_name):
            return False
        
        # Check if values are all integers (not floats with decimals)
        non_null = series.dropna()
        if len(non_null) > 0:
            # If there are decimal values, it's not a discrete code
            has_decimals = not np.allclose(non_null, non_null.astype(int), rtol=0, atol=1e-9)
            if has_decimals:
                return False
        
        # High uniqueness ratio suggests ID column
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.9:
            return True
        
        # Check for irregular spacing (suggests codes rather than measurements)
        if series.nunique() > 2:
            sorted_unique = np.sort(series.dropna().unique())
            if len(sorted_unique) > 1:
                diffs = np.diff(sorted_unique)
                if len(diffs) > 0:
                    # Only flag as discrete if spacing is very irregular
                    cv = np.std(diffs) / (np.mean(diffs) + 1e-10)
                    if cv > 1.0:
                        return True
        
        return False
        
    @staticmethod
    def detect_features(df, target=None):
        """
        Enhanced feature detection with support for datetime and text columns.
        FIXED: Better handling of numeric columns stored as object dtype.
        """
        features = {
            'numerical': [],
            'categorical': [],
            'id': [],
            'datetime': [],
            'text': []
        }

        if df is None or df.empty:
            return features

        for col in df.columns:
            if col == target:
                continue

            series = df[col]

            # 1. Check for ID columns
            if series.nunique() == len(df) and pd.api.types.is_numeric_dtype(series):
                features['id'].append(col)
                continue

            # 2. Check if already numeric dtype
            if pd.api.types.is_numeric_dtype(series):
                if FeatureDetector.is_discrete_code(series):
                    features['categorical'].append(col)
                else:
                    features['numerical'].append(col)
                continue

            # 3. Try numeric conversion for object dtype
            if series.dtype == 'object':
                try:
                    numeric_converted = pd.to_numeric(series, errors='coerce')
                    non_null_count = series.notna().sum()
                    if non_null_count > 0 and numeric_converted.notna().sum() / non_null_count > 0.8:
                        if FeatureDetector.is_discrete_code(numeric_converted):
                            features['categorical'].append(col)
                        else:
                            features['numerical'].append(col)
                        continue
                except Exception:
                    pass

            # 4. Check for datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                features['datetime'].append(col)
                continue

            # 5. Check string/object columns
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                sample = series.dropna().astype(str).head(20)
                
                if len(sample) == 0:
                    features['categorical'].append(col)
                    continue
                
                datetime_patterns = [
                    r'^\d{4}-\d{2}-\d{2}',
                    r'^\d{2}/\d{2}/\d{4}',
                    r'^\d{4}/\d{2}/\d{2}',
                    r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}',
                ]
                
                is_datetime = False
                for val in sample:
                    if any(re.match(pattern, str(val)) for pattern in datetime_patterns):
                        try:
                            pd.to_datetime(sample, errors='raise')
                            is_datetime = True
                            break
                        except (ValueError, TypeError):
                            pass
                
                if is_datetime:
                    features['datetime'].append(col)
                    continue

                unique_ratio = series.nunique() / len(series.dropna()) if len(series.dropna()) > 0 else 0
                unique_count = series.nunique()
                
                if unique_ratio > 0.5 or unique_count > 50:
                    features['text'].append(col)
                else:
                    features['categorical'].append(col)
            else:
                features['categorical'].append(col)
        
        return features


@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV file for downloading."""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV file for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def generate_config_hash(config):
    """Generates a stable hash for the configuration dictionary for cache invalidation."""
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return tuple(clean_dict(v) for v in d)
        else:
            return d
            
    # Use a copy of the config for cleaning to avoid modifying the session state
    cleaned_config = clean_dict(copy.deepcopy(config))
    config_str = json.dumps(cleaned_config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def validate_config(config, phase='ingestion'):
    """
    Validates the session state configuration based on the current workflow step.
    """
    errors = []
    if phase == 'ingestion':
        df = get_processed_df()
        if df is None or df.empty:
            errors.append("Please upload a dataset to begin.")
    
    if phase == 'config' or phase == 'training':
        if not config.get('target'):
            errors.append("Please select a 'Target Variable' in Step 4.")
            
        # Ensure target is not in features
        if config.get('target') in config.get('numerical_features', []) or \
           config.get('target') in config.get('categorical_features', []):
            errors.append(f"The target variable ('{config.get('target')}') cannot be selected as a feature.")
            
        if not config.get('numerical_features') and not config.get('categorical_features'):
            errors.append("Please select at least one feature (numerical or categorical) in Step 4.")
    
    if phase == 'training':
        if not config.get('selected_models'):
            errors.append("Please select at least one model to benchmark in Step 4.")
    
    if phase == 'ensemble':
        if not config.get('ensemble_base_models'):
             errors.append("Please select at least one base model for the ensemble.")

    # Validate numeric bounds
    if 'test_size' in config:
        test_size = config['test_size']
        if not (0.0 < test_size < 1.0):
            errors.append(f"Test size must be between 0 and 1 (currently: {test_size})")

    if 'cv_folds' in config:
        cv_folds = config['cv_folds']
        if cv_folds < 2:
            errors.append(f"CV folds must be at least 2 (currently: {cv_folds})")

    if 'bayes_iterations' in config:
        bayes_iter = config['bayes_iterations']
        if bayes_iter < 1:
            errors.append(f"Bayes iterations must be positive (currently: {bayes_iter})")
        if bayes_iter > 100:
            errors.append(f"Warning: {bayes_iter} iterations may take very long. Consider 50 or less.")
    return errors

def generate_fastapi_script(numerical_features, categorical_features, problem_type):
    """Generates a Python script for a FastAPI prediction endpoint."""
    def sanitize_identifier(name):
        """Sanitize a string to be a valid Python identifier."""
        # Replace invalid chars with '_'
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Prepend 'f_' if it starts with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = 'f_' + sanitized
        # Ensure it's not a reserved keyword (simplified check)
        if sanitized in ['class', 'def', 'import', 'from', 'if', 'else']:
            sanitized = sanitized + '_'
        return sanitized or 'feature'

    # Sanitize feature names for Python identifiers
    numerical_features_safe = [sanitize_identifier(f) for f in numerical_features]
    categorical_features_safe = [sanitize_identifier(f) for f in categorical_features]
    
    # Create Pydantic schema field definitions
    feature_schemas = ",\n    ".join(
        [f"{feat}: float = 0.0" for feat in numerical_features_safe] + 
        [f"{feat}: str = 'N/A'" for feat in categorical_features_safe]
    )

    # Map sanitized names back to original column names for DataFrame creation
    feature_mapping = dict(zip(numerical_features_safe + categorical_features_safe, numerical_features + categorical_features))
    df_creation_lines = ",\n        ".join([f"'{feature_mapping[feat]}': [data.{feat}]" for feat in numerical_features_safe + categorical_features_safe])

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
pipeline = None
try:
    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. API will not be able to make predictions.")
except Exception as e:
    print(f"An error occurred loading the model: {{e}}")


# 3. Define the input data schema
class PredictionFeatures(BaseModel):
    {feature_schemas}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: PredictionFeatures):
    if pipeline is None:
        return {{"error": "Model not loaded. Please ensure model.pkl is in the same directory."}}
    try:
        # Convert input data to DataFrame
        # IMPORTANT: Column names must match the feature names used during training!
        df = pd.DataFrame({{
            {df_creation_lines}
        }})

        # Make prediction
        prediction = pipeline.predict(df)
        prediction_value = prediction[0]

        # Convert numpy types to native Python types for JSON serialization
        if isinstance(prediction_value, (np.generic)):
            prediction_value = prediction_value.item()

        # For classification, return predicted class and potentially probability
        if "{problem_type}" == "Classification":
            response = {{"prediction": prediction_value}}
            # Try to get prediction probability if the model supports it
            try:
                if hasattr(pipeline, 'predict_proba'):
                    proba = pipeline.predict_proba(df)
                    # Assuming binary classification for simplicity (proba of class 1)
                    if proba.shape[1] == 2:
                        response["probability"] = proba[0, 1].item()
                    elif proba.shape[1] > 2:
                        # For multi-class, return all probabilities
                        response["probabilities"] = proba[0].tolist()
            except Exception:
                pass
            return response
        else:
            # For regression
            return {{"prediction": prediction_value}}
            
    except Exception as e:
        return {{"error": str(e), "details": "An error occurred during prediction. Check the input format."}}

@app.get("/")
def read_root():
    return {{"message": "Welcome to the Model Prediction API. Use the /predict endpoint for predictions."}}

# To run this script: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    return script

def generate_html_report(best_model_name):
    """Generates a self-contained HTML report of the results with robust error handling."""
    results_data = st.session_state.get('results_data')
    if not results_data or not isinstance(results_data, tuple) or len(results_data) != 6:
        return "<html><body><h1>Error: Results data unavailable.</h1></body></html>"

    results_df, le, _, _, X_test, y_test = results_data
    
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return "<html><body><h1>Error: No models were trained.</h1></body></html>"

    def fig_to_base64(fig):
        """Converts a Plotly figure to a base64 encoded PNG for static display."""
        if fig is None: return ""
        try:
            buf = BytesIO()
            # Must save as static image (e.g., png) for self-contained HTML
            fig.write_image(buf, format="png", engine="kaleido")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            # Fallback if kaleido is not installed (e.g. in some environments)
            logging.warning("Plotly image export requires 'kaleido'. Install it for full report.")
            return ""
        except Exception:
            return ""

    def fig_to_html(fig):
        """Converts a Plotly figure to an HTML div for interactive display."""
        if fig is None: return ""
        # Use full_html=False to embed the div, and include_plotlyjs='cdn' to load library from CDN
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    ai_summary = st.session_state.get('ai_summary', "<p>AI summary not generated.</p>")
    ai_summary_html = ai_summary.replace('\n', '<br>')

    metric_col = st.session_state.config.get('primary_metric_display', 'Score')
    results_df_sorted = results_df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    best_model_row_df = results_df_sorted[results_df_sorted['Model'] == best_model_name]
    if best_model_row_df.empty:
        return f"<html><body><h1>Error: Model '{best_model_name}' not found in results.</h1></body></html>"
    
    best_model_id = best_model_row_df['model_id'].iloc[0]
    best_model = st.session_state.model_store.get_model(best_model_id)

    if best_model is None:
        return f"<html><body><h1>Error: Model '{best_model_name}' could not be loaded.</h1></body></html>"

    # --- Generate Performance Plot (Confusion Matrix or Actual vs. Predicted) ---
    perf_fig = None
    try:
        y_pred = best_model.predict(X_test)
        if st.session_state.config.get('problem_type') == 'Classification':
            # Handle class labels being numbers or strings
            labels = np.unique(np.concatenate((y_test, y_pred)))
            
            # Use only integer labels for confusion matrix input
            cm_labels = [int(l) for l in labels if pd.api.types.is_numeric_dtype(l)]
            cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
            
            # Map numerical labels back to original class names if LabelEncoder was used
            class_names = le.inverse_transform(cm_labels) if le else [str(l) for l in cm_labels]
            
            perf_fig = px.imshow(
                cm, 
                text_auto=True, 
                labels=dict(x="Predicted", y="Actual"), 
                x=class_names, 
                y=class_names, 
                title="Confusion Matrix"
            )
        else:
            perf_fig = px.scatter(
                x=y_test, 
                y=y_pred, 
                labels={'x': 'Actual', 'y': 'Predicted'}, 
                title='Actual vs. Predicted', 
                trendline='ols'
            )
    except Exception as e:
        logging.error(f"Error generating performance plot: {e}")
        perf_fig = None

    # Get SHAP figure
    shap_fig = st.session_state.get('shap_fig')
    shap_fig_base64 = fig_to_base64(shap_fig)
    perf_fig_html = fig_to_html(perf_fig)
    
    shap_html_content = ""
    if shap_fig_base64:
        # Use a data URI for the static SHAP image
        shap_html_content = f"<img src='data:image/png;base64,{shap_fig_base64}' style='max-width:100%; height:auto; display:block; margin:auto;'>"
    else:
        # Check if the figure was generated but conversion failed
        if shap_fig is not None:
             shap_html_content = "<p>SHAP plot conversion to static image failed (is 'kaleido' installed?).</p>"
        else:
             shap_html_content = "<p>SHAP plot not generated or available.</p>"


    # Construct final HTML
    html = f"""
    <html>
        <head><title>MLOps Agent Report</title>
        <style>body{{font-family:sans-serif;margin:2em;}} h1,h2{{color:#0c4a6e;}} table{{border-collapse:collapse;width:100%;margin-bottom:2em;font-size:0.9em;}} th,td{{border:1px solid #ddd;padding:12px;text-align:left;}} th{{background-color:#f2f2f2;}} .container{{max-width:1200px;margin:auto;}} .section{{background-color:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:1.5em;margin-bottom:2em;}}</style>
        </head><body><div class="container">
        <h1>MLOps Agent Final Report</h1>
        <p><strong>Dataset:</strong> {st.session_state.file_name or "Sample Data"}</p>
        <p><strong>Problem Type:</strong> {st.session_state.config.get('problem_type', 'N/A')}</p>
        <p><strong>Best Model:</strong> <strong>{best_model_name}</strong></p>
        
        <div class="section">
            <h2>AI Executive Summary</h2>
            {ai_summary_html}
        </div>
        
        <div class="section">
            <h2>Performance Metrics on Test Set</h2>
            <p><strong>Primary Metric:</strong> {metric_col}</p>
            {results_df_sorted.drop(columns=['model_id']).rename(columns=lambda x: x.replace('_', ' ').title()).to_html(index=False)}
        </div>
        
        <div class="section">
            <h2>Performance Visualization</h2>
            {perf_fig_html}
        </div>
        
        <div class="section">
            <h2>Feature Importance (SHAP)</h2>
            {shap_html_content}
        </div>
        
        <div class="section">
            <h2>Configuration Summary</h2>
            <ul>
                <li><strong>Features Count:</strong> {len(st.session_state.config.get('numerical_features', [])) + len(st.session_state.config.get('categorical_features', []))}</li>
                <li><strong>Test Split:</strong> {st.session_state.config.get('test_size')}</li>
                <li><strong>CV Folds:</strong> {st.session_state.config.get('cv_folds')}</li>
                <li><strong>AI Features Used:</strong> {'Yes' if st.session_state.get('ai_features_approved') else 'No'}</li>
            </ul>
        </div>
        
        </div></body></html>"""
    return html

def save_experiment_run(config, results_df, model_id):
    """Save experiment to history for comparison."""
    if results_df is None or results_df.empty:
        return
    
    primary_metric_display = config.get('primary_metric_display', 'ROC AUC')
    best_row = results_df.sort_values(by=primary_metric_display, ascending=False).iloc[0]
    
    run_record = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_hash': generate_config_hash(config),
        'problem_type': config.get('problem_type', 'N/A'),
        'models': config.get('selected_models', []),
        'best_model': best_row['Model'],
        'best_score': best_row[primary_metric_display],
        'metric_name': primary_metric_display,
        'num_features': len(config.get('numerical_features', [])) + len(config.get('categorical_features', [])),
        'feature_list': config.get('numerical_features', []) + config.get('categorical_features', []),
        'test_size': config.get('test_size'),
        'cv_folds': config.get('cv_folds'),
        'bayes_iterations': config.get('bayes_iterations'),
        'ai_features_used': bool(st.session_state.get('ai_features_approved')),
        'full_config': config.copy(),
        # Drop model_id before saving results history
        'full_results': results_df.drop(columns='model_id', errors='ignore').to_dict('records')
    }
    
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
        
    st.session_state.experiment_history.append(run_record)
    
    # Keep history to a reasonable size
    if len(st.session_state.experiment_history) > 50:
        st.session_state.experiment_history = st.session_state.experiment_history[-50:]
        
    return run_record

def load_config_from_experiment(run_record):
    """Load a previous experiment's configuration."""
    if not run_record:
        return
    
    saved_config = run_record['full_config']
    
    # List of keys to safely load from the saved config
    keys_to_load = [
        'problem_type', 'selected_models', 'numerical_features', 
        'categorical_features', 'test_size', 'cv_folds', 
        'bayes_iterations', 'imputation_strategy', 'scaler',
        'handle_outliers', 'use_statistical_feature_selection',
        'imbalance_method', 'primary_metric_display', 'target',
        'k_features', 'groq_api_key', 'groq_model', 'groq_fallback_model'
    ]
    
    for key in keys_to_load:
        if key in saved_config:
            st.session_state.config[key] = saved_config[key]
            
    # Set the workflow step to configuration for review
    st.session_state.workflow_step = 4
            
    # BUG FIX - Resolved circular import by moving ui_components import into the local function scope.
    try:
        from .ui_components import sync_all_widgets
        sync_all_widgets()
    except ImportError:
        logging.error("Could not import ui_components for widget sync.")

    # UX ENHANCEMENT - Replaced emoji-based toasts with professional st.success messages.
    st.success(f"Loaded configuration from experiment on {run_record['timestamp']}")

def generate_deployment_package(best_model, config):
    """
    Generate a complete deployment package as a zip file in memory.
    Includes model, FastAPI app, Dockerfile, requirements, README, and metadata.
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Model file (model.pkl)
        model_bytes = pickle.dumps(best_model)
        zip_file.writestr('model.pkl', model_bytes)

        # 2. Requirements file (requirements.txt)
        requirements = """fastapi==0.111.0
uvicorn[standard]==0.30.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
pydantic
"""
        # Add model-specific libraries if they were used
        if any('XGBoost' in model for model in config.get('selected_models', [])):
            requirements += "xgboost==2.0.3\n"
        if any('LightGBM' in model for model in config.get('selected_models', [])):
            requirements += "lightgbm==4.3.0\n"
        zip_file.writestr('requirements.txt', requirements)

        # 3. FastAPI script (app.py)
        app_code = generate_fastapi_script(
            config.get('numerical_features', []),
            config.get('categorical_features', []),
            config.get('problem_type')
        )
        zip_file.writestr('app.py', app_code)

        # 4. Dockerfile
        dockerfile = """# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]"""
        zip_file.writestr('Dockerfile', dockerfile)

        # 5. README file (README.md)
        readme = f"""# ML Model Deployment Package

This package contains a trained machine learning model and all the necessary files to deploy it as a REST API.

## Quick Start

### 1. Running Locally (Without Docker)

First, install the required packages:
```bash
pip install -r requirements.txt
```
Then, run the FastAPI server:
```bash
uvicorn app:app --reload
```
The API will be available at http://127.0.0.1:8000.

### 2. Deploying with Docker

Build the Docker image:
```bash
docker build -t ml-model-api .
```
Run the Docker container:
```bash
docker run -d -p 8000:8000 ml-model-api
```
The API will be available at http://localhost:8000.

## Model Details

* **Problem Type:** {config.get('problem_type', 'N/A')}
* **Target Variable:** {config.get('target', 'N/A')}
* **Primary Metric:** {config.get('primary_metric_display', 'N/A')}
* **Features Used:** {len(config.get('numerical_features', [])) + len(config.get('categorical_features', []))}
* **Training Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
"""
        zip_file.writestr('README.md', readme)

        # 6. Model Metadata (model_metadata.json)
        metadata = {
            'model_type': config.get('problem_type'),
            'features': {
                'numerical': config.get('numerical_features', []),
                'categorical': config.get('categorical_features', [])
            },
            'training_date': pd.Timestamp.now().isoformat()
        }
        zip_file.writestr('model_metadata.json', json.dumps(metadata, indent=2))

        # 7. Config YAML for easier editing/reference
        config_yaml = f"""# Model Configuration
problem_type: {config.get('problem_type')}
target: {config.get('target')}

Features:
  numerical_features:
{chr(10).join(f'    - {f}' for f in config.get('numerical_features', []))}
  categorical_features:
{chr(10).join(f'    - {f}' for f in config.get('categorical_features', []))}

Preprocessing:
  test_size: {config.get('test_size')}
  imputation_strategy: {config.get('imputation_strategy')}
  scaler: {config.get('scaler')}
  handle_outliers: {config.get('handle_outliers')}
"""
        zip_file.writestr('config.yaml', config_yaml)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()






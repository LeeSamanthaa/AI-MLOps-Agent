### `modules/ui_components.py`
# ============================================================================
# FILE 3: modules/ui_components.py
# CHANGES: Added UI feedback for API usage (token counts, costs, warnings).
# Implemented a manual retry mechanism for failed API calls.
# Sanitized AI-generated text to prevent potential injection attacks.
# Ensured API key widget state is correctly managed.
# FIX: Replaced sync_all_widgets with a crash-proof version that safely handles state updates for active widgets.
# FIX: Removed blocking time.sleep() calls in the data cleaning panel for a better user experience.
# CRITICAL FIX: Added cache clearing for get_processed_df when loading new data.
# ============================================================================
# modules/ui_components.py
# UX ENHANCEMENT - Implemented a non-linear workflow allowing users to revisit steps.
# UX ENHANCEMENT - Added quick navigation with descriptive labels and automatic EDA refresh on feature changes.
# BUG FIX - Resolved circular import by moving initialize_session_state import to local scope.
# UX FIX - Removed redundant visual step header, leaving only the clean quick navigation buttons.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import re
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
import logging
import copy
import streamlit.components.v1 as components
import time
import os
import hashlib

# Internal module imports
from .llm_utils import (generate_llm_response, get_feature_engineering_prompt,
                        get_ai_config_prompt, get_rag_expert_context,
                        generate_dynamic_data_context, get_vif_analysis_prompt,
                        get_ensemble_guidance_prompt, get_eda_summary_prompt,
                        execute_data_transformation, parse_ai_recommendations)
from .pipeline import (run_pipeline, get_models_and_search_spaces, suggest_base_models,get_shap_values)
from .transformers import AIFeatureEngineer
from .utils import (
    load_data, get_sample_datasets, FeatureDetector, convert_df_to_csv,
    validate_config, generate_fastapi_script, get_default_config,
    get_processed_df, get_dataset_sample, generate_config_hash,
    generate_html_report, profile_data
)
from .chatbot_executor import ChatbotExecutor

# --- Library Availability Checks ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Centralized State & Widget Management ---

# FIX: Replaced the original sync_all_widgets function with a safe, two-part implementation.
def sync_widget_safe(key, value):
    """Safely sync a single widget's state without crashing if the widget is locked/active."""
    widget_key = f'widget_{key}'
    try:
        st.session_state[widget_key] = value
    except st.errors.StreamlitAPIException:
        # Fix - Add Deferred Sync Mechanism
        logging.warning(f"Could not sync widget '{widget_key}' as it is active. Deferring.")
        if 'deferred_widget_syncs' not in st.session_state:
            st.session_state.deferred_widget_syncs = {}
        st.session_state.deferred_widget_syncs[widget_key] = value
        pass

def sync_all_widgets():
    """
    Synchronizes all widget states with st.session_state.config using a crash-proof method.
    It iterates through a map of config keys and calls a safe sync function for each.
    """
    # Fix - Apply deferred syncs from previous run
    if 'deferred_widget_syncs' in st.session_state:
        for widget_key, value in st.session_state.deferred_widget_syncs.items():
            try:
                st.session_state[widget_key] = value
            except st.errors.StreamlitAPIException:
                pass
        st.session_state.deferred_widget_syncs = {}
        
    config = st.session_state.config

    sync_map = {
        'problem_type': config['problem_type'],
        'target': config.get('target'),
        'numerical_features': config['numerical_features'],
        'categorical_features': config['categorical_features'],
        'selected_models': config['selected_models'],
        'test_size': config['test_size'],
        'cv_folds': config['cv_folds'],
        'imputation_strategy': config['imputation_strategy'],
        'scaler': config['scaler'],
        'handle_outliers': config['handle_outliers'],
        'use_statistical_feature_selection': config['use_statistical_feature_selection'],
        'primary_metric_display': config['primary_metric_display'],
        'imbalance_method': config['imbalance_method'],
        'bayes_iterations': config['bayes_iterations'],
        'ensemble_base_models': config.get('ensemble_base_models', [])
    }

    # Safely sync each widget one by one
    for key, value in sync_map.items():
        sync_widget_safe(key, value)


def update_config(key, widget_key=None):
    """
    Callback to update a key in st.session_state.config.
    """
    widget_key = widget_key or f"widget_{key}"
    if widget_key in st.session_state:
        st.session_state.config[key] = st.session_state[widget_key]

# --- Enhanced Callbacks ---

def handle_problem_type_change():
    """Callback for when the problem type changes to reset dependent configs."""
    update_config('problem_type')
    st.session_state.config['selected_models'] = []
    st.session_state.config['ensemble_base_models'] = []

    if st.session_state.config['problem_type'] == 'Classification':
        st.session_state.config['primary_metric_display'] = 'ROC AUC'
    else:
        st.session_state.config['primary_metric_display'] = 'R2 Score'

    st.session_state.phase1_complete = False
    st.session_state.phase1_results = None
    sync_all_widgets()

def handle_target_change():
    """Auto-infers problem type based on target column characteristics."""
    update_config('target')
    df = get_processed_df()
    target = st.session_state.config.get('target')

    if df is None or target is None or target not in df.columns:
        return

    is_numeric = pd.api.types.is_numeric_dtype(df[target])
    unique_values = df[target].nunique()
    inferred_problem_type = "Regression" if is_numeric and unique_values > 25 else "Classification"

    if st.session_state.config['problem_type'] != inferred_problem_type:
        st.session_state.config['problem_type'] = inferred_problem_type
        st.info(f"Detected '{target}' as a {inferred_problem_type.lower()} target. Problem type updated.")
        handle_problem_type_change() # This will sync widgets
    else:
        sync_all_widgets()


def handle_reset_configuration():
    """Callback to fully reset the application state to its default."""
    from .utils import initialize_session_state
    api_key = st.session_state.config.get('groq_api_key', '')
    initialize_session_state(force=True)
    st.session_state.config['groq_api_key'] = api_key
    st.session_state.view = 'configuration'
    st.success("Project has been reset.")


# --- Handlers for User Interactions ---

def handle_auto_config(df):
    """Handles AI-powered automatic configuration of the pipeline."""
    if df is None:
        st.warning("Please load data before using AI Auto-Configuration.")
        return
    api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar.")
        return

    with st.spinner("AI is analyzing dataset..."):
        prompt = get_ai_config_prompt(df)
        response = generate_llm_response(prompt, api_key, is_json=True)

        if not response:
            st.error("AI configuration failed. No response from the agent.")
            return

        try:
            ai_config = json.loads(response)
            problem_type = ai_config.get('problem_type', 'Classification')
            
            st.session_state.config['problem_type'] = problem_type
            st.session_state.config['target'] = ai_config.get('target_column')
            st.session_state.config['numerical_features'] = ai_config.get('numerical_features', [])
            st.session_state.config['categorical_features'] = ai_config.get('categorical_features', [])
            st.session_state.config['primary_metric_display'] = 'ROC AUC' if problem_type == 'Classification' else 'R2 Score'
            all_models_list = [m for m in get_models_and_search_spaces(problem_type).keys() if 'Voting' not in m]
            st.session_state.config['selected_models'] = all_models_list
            
            confidence = ai_config.get('confidence_score', 'N/A')
            rationale = ai_config.get('rationale', 'No rationale provided.')
            # Sanitize rationale to prevent HTML/script injection
            safe_rationale = rationale.replace('<', '&lt;').replace('>', '&gt;')
            st.session_state.ai_config_rationale = f"**Confidence: {confidence}/100**\n\n{safe_rationale}"
            
            sync_all_widgets()
            st.success("AI configuration applied successfully.")
            
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"AI configuration failed. Could not parse the AI's response: {e}")
            logging.error(f"AI config parsing failed. Response was: {response}", exc_info=True)


def handle_feature_recommendations(df):
    """Handles AI-powered feature engineering recommendations."""
    if df is None:
        st.warning("Please load data before getting AI recommendations.")
        return
    # CRITICAL: Clear previous recommendations to get fresh ones
    st.session_state.feature_recommendations = []
    api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar.")
        return
        
    with st.spinner("AI is analyzing data for feature ideas..."):
        target = st.session_state.config.get('target')
        if not target:
            st.warning("Please select a target variable before getting recommendations.")
            return
            
        feats = FeatureDetector.detect_features(df, target)
        prompt = get_feature_engineering_prompt(
            {'shape': df.shape, 'description': df.describe().to_string()},
            target, feats['numerical'], feats['categorical'])
        recs = generate_llm_response(prompt, api_key, is_json=True)
        
        if not recs: return
        try:
            st.session_state.feature_recommendations = json.loads(recs).get("recommendations", [])
            st.info("AI has generated new feature recommendations.")
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI recommendations. Error: {e}")
            logging.error(f"AI feature rec parsing failed. Response was: {recs}", exc_info=True)



def handle_approve_features():
    """
    Approves and applies AI-generated features with atomic state updates.
    If any step fails, rolls back to previous state.
    """
    # Store original state for rollback
    original_df = st.session_state.df.copy()
    original_config = copy.deepcopy(st.session_state.config)
    
    try:
        # Step 1: Apply features to base DataFrame
        if not st.session_state.get('feature_recommendations'):
            st.warning("No feature recommendations to apply.")
            return
        
        # Clear cache before processing
        get_processed_df.clear()
        
        # Get current processed df to identify new columns
        # This call will apply features in 'preview mode' and log their creation once.
        processed_df = get_processed_df()
        if processed_df is None:
            st.error("Could not apply features because the dataset is not available.")
            return

        base_cols = set(original_df.columns)
        new_cols = [col for col in processed_df.columns if col not in base_cols]
        
        if not new_cols:
            st.warning("No new features were created.")
            return
        
        # Step 2: Permanently apply AI features to base df
        # FIX: The 'processed_df' variable already contains the transformed data
        # from the get_processed_df() call above. We just assign it.
        # This prevents the AIFeatureEngineer from running a second time.
        st.session_state.df = processed_df
        logging.info(f"Applied AI features to base DataFrame. New columns: {new_cols}")
        
        # Step 3: Mark features as approved
        st.session_state.ai_features_approved = True
        
        # Step 4: Clear recommendations (they're now permanent)
        applied_recommendations = st.session_state.feature_recommendations.copy()
        st.session_state.feature_recommendations = []
        
        # Step 5: Re-fetch processed df
        get_processed_df.clear()
        # This call will now use the approved features in st.session_state.df
        processed_df = get_processed_df()
        
        # Step 6: Force numeric dtype for new columns and detect types
        for col in new_cols:
            if col not in processed_df.columns:
                continue
                
            # Try to convert to numeric if stored as object
            if processed_df[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(processed_df[col], errors='coerce')
                    if converted.notna().sum() / len(converted) > 0.8:
                        st.session_state.df[col] = converted
                        processed_df[col] = converted
                        logging.info(f"Converted AI feature '{col}' to numeric dtype")
                except Exception as e:
                    logging.warning(f"Could not convert {col} to numeric: {e}")
        
        # Step 7: Run type detection on full DataFrame
        target = st.session_state.config.get('target')
        new_feature_types = FeatureDetector.detect_features(processed_df, target=target)
        
        # Step 8: Filter to only NEW columns
        detected_numerical = [col for col in new_feature_types.get('numerical', []) if col in new_cols]
        detected_categorical = [col for col in new_feature_types.get('categorical', []) if col in new_cols]
        for col in new_cols:
            if '_x_' in col or '_pow' in col or '_log' in col:
                # These should ALWAYS be numerical
                if col in detected_categorical:
                    detected_categorical.remove(col)
                    logging.warning(f"Forcing '{col}' from categorical to numerical (interaction/polynomial feature)")
                if col not in detected_numerical:
                    detected_numerical.append(col)

        logging.info(f"New features detected - Numerical: {detected_numerical}, Categorical: {detected_categorical}")
        
        # Step 9: Update config with new features
        num_features_set = set(st.session_state.config.get('numerical_features', []))
        num_features_set.update(detected_numerical)
        st.session_state.config['numerical_features'] = sorted(list(num_features_set))
        
        cat_features_set = set(st.session_state.config.get('categorical_features', []))
        cat_features_set.update(detected_categorical)
        st.session_state.config['categorical_features'] = sorted(list(cat_features_set))

        # Step 10: Sync widgets
        sync_all_widgets()
        
        # Step 11: Clear cache one final time
        get_processed_df.clear()
        
        # Success!
        st.success(f"✓ AI features applied: {len(detected_numerical)} numerical, {len(detected_categorical)} categorical")
        st.session_state.show_eda = True
        st.info("EDA has been updated. Review changes in Step 2.")
        
    except Exception as e:
        # Rollback on any error
        st.session_state.df = original_df
        st.session_state.config = original_config
        st.session_state.ai_features_approved = False
        get_processed_df.clear()
        
        error_msg = f"Failed to apply AI features: {e}"
        st.error(error_msg)
        logging.error(f"Feature approval failed with rollback: {e}", exc_info=True)

def apply_ai_recommendations():
    """
    Applies AI-driven configuration changes with comprehensive action support.
    Handles all recommendation types from the enhanced parser.
    """
    from .pipeline import get_models_and_search_spaces
    
    msgs = []
    parsed_actions = st.session_state.get('parsed_actions', [])
    
    if not parsed_actions:
        st.warning("No parsed recommendations found to apply.")
        return
    
    problem_type = st.session_state.config.get('problem_type')
    valid_models = get_models_and_search_spaces(problem_type).keys()
    
    for action in parsed_actions:
        action_type = action.get('type')
        
        try:
            if action_type == 'increase_tuning':
                # Allow custom value or default increment
                new_val = action.get('param_value')
                if new_val:
                    new_iter = min(100, int(new_val))
                else:
                    new_iter = min(100, st.session_state.config.get('bayes_iterations', 25) + 15)
                st.session_state.config['bayes_iterations'] = new_iter
                msgs.append(f"Increased **Tuning Iterations** to {new_iter}.")
                
            elif action_type == 'try_smote':
                if problem_type == 'Classification':
                    st.session_state.config['imbalance_method'] = 'SMOTE'
                    msgs.append("Enabled **SMOTE** for imbalanced data handling.")
                else:
                    msgs.append("⚠️ SMOTE only applies to classification problems.")
                    
            elif action_type == 'try_feature_selection':
                st.session_state.config['use_statistical_feature_selection'] = True
                msgs.append("Enabled **Statistical Feature Selection**.")
                
            elif action_type == 'add_model':
                model_name = action.get('model_name')
                current_models = st.session_state.config.get('selected_models', [])
                
                # Fuzzy matching for model names
                matched_model = None
                if model_name:
                    model_name_lower = model_name.lower()
                    for valid_model in valid_models:
                        if model_name_lower in valid_model.lower() or valid_model.lower() in model_name_lower:
                            matched_model = valid_model
                            break
                
                if matched_model and matched_model not in current_models and 'Voting' not in matched_model:
                    current_models.append(matched_model)
                    st.session_state.config['selected_models'] = current_models
                    msgs.append(f"Added **{matched_model}** to model selection.")
                elif matched_model in current_models:
                    msgs.append(f"⚠️ {matched_model} is already selected.")
                elif 'Voting' in str(matched_model):
                    msgs.append(f"⚠️ Voting ensembles are created separately in Step 5.")
                else:
                    msgs.append(f"⚠️ Could not find model '{model_name}' for {problem_type}.")
                    
            elif action_type == 'enable_regularization':
                model_name = action.get('model_name')
                param_name = action.get('param_name')
                param_value = action.get('param_value')
                
                if model_name and param_name and param_value is not None:
                    if 'manual_params' not in st.session_state.config:
                        st.session_state.config['manual_params'] = {}
                    if model_name not in st.session_state.config['manual_params']:
                        st.session_state.config['manual_params'][model_name] = {}
                    
                    st.session_state.config['manual_params'][model_name][param_name] = param_value
                    st.session_state.config['manual_tuning_enabled'] = True
                    msgs.append(f"Set **{param_name}={param_value}** for {model_name}.")
                else:
                    # FIX: Apply sensible defaults if parameters are missing
                    applied_default = False
                    models_to_regularize = []
                    
                    if model_name:
                        models_to_regularize.append(model_name)
                    else:
                        # If no model specified, apply to all compatible models in selection
                        for m in st.session_state.config.get('selected_models', []):
                            if 'Logistic' in m or 'XGBoost' in m or 'LightGBM' in m:
                                models_to_regularize.append(m)
                                
                    if not models_to_regularize:
                        msgs.append("⚠️ Regularization recommended, but no compatible models (e.g., Logistic, XGB, LGBM) are selected.")
                        continue

                    if 'manual_params' not in st.session_state.config:
                        st.session_state.config['manual_params'] = {}
                    
                    for m in models_to_regularize:
                        if m not in st.session_state.config['manual_params']:
                            st.session_state.config['manual_params'][m] = {}
                        
                        # Apply default regularization
                        if 'Logistic' in m:
                            st.session_state.config['manual_params'][m]['C'] = 0.1 # Default 'C'
                            msgs.append(f"Applied default regularization (**C=0.1**) to **{m}**.")
                            applied_default = True
                        elif 'XGBoost' in m:
                            st.session_state.config['manual_params'][m]['reg_alpha'] = 1.0 # Default 'reg_alpha'
                            msgs.append(f"Applied default regularization (**reg_alpha=1.0**) to **{m}**.")
                            applied_default = True
                        elif 'LightGBM' in m:
                            st.session_state.config['manual_params'][m]['reg_lambda'] = 1.0 # Default 'reg_lambda'
                            msgs.append(f"Applied default regularization (**reg_lambda=1.0**) to **{m}**.")
                            applied_default = True

                    if applied_default:
                        st.session_state.config['manual_tuning_enabled'] = True
                    else:
                        msgs.append("⚠️ Regularization recommended, but could not apply defaults.")
                    
            elif action_type == 'create_polynomial_features':
                degree = action.get('degree', 2)
                num_features = st.session_state.config.get('numerical_features', [])
                
                if not num_features:
                    msgs.append("⚠️ No numerical features available for polynomial expansion.")
                else:
                    # Create polynomial recommendations for top features
                    top_features = num_features[:3]  # Take top 3 to avoid explosion
                    for feat in top_features:
                        rec = {
                            "technique": "polynomial",
                            "column": feat,
                            "degree": degree,
                            "confidence_score": 75,
                            "rationale_long": f"Creating degree-{degree} polynomial features for {feat} to capture non-linear relationships."
                        }
                        if rec not in st.session_state.feature_recommendations:
                            st.session_state.feature_recommendations.append(rec)
                    
                    msgs.append(f"Added **polynomial (degree {degree})** feature recommendations. Review in Step 3.")
                    
            elif action_type == 'try_ensemble_strategy':
                strategy = action.get('strategy', 'voting').lower()
                
                if strategy == 'voting':
                    msgs.append("**Voting Ensemble**: Create in Step 5 after benchmark completes.")
                elif strategy in ['stacking', 'bagging', 'blending']:
                    msgs.append(f"**{strategy.capitalize()} Ensemble**: This requires custom implementation. Consider voting ensemble as an alternative.")
                else:
                    msgs.append(f"Unknown ensemble strategy: {strategy}")
                    
            elif action_type == 'investigate_interactions':
                num_features = st.session_state.config.get('numerical_features', [])
                
                if len(num_features) < 2:
                    msgs.append("Need at least 2 numerical features for interactions.")
                else:
                    # Create interaction recommendations for top feature pairs
                    pairs_to_try = [
                        (num_features[0], num_features[1]),
                        (num_features[0], num_features[2]) if len(num_features) > 2 else None
                    ]
                    
                    for pair in pairs_to_try:
                        if pair:
                            rec = {
                                "technique": "interaction",
                                "column_1": pair[0],
                                "column_2": pair[1],
                                "confidence_score": 70,
                                "rationale_long": f"Testing interaction between {pair[0]} and {pair[1]} to capture combined effects."
                            }
                            if rec not in st.session_state.feature_recommendations:
                                st.session_state.feature_recommendations.append(rec)
                    
                    msgs.append("Added **interaction feature** recommendations. Review in Step 3.")
                    
            elif action_type == 'adjust_test_size':
                new_size = action.get('new_size', 0.25)
                if 0.01 < new_size < 0.99:
                    st.session_state.config['test_size'] = new_size
                    msgs.append(f"Adjusted **test size** to {new_size:.0%}.")
                else:
                    msgs.append(f"Invalid test size: {new_size}")
                    
            elif action_type == 'increase_cv_folds':
                new_folds = action.get('new_folds', 10)
                if 2 <= new_folds <= 20:
                    st.session_state.config['cv_folds'] = new_folds
                    msgs.append(f"Increased **CV folds** to {new_folds}.")
                else:
                    msgs.append(f" CV folds must be between 2 and 20 (got {new_folds}).")
                    
            else:
                msgs.append(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logging.error(f"Error applying recommendation {action_type}: {e}", exc_info=True)
            msgs.append(f"Failed to apply: {action.get('action_text', 'Unknown')}")
    
    # Safe sync with crash protection
    sync_all_widgets()
    
    # Display results
    if msgs:
        st.session_state.guide_message = "**AI Recommendations Applied:**\n\n" + "\n".join([f"- {m}" for m in msgs])
    else:
        st.session_state.guide_message = "No recommendations could be applied. Please check the logs."
    
    # Navigate back to configuration
    st.session_state.view = 'configuration'
    st.session_state.workflow_step = 4
    st.rerun()

# --- Main Page Display Functions ---

def display_welcome_page():
    """Renders comprehensive welcome page with detailed workflow explanation."""
    st.title("AI MLOps Agent")
    st.markdown("""
    <div style='text-align: center; font-size: 1.3em; margin-bottom: 30px;'>
        <strong>From Raw Data to Deployed Model in Minutes</strong><br>
        <span style='font-size: 0.85em; color: #666;'>Enterprise-grade ML automation with AI-powered insights</span>
    </div>
    """, unsafe_allow_html=True)

    # === SECTION 1: How It Works ===
    st.header("Complete Workflow Overview", divider='blue')

    workflow_tabs = st.tabs(["Data Ingestion", "Data Cleaning", "EDA & Features", "Training", "Evaluation", "Deployment"])

    with workflow_tabs[0]:
        st.subheader("Step 1: Data Ingestion & Quality Assessment")
        st.markdown("""
        **What Happens:**
        1. **Upload your data** (CSV, Excel, JSON, Parquet up to 500MB)
        2. **AI analyzes the dataset** and generates:
           - Data quality score (0-100)
           - Missing value report
           - Duplicate detection
           - Column type inference (numeric, categorical, datetime, text, ID)
        
        **Automatic AI Configuration:**
        - Click "Auto-Configure" and the AI will:
          - Predict problem type (Classification vs Regression)
          - Identify the target variable
          - Select appropriate features
          - Provide confidence score and detailed rationale
        
        **Example:**
        ```
        Dataset: customer_churn.csv (10,000 rows × 15 columns)
        Quality Score: 87/100
        AI Detected: Classification problem
        Target: "Churn" (binary: 0/1)
        Confidence: 92%
        Rationale: "The 'Churn' column has two unique values (0, 1) representing a binary outcome..."
        ```
        """)

    with workflow_tabs[1]:
        st.subheader("Step 2: Advanced Data Cleaning")
        st.markdown("""
        **Two Ways to Clean Your Data:**
        
        ### Visual Cleaning Tools (Widgets)
        Interactive panels for:
        - **Type Conversion**: Convert strings to datetime, numbers to categories
        - **Value Replacement**: Map "Yes"→1, "No"→0, etc.
        - **Column Operations**: Rename, drop, reorder columns
        - **Row Filtering**: Keep rows where Salary > 50000
        - **Outlier Removal**: IQR method, Z-score, percentile-based
        
        **Live Preview:** See changes before applying!
        
        ### Conversational Cleaning (Chatbot)
        Give natural language commands:
        - "Drop the CustomerID column"
        - "Fill missing Age values with median"
        - "Convert OrderDate to datetime format"
        - "Keep only rows where Country is 'USA'"
        - "Remove outliers from the Price column using IQR method"
        
        **How It Works:**
        1. You type a command in plain English
        2. AI generates safe Python/pandas code
        3. Code is validated (no file I/O, no system calls)
        4. You see the exact code before execution
        5. Dataset updates instantly
        
        **Example Transformation History:**
        ```python
        Query: "Fill missing values in Age with median"
        Code: df_modified = df.copy()
              df_modified['Age'].fillna(df['Age'].median(), inplace=True)
        Result: ✅ 247 missing values filled
        ```
        """)

    with workflow_tabs[2]:
        st.subheader("Step 3: EDA & AI Feature Engineering")
        st.markdown("""
        **Automated Exploratory Data Analysis:**
        - Distribution plots for numerical features
        - Interactive correlation heatmap
        - Categorical value counts with target breakdown
        - **VIF Analysis** for multicollinearity detection
            - AI explains high VIF scores
            - Suggests which features to drop
            - One-click application
        
        **AI Feature Recommendations:**
        - Click "Get AI Feature Recommendations" to receive:
          - **Polynomial features** (Age², Age³)
          - **Interaction features** (Age × Income)
          - **Binning suggestions** (Age → Age_Group)
          - **Log transformations** for skewed data
        - Each recommendation includes:
          - **Confidence score** (0-100%)
          - **Detailed rationale** explaining why this feature helps
          - One-click approval to create and apply all features
        
        **Manual Feature Creation:**
        - Create your own:
          - **Interaction:** Multiply two features
          - **Polynomial:** Square or cube a feature
          - Custom formulas via chatbot
          
        **Example AI Recommendation:**
        ```
        Technique: Interaction
        Columns: Age × Balance
        Confidence: 88%
        Rationale: "Credit scoring models often benefit from interactions between age and financial metrics. Older customers with high balances represent a distinct risk profile..."
        ```
        """)

    with workflow_tabs[3]:
        st.subheader("Step 4: Model Training & Hyperparameter Tuning")
        st.markdown("""
        **Model Selection:**
        - **Classification:** Logistic Regression, Random Forest, XGBoost, LightGBM, Balanced RF
        - **Regression:** Linear Regression, Random Forest, XGBoost, LightGBM
        - **Ensemble:** Voting Classifier/Regressor (combines best models)
        
        **Automated Preprocessing Pipeline:**
        - **Imputation:** Fill missing values (median, mean, most_frequent)
        - **Outlier Handling:** Cap extreme values using IQR
        - **Scaling:** StandardScaler or MinMaxScaler
        - **Encoding:** One-Hot encoding for categorical features
        - **Feature Selection** (optional): Select top K features using statistical tests
        
        **Hyperparameter Optimization:**
        - **Automatic (Bayesian):** Intelligent search finds optimal parameters
            - *Example: For Random Forest, searches n_estimators (100-500), max_depth (10-100)*
        - **Manual Mode:** Set exact values yourself
            - L1/L2 regularization controls (C, alpha, reg_lambda)
            - Tree depth limits
            - Learning rates
            
        **Advanced Options:**
        - **SMOTE:** Handle imbalanced datasets (fraud detection, rare diseases)
        - **Cross-Validation:** 2-10 folds for robust evaluation
        - **Quick Test Mode:** Reduced iterations for fast experimentation
        
        **Training Output:**
        - Progress bar with real-time updates
        - Parallel processing for multiple models
        - All metrics logged to MLflow (optional)
        """)
        
    with workflow_tabs[4]:
        st.subheader("Step 5: Model Evaluation & Insights")
        st.markdown("""
        **Performance Dashboard:**
        - Sortable table with all metrics (Accuracy, Precision, Recall, F1, ROC AUC, R²)
        - Color highlighting for best scores
        - Comparison across all trained models
        
        **AI Executive Summary:**
        - Natural language explanation of results
        - Interpretation of best model's performance
        - 2-3 actionable recommendations for next run
            - "Try SMOTE for imbalanced data"
            - "Increase tuning iterations to 50"
            - "Add XGBoost model to selection"
        
        **Explainability Tools:**
        - **SHAP Values:** See which features drive each prediction
        - **Partial Dependence Plots:** How changing one feature affects output
        - **Confusion Matrix** (classification): See misclassification patterns
        - **Actual vs Predicted Plot** (regression): Visual fit quality
        
        **Data Drift Monitoring:**
        - Upload new production data
        - Kolmogorov-Smirnov test compares distributions
        - Alerts when model needs retraining
        
        **Example AI Summary:**
        ```
        Your Random Forest model achieved 94.2% accuracy on the test set. This is excellent performance for a customer churn prediction task. The model shows balanced precision (92%) and recall (91%), meaning it catches most churners without too many false alarms.
        
        Recommendations for next run:
        1. Try adding an ensemble (Voting Classifier) combining Random Forest and XGBoost
        2. Increase Bayesian tuning iterations from 25 to 50 for marginal gains
        3. Consider SMOTE since your dataset has 85% non-churners vs 15% churners
        ```
        """)

    with workflow_tabs[5]:
        st.subheader("Step 6: One-Click Deployment")
        st.markdown("""
        **Deployment Package Includes:**
        - `model.pkl`: Trained scikit-learn pipeline
        - `app.py`: Production-ready FastAPI server
        - `requirements.txt`: All dependencies
        - `Dockerfile`: For containerized deployment
        - `README.md`: Setup instructions
        - `model_metadata.json`: Feature list, problem type, training date
        
        **Deployment Options:**
        
        A) **Local Testing:**
        ```bash
        pip install -r requirements.txt
        uvicorn app:app --reload
        # API available at http://localhost:8000
        ```
        
        B) **Docker Deployment:**
        ```bash
        docker build -t ml-api .
        docker run -p 8000:8000 ml-api
        ```
        
        C) **Cloud Deployment:**
        - **AWS:** Elastic Beanstalk, ECS, Lambda
        - **Google Cloud:** Cloud Run, App Engine
        - **Azure:** App Service, Container Instances
        
        **API Endpoint:**
        ```python
        POST /predict
        {
          "Age": 35,
          "Income": 75000,
          "CreditScore": 720,
          "Geography": "France"
        }
        
        Response:
        {
          "prediction": 0  # 0 = won't churn, 1 = will churn
        }
        ```
        
        **Bonus: HTML Report**
        - Self-contained report with all visualizations
        - AI summary embedded
        - SHAP plots included as base64 images
        - Share with stakeholders who don't use Python
        """)

    st.markdown("---")
    
    # === SECTION 2: Key Features ===
    st.header("Key Features", divider='violet')
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("""
        **AI-Powered Automation**
        - Auto-detect problem type & target
        - Smart feature recommendations
        - Conversational data cleaning
        - Results interpretation
        - Proactive improvement suggestions
        """)
    with feature_cols[1]:
        st.markdown("""
        **Enterprise ML Capabilities**
        - 10+ production-ready models
        - Bayesian hyperparameter tuning
        - SHAP explainability
        - Data drift monitoring
        - MLflow experiment tracking
        """)
    with feature_cols[2]:
        st.markdown("""
        **Production-Ready Output**
        - FastAPI deployment code
        - Docker containerization
        - Complete package with docs
        - HTML reports for stakeholders
        - Model versioning & metadata
        """)
        
    st.markdown("---")
    
    # === SECTION 3: Get Started ===
    st.header("Get Started", divider='green')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Your Data")
        file_types = ["csv", "xlsx", "xls", "json", "parquet"]
        if uploaded_file := st.file_uploader(
            f"Supported: {', '.join(file_types)} (max 500MB)",
            type=file_types
        ):
            st.session_state.df = load_data(uploaded_file)
            if st.session_state.df is not None:
                st.session_state.data_loaded = True
                st.session_state.view = 'configuration'
                
                # CRITICAL FIX: Clear cached processed dataframe
                get_processed_df.clear()
                
                # Clear AI features state when loading new data
                st.session_state.ai_features_approved = False
                st.session_state.feature_recommendations = []
                
                st.rerun()
    with col2:
        st.subheader("Or Try a Sample Dataset")
        sample_datasets = get_sample_datasets()
        dataset_choice = st.selectbox("Choose Sample:", list(sample_datasets.keys()))
        
        if st.button(f"Load '{dataset_choice}'", type="primary", width='stretch'):
            st.session_state.df = sample_datasets[dataset_choice]
            problem_type = "Classification" if "Classification" in dataset_choice else "Regression"
            st.session_state.config['problem_type'] = problem_type
            st.session_state.config['primary_metric_display'] = 'ROC AUC' if problem_type == "Classification" else 'R2 Score'
            st.session_state.data_loaded = True
            st.session_state.view = 'configuration'
            
            # CRITICAL FIX: Clear cached processed dataframe
            get_processed_df.clear()
            
            # Clear AI features state when loading new data
            st.session_state.ai_features_approved = False
            st.session_state.feature_recommendations = []
            
            st.rerun()

    st.markdown("---")
    st.info("""
    **Pro Tips:**
    - Start with "Auto-Configure" to let AI set up everything
    - Use the chatbot for quick data cleaning ("drop column X")
    - Always review the EDA before training
    - Try the Quick Test mode first (faster iterations)
    - Download the complete deployment package, not just the model
    """)
    st.markdown("---")
    # Fix - Insert professional contact section
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #666; font-size: 0.9em;'>
        We welcome your feedback and questions. Please contact us at: 
        <a href='mailto:samantha.dataworks@gmail.com' style='color: #0066cc; text-decoration: none;'>
            samantha.dataworks@gmail.com
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Master Sidebar for Navigation and Status ---
def display_master_sidebar():
    """
    Renders the persistent sidebar, including workflow navigation, status, quick actions,
    and the AI Co-Pilot chat.
    """
    with st.sidebar:
        st.title("AI MLOps Agent")
        st.markdown("---")

        # --- Workflow Navigation & Status ---
        st.header("Workflow Status")
        steps = {
            'ingestion': "1. Ingestion & Quality",
            'eda': "2. EDA",
            'feature_eng': "3. Feature Engineering",
            'config': "4. Model Configuration",
            'training': "5. Training & Validation",
            'evaluation': "6. Evaluation"
        }
        for i, (key, name) in enumerate(steps.items()):
            is_current = st.session_state.workflow_step == (i + 1)
            
            if is_current:
                st.markdown(f"**→ {name}**")
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;{name}")
        
        st.markdown("---")

        # --- Data Status ---
        st.header("Data Status")
        if st.session_state.get('data_loaded', False) and st.session_state.df is not None:
            df = get_processed_df()
            if df is not None:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
            else:
                st.info("Processing data...")
        else:
            st.info("No data loaded.")

        st.markdown("---")
        
        # --- Quick Actions ---
        st.header("Quick Actions")
        if st.button("Reset Project", width='stretch', on_click=handle_reset_configuration):
            st.rerun()
        if st.button("View Experiments", width='stretch'):
            st.session_state.view = 'experiments'
            st.rerun()

        st.markdown("---")

        # --- AI Co-Pilot Chat with Extended Capabilities ---
        st.title("AI Co-Pilot")
        st.write("Ask questions OR give commands to clean/modify data.")
        
        # Initialize the executor once and store it in session state
        if 'chatbot_executor' not in st.session_state:
            st.session_state.chatbot_executor = ChatbotExecutor()
        executor = st.session_state.chatbot_executor

        api_key_default = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))

        st.text_input(
            "Groq API Key",
            type="password",
            value=api_key_default,
            key='widget_groq_api_key',
            on_change=update_config,
            args=('groq_api_key',),
            help="Your API key for the Groq service, used for all AI-powered features. Can also be set via GROQ_API_KEY environment variable."
        )
        # Display token usage and warning if available
        if st.session_state.get('llm_token_usage'):
            usage = st.session_state.llm_token_usage
            st.caption(f"Session Tokens: {usage['total_tokens']:,} | Est. Cost: ${usage['estimated_cost']:.4f}")
            if usage['total_tokens'] > 50000:
                st.warning("High token usage. Please monitor your costs.")
        
        # Display last API error and provide a retry mechanism
        if st.session_state.get('last_api_error'):
            with st.expander("Last API Error", expanded=True):
                st.error(st.session_state.last_api_error)
                if st.button("Clear Error and Retry"):
                    st.session_state.last_api_error = None
                    st.rerun()

        # Show chatbot capabilities
        with st.expander("What can the chatbot do?"):
            st.markdown("""
            **Configuration Commands:**
            - `set problem type to regression`
            - `set target to Price`
            - `add models XGBoost and RandomForest`
            - `set test size to 25%`
            - `enable smote`
            - `set tuning iterations to 40`
            
            **Data Cleaning Commands:**
            - `drop column CustomerID`
            - `fill missing values in Age with median`
            - `rename column OldName to NewName`

            **Workflow & Guidance:**
            - `show current status`
            - `what should I do next?`
            - `am I ready to train?`
            
            **Questions:**
            - "What does high VIF mean?"
            - "Explain the difference between precision and recall"
            """)

        st.markdown("---")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question or give a command..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Agent is thinking..."):
                    api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))
                    if not api_key:
                        response = "Please provide a Groq API key to use the Co-Pilot."
                        message_placeholder.error(response)
                    else:
                        # First, try to execute the prompt as a command
                        is_command, response = executor.parse_and_execute(prompt)
                        
                        if not is_command:
                            # If it's not a command, fall back to RAG for Q&A
                            results_context = "No model run yet."
                            if st.session_state.get('results_data'):
                                results_df = st.session_state.results_data[0]
                                if results_df is not None and not results_df.empty:
                                    results_context = f"CURRENT MODEL RESULTS:\n{results_df.to_string()}"
                            
                            data_context = generate_dynamic_data_context(
                                get_processed_df(), 
                                st.session_state.config, 
                                st.session_state.eda_summary
                            )
                            
                            # Add current status to the RAG context
                            status_summary = executor.get_status_summary()

                            rag_prompt = f"""You are an AI assistant helping a user build an ML pipeline.
{status_summary}

Context:
- Expert Knowledge: {get_rag_expert_context()}
- Dataset & EDA: {data_context}
- Experiment Results: {results_context}

User's Question: {prompt}

Based on the user's question and the current pipeline status, provide a clear, helpful, and actionable response.
If the user asks what to do next, analyze the status and give a specific recommendation.
"""
                            
                            response = generate_llm_response(rag_prompt, api_key)
                            if not response:
                                response = "The agent could not respond. Please check your API key and try again."
                                
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})


def display_workflow_step_header():
    """Displays a clean quick navigation bar for the MLOps workflow."""
    steps = ["Ingestion", "EDA", "Feature Eng.", "Configuration", "Training", "Evaluation"]
    
    st.markdown("**Quick Navigation:**")
    cols = st.columns(len(steps))
    step_keys = list(st.session_state.step_completion.keys())
    for i, step_name in enumerate(steps):
        with cols[i]:
            if st.button(step_name, key=f"nav_{step_keys[i]}", width='stretch'):
                st.session_state.workflow_step = i + 1
                st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)


# --- Overhauled Configuration Page with Non-Linear Workflow ---
def display_configuration_page():
    """Renders the main pipeline configuration view with a non-linear workflow."""
    st.title("MLOps Pipeline Control Panel")
    
    if st.session_state.guide_message:
        st.info(st.session_state.guide_message)
        st.session_state.guide_message = None

    display_workflow_step_header()
    df_for_config = get_processed_df()
    
    # --- Step 1: Data Ingestion & Quality Assessment ---
    with st.container(border=True):
        st.header("Step 1: Data Ingestion & Quality Assessment")
        with st.expander("Expand to manage data and quality", expanded=st.session_state.workflow_step == 1):
            display_auto_config_section(st.session_state.df)
            display_data_quality_section(df_for_config)
            display_data_cleaning_section()
            display_conversational_data_transformation()
            display_advanced_data_cleaning_panel()

            # FIX: Added width='stretch'
            if st.button("Proceed to EDA", type="primary", width='stretch'):
                st.session_state.workflow_step = 2
                st.rerun()
            
    # --- Step 2: Exploratory Data Analysis (EDA) ---
    with st.container(border=True):
        st.header("Step 2: Exploratory Data Analysis (EDA)")
        with st.expander("Expand to explore data", expanded=st.session_state.workflow_step == 2):
            st.markdown("Understand your data before creating features or training models. The VIF analysis is located inside the 'Multicollinearity' tab of the EDA report.")
            st.button("Show/Hide Comprehensive EDA Report", width='stretch', on_click=lambda: st.session_state.update(show_eda=not st.session_state.get('show_eda', False)))
            if st.session_state.get('show_eda'):
                display_comprehensive_eda(df_for_config)

            # FIX: Added width='stretch'
            if st.button("Proceed to Feature Engineering", type="primary", width='stretch'):
                st.session_state.workflow_step = 3
                st.rerun()

    # --- Step 3: Feature Engineering & Selection ---
    with st.container(border=True):
        st.header("Step 3: Feature Engineering & Selection")
        with st.expander("Expand to create features", expanded=st.session_state.workflow_step == 3):
            col1, col2 = st.columns(2)
            with col1: display_ai_feature_engineering(st.session_state.df)
            with col2: display_manual_feature_engineering(df_for_config)
            
            # FIX: Added width='stretch'
            if st.button("Proceed to Model Configuration", type="primary", width='stretch'):
                st.session_state.workflow_step = 4
                st.rerun()
    
    # --- Step 4: Model Selection & Configuration ---
    with st.container(border=True):
        st.header("Step 4: Model Selection & Configuration")
        with st.expander("Expand to configure pipeline", expanded=st.session_state.workflow_step == 4):
            display_pipeline_settings(df_for_config)
            display_preprocessing_summary()
            
            # FIX: Added width='stretch'
            if st.button("Proceed to Training", type="primary", width='stretch'):
                st.session_state.workflow_step = 5
                st.rerun()

    # --- Step 5: Training & Validation ---
    with st.container(border=True):
        st.header("Step 5: Training & Validation")
        with st.expander("Expand to run training", expanded=st.session_state.workflow_step == 5):
            display_run_section()

def display_conversational_data_transformation():
    """UI for transforming data using natural language."""
    st.subheader("Conversational Data Transformation")
    if 'transformation_history' not in st.session_state:
        st.session_state.transformation_history = []

    user_query = st.text_input("Describe the data transformation you want to perform (e.g., 'drop column X', 'create a new feature by dividing A by B'):", key="data_transform_query")

    # FIX: Added width='stretch'
    if st.button("Apply Transformation", key="apply_transform_button", width='stretch'):
        df = get_processed_df()
        api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))
        if df is not None and user_query and api_key:
            df_modified, code_executed = execute_data_transformation(df, user_query, api_key)
            if df_modified is not None:
                st.session_state.df = df_modified
                st.session_state.transformation_history.append({'query': user_query, 'code': code_executed})
                st.success("Transformation applied successfully.")
                st.rerun()
        elif not api_key:
            st.error("Please provide a Groq API key in the sidebar.")
        elif not user_query:
            st.warning("Please describe the transformation you want to perform.")

    if st.session_state.transformation_history:
        with st.expander("View Transformation History"):
            for item in reversed(st.session_state.transformation_history):
                st.markdown(f"**Query:** `{item['query']}`")
                st.code(item['code'], language='python')

def display_advanced_data_cleaning_panel():
    """
    Comprehensive data cleaning interface with live preview.
    All operations update st.session_state.df directly.
    """
    st.subheader("Advanced Data Cleaning Toolkit")
    
    df = st.session_state.get('df')
    if df is None:
        st.warning("Load data first to use cleaning tools.")
        return
        
    # Create tabs for different cleaning operations
    clean_tabs = st.tabs([
        "Type Conversion", 
        "Value Replacement", 
        "Column Operations",
        "Row Filtering",
        "Outlier Handling"
    ])
    
    # TAB 1: Type Conversion
    with clean_tabs[0]:
        st.markdown("### Convert Column Data Types")
        col_to_convert = st.selectbox(
            "Select column:",
            options=df.columns.tolist(),
            key="type_convert_col"
        )
        
        current_type = str(df[col_to_convert].dtype)
        st.info(f"Current type: `{current_type}`")
        
        new_type = st.selectbox(
            "Convert to:",
            options=["int64", "float64", "string", "datetime64", "category"],
            key="new_type"
        )
        
        # Show preview
        if st.button("Preview Conversion", key="preview_type_convert"):
            try:
                if new_type == "datetime64":
                    preview = pd.to_datetime(df[col_to_convert], errors='coerce')
                elif new_type == "category":
                    preview = df[col_to_convert].astype('category')
                else:
                    preview = df[col_to_convert].astype(new_type)
                
                st.success("Conversion preview successful")
                st.dataframe(preview.head(10).to_frame())
                
                if st.button("Apply Conversion", key="apply_type_convert"):
                    # Apply to base dataframe
                    if new_type == "datetime64":
                        st.session_state.df[col_to_convert] = pd.to_datetime(
                            st.session_state.df[col_to_convert], 
                            errors='coerce'
                        )
                    else:
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(new_type)
                    
                    # CRITICAL: Clear processed df cache so changes are reflected
                    get_processed_df.clear()
                    
                    # CRITICAL: If this column is an AI-generated feature, we need to update
                    # the feature recommendations so it doesn't get recreated as float64
                    if st.session_state.get('ai_features_approved'):
                        # Features are already in base df, conversion will persist
                        logging.info(f"Converted column '{col_to_convert}' to {new_type} (AI feature)")
                    
                    st.success(f"Converted '{col_to_convert}' to {new_type}")
                    st.info("Conversion applied! Refresh the EDA to see changes.")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Conversion failed: {e}")
                logging.error(f"Type conversion error for {col_to_convert}: {e}", exc_info=True)

    # TAB 2: Value Replacement
    with clean_tabs[1]:
        st.markdown("### Replace or Map Values")
        replace_col = st.selectbox(
            "Select column:",
            options=df.columns.tolist(),
            key="replace_col"
        )
        
        st.write("**Unique values in column:**")
        unique_vals = df[replace_col].unique()[:20]  # Show first 20
        st.write(unique_vals)
        
        col_a, col_b = st.columns(2)
        old_value = col_a.text_input("Value to replace:", key="old_val")
        new_value = col_b.text_input("Replace with:", key="new_val")
        
        if st.button("Apply Replacement", key="apply_replace"):
            if old_value:
                st.session_state.df[replace_col] = st.session_state.df[replace_col].replace(old_value, new_value)
                st.success(f"Replaced '{old_value}' with '{new_value}' in '{replace_col}'")
                get_processed_df.clear()
                st.rerun()

    # TAB 3: Column Operations
    with clean_tabs[2]:
        st.markdown("### Column Operations")
        
        # Rename column
        st.write("**Rename Column**")
        col_to_rename = st.selectbox("Select column:", df.columns.tolist(), key="rename_col")
        new_col_name = st.text_input("New name:", value=col_to_rename, key="new_col_name")
        
        if st.button("Rename", key="apply_rename"):
            if new_col_name and new_col_name != col_to_rename:
                st.session_state.df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'")
                get_processed_df.clear()
                st.rerun()
                
        st.markdown("---")
        
        # Drop column
        st.write("**Drop Column**")
        col_to_drop = st.selectbox("Select column to drop:", df.columns.tolist(), key="drop_col")
        
        if st.button("Drop Column (Cannot Undo!)", key="apply_drop", type="secondary"):
            st.session_state.df.drop(columns=[col_to_drop], inplace=True)
            st.success(f"Dropped column '{col_to_drop}'")
            get_processed_df.clear()
            st.rerun()

    # TAB 4: Row Filtering
    with clean_tabs[3]:
        st.markdown("### Filter Rows by Condition")
        filter_col = st.selectbox("Filter by column:", df.columns.tolist(), key="filter_col")
        
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            st.write("**Numeric Filter**")
            filter_op = st.selectbox("Operation:", [">", "<", ">=", "<=", "==", "!="], key="filter_op")
            filter_val = st.number_input("Value:", value=float(df[filter_col].median()), key="filter_val")
            
            if st.button("Preview Filter", key="preview_filter"):
                if filter_op == ">":
                    preview_df = df[df[filter_col] > filter_val]
                elif filter_op == "<":
                    preview_df = df[df[filter_col] < filter_val]
                elif filter_op == ">=":
                    preview_df = df[df[filter_col] >= filter_val]
                elif filter_op == "<=":
                    preview_df = df[df[filter_col] <= filter_val]
                elif filter_op == "==":
                    preview_df = df[df[filter_col] == filter_val]
                else:  # !=
                    preview_df = df[df[filter_col] != filter_val]
                
                st.info(f"Filter will keep {len(preview_df)} rows out of {len(df)}")
                st.dataframe(preview_df.head(20))
                
                if st.button("Apply Filter", key="apply_filter"):
                    st.session_state.df = preview_df
                    st.success(f"Applied filter. Dataset now has {len(st.session_state.df)} rows.")
                    get_processed_df.clear()
                    st.rerun()
        else:
            st.info("String/categorical filtering: Use chatbot with command like 'keep only rows where Country is USA'")

    # TAB 5: Outlier Handling
    with clean_tabs[4]:
        st.markdown("### Remove Statistical Outliers")
        outlier_col = st.selectbox(
            "Select numeric column:",
            options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
            key="outlier_col"
        )
        
        method = st.radio("Method:", ["IQR (1.5x)", "IQR (3x)", "Z-score (3)", "Percentile"], key="outlier_method")
        
        if st.button("Preview Outliers", key="preview_outliers"):
            series = df[outlier_col]
            
            if "IQR" in method:
                factor = 1.5 if "1.5" in method else 3.0
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
                outliers = df[(series < lower) | (series > upper)]
            elif "Z-score" in method:
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = df[z_scores > 3]
            else:  # Percentile
                lower, upper = series.quantile(0.01), series.quantile(0.99)
                outliers = df[(series < lower) | (series > upper)]
            
            st.warning(f"Found {len(outliers)} outlier rows ({len(outliers)/len(df)*100:.1f}%)")
            st.dataframe(outliers.head(20))
            
            if st.button("Remove Outliers", key="apply_outlier_removal"):
                st.session_state.df = df.drop(outliers.index)
                st.success(f"Removed {len(outliers)} outliers. Dataset now has {len(st.session_state.df)} rows.")
                get_processed_df.clear()
                st.rerun()

def display_auto_config_section(df):
    """UI for the initial AI-powered setup."""
    st.subheader("AI Auto-Configuration")
    st.markdown("Use AI to analyze your data and automatically set the problem type, target, and features.")
    if st.button("Auto-Configure Pipeline with AI", width='stretch', on_click=handle_auto_config, args=(df,)):
        st.rerun()
    if st.session_state.ai_config_rationale:
        with st.expander("View AI Rationale", expanded=False):
            st.info(st.session_state.ai_config_rationale)

def display_data_quality_section(df):
    """Displays a data quality score and summary."""
    st.subheader("Data Quality Report")
    if df is None:
        st.info("Load data to generate a quality report.")
        return
    
    report, score = profile_data(df)
    
    delta_text = ""
    if score > 85: delta_text = "Good"
    elif score > 60: delta_text = "Fair"
    else: delta_text = "Poor"
        
    st.metric(label="Data Quality Score", value=f"{score:.1f}/100", 
              delta=delta_text,
              delta_color="off")
    
    with st.expander("View Full Quality Report"):
        st.json(report)
        st.markdown(f"**Data Shape:** `{df.shape[0]}` rows, `{df.shape[1]}` columns.")

def display_ai_feature_engineering(df):
    """UI for AI-powered feature recommendations."""
    with st.container(border=True):
        st.subheader("AI-Powered Features")
        st.markdown("Let AI suggest new features.")
        if st.button("Get AI Feature Recommendations", width='stretch'):
            handle_feature_recommendations(df)
        if st.session_state.feature_recommendations:
            for rec in st.session_state.feature_recommendations:
                col_display = rec.get('column', f"{rec.get('column_1', 'N/A')} & {rec.get('column_2', 'N/A')}")
                confidence = rec.get('confidence_score', 'N/A')
                st.info(f"**Action:** `{rec.get('technique', 'N/A').capitalize()}` on **`{col_display}`** (Confidence: {confidence}%)\n\n**Rationale:**\n{rec.get('rationale_long', 'N/A')}")
            
            if st.button("Approve & Apply AI Features", on_click=handle_approve_features, width='stretch'):
                old_feature_count = len(st.session_state.config.get('numerical_features', [])) + len(st.session_state.config.get('categorical_features', []))
                st.session_state.feature_count_before = old_feature_count
                st.rerun()
        
        # Show what changed (if just applied)
        if st.session_state.get('feature_count_before'):
            new_count = len(st.session_state.config.get('numerical_features', [])) + len(st.session_state.config.get('categorical_features', []))
            added = new_count - st.session_state.feature_count_before
            
            if added > 0:
                st.success(f"Added {added} new features. Total features: {new_count}")
                st.session_state.feature_count_before = None  # Clear after showing


def display_manual_feature_engineering(df):
    """UI for users to create their own features manually."""
    with st.container(border=True):
        st.subheader("Manual Features")
        st.markdown("Create your own features.")

        if df is None:
            st.info("Load data to create features.")
            return

        num_features = st.session_state.config.get('numerical_features', [])
        
        with st.expander("Create Interaction Feature"):
            c1, c2 = st.columns(2)
            feat1 = c1.selectbox("Feature 1", num_features, key="man_int_1", index=0 if num_features else None)
            feat2 = c2.selectbox("Feature 2", num_features, key="man_int_2", index=min(1, len(num_features)-1) if num_features else None)
            if st.button("Create Interaction"):
                if feat1 and feat2:
                    rec = {"technique": "interaction", "column_1": feat1, "column_2": feat2}
                    st.session_state.feature_recommendations.append(rec)
                    handle_approve_features() # This will apply and sync
                    st.rerun()

        with st.expander("Create Polynomial Feature"):
            feat_poly = st.selectbox("Feature", num_features, key="man_poly", index=0 if num_features else None)
            degree = st.slider("Degree", 2, 5, 2, key="man_poly_deg")
            if st.button("Create Polynomial"):
                if feat_poly:
                    rec = {"technique": "polynomial", "column": feat_poly, "degree": degree}
                    st.session_state.feature_recommendations.append(rec)
                    handle_approve_features() # This will apply and sync
                    st.rerun()

def display_run_section():
    """Displays the main action buttons for running the pipeline."""
    phase1_complete = st.session_state.get('phase1_complete', False)
    
    st.info("This is the final step before model evaluation. Running the benchmark will train all selected individual models. You can then create an ensemble on the results page.")
    
    if phase1_complete:
        st.success("Benchmark complete! Proceed to the results page to analyze performance and build an ensemble.")
    
    validation_errors = validate_config(st.session_state.config, phase='training')
    if validation_errors:
        for error in validation_errors:
            st.warning(error)
            
    run_button_disabled = bool(validation_errors)

    cols = st.columns(3)
    
    if phase1_complete:
        if cols[0].button("View Results & Create Ensemble", width='stretch', type="primary"):
            st.session_state.workflow_step = 6
            st.session_state.view = 'results'
            st.rerun()
    
    st.toggle("Enable Quick Test Mode", value=st.session_state.config.get('quick_test', False), key='widget_quick_test', on_change=update_config, args=('quick_test',), help="Reduces CV folds and tuning iterations for a fast test run.")

    if cols[1].button("Benchmark Individual Models", type="primary", width='stretch', disabled=run_button_disabled):
        # CRITICAL: Clear all cached and previous state before new run
        st.session_state.phase1_complete = False
        st.session_state.phase1_results = None  # Clear previous benchmark
        st.session_state.results_data = None  # Clear previous results
        st.session_state.ai_summary = None  # Force new AI summary
        st.session_state.parsed_actions = []  # Clear old recommendations
    
        problem_type = st.session_state.config.get('problem_type', 'Classification')
        display_metric = st.session_state.config.get('primary_metric_display', 'ROC AUC')
        metric_map = {'ROC AUC': 'roc_auc', 'F1-Score': 'f1_weighted'} if problem_type == 'Classification' else {'R2 Score': 'r2', 'Negative MSE': 'neg_mean_squared_error'}
        st.session_state.config['primary_metric'] = metric_map.get(display_metric, 'roc_auc' if problem_type == 'Classification' else 'r2')

         # Generate new config hash for this specific run
        import time
        config_for_hash = st.session_state.config.copy()
        config_for_hash['run_timestamp'] = time.time()  # Force unique hash per run
        config_hash = generate_config_hash(config_for_hash)
    
        progress_placeholder = st.empty()

        # Clear model store and pipeline cache
        st.session_state.model_store.clear()
        run_pipeline.clear()

        logging.info(f"Starting NEW benchmark run with config hash: {config_hash}")
        results = run_pipeline(get_processed_df(), config_hash, progress_placeholder, st.session_state.model_store, phase='benchmark')

        if results and results[0] is not None:
            st.session_state.phase1_results = results[0]
            st.session_state.results_data = results
            st.session_state.phase1_complete = True
            st.session_state.workflow_step = 6
            st.session_state.view = 'results'
            st.rerun()

def display_data_cleaning_section():
    """UI for manual data cleaning steps like dropping duplicates and NaNs."""
    st.subheader("Data Cleaning Tools")
    df = st.session_state.get('df')
    if df is None:
        st.info("Load data to perform cleaning operations.")
        return
    
    c1, c2 = st.columns(2)
    with c1:
        num_duplicates = df.duplicated().sum()
        if st.button(f"Remove {num_duplicates} Duplicate Rows", disabled=num_duplicates == 0, width='stretch'):
            rows_before = len(st.session_state.df)
            st.session_state.df = df.drop_duplicates(inplace=False)
            st.success(f"Removed {num_duplicates} duplicate rows. Dataset now has {len(st.session_state.df)} rows.")
            st.rerun()
    
    with c2:
        if st.button("Drop Rows with Any Missing Values", width='stretch'):
            rows_before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.dropna(inplace=False)
            rows_after = len(st.session_state.df)
            st.success(f"Removed {rows_before - rows_after} rows with missing values.")
            st.rerun()

def display_preprocessing_summary():
    """Displays a summary of the automated preprocessing pipeline."""
    st.subheader("Preprocessing Pipeline Summary")
    with st.container(border=True):
        st.markdown("Based on your selections, the following automated steps will be applied to the data before training:")
        
        config = st.session_state.config
        steps_md = []
        
        steps_md.append(f"1. **Impute Missing Values**: Numerical data filled with **`{config['imputation_strategy']}`**; categorical data filled with a constant.")
        if config['handle_outliers']:
            steps_md.append("2. **Handle Outliers**: Extreme values will be capped using the IQR method to improve model robustness.")
        
        step_num = 3 if config['handle_outliers'] else 2
        steps_md.append(f"{step_num}. **Scale Numerical Features**: All numerical features will be transformed using **`{config['scaler']}`**.")
        steps_md.append(f"{step_num + 1}. **Encode Categorical Features**: Text-based categories will be converted into numerical format using **One-Hot Encoding**.")
        
        st.markdown("\n".join([f"- {s}" for s in steps_md]))


def display_comprehensive_eda(df):
    """Renders an interactive EDA report with integrated AI summaries and guidance."""
    if df is None:
        st.warning("Cannot display EDA because no data is loaded or processed.")
        return
    
    # FIX: Validate and sync config features with actual DataFrame columns
    target = st.session_state.config.get('target')
    current_num_features = st.session_state.config.get('numerical_features', [])
    current_cat_features = st.session_state.config.get('categorical_features', [])
    
    # Filter to only columns that exist in current DataFrame
    valid_num_features = [f for f in current_num_features if f in df.columns and f != target]
    valid_cat_features = [f for f in current_cat_features if f in df.columns and f != target]
    
    # Update config if features were removed
    if len(valid_num_features) != len(current_num_features) or len(valid_cat_features) != len(current_cat_features):
        st.session_state.config['numerical_features'] = valid_num_features
        st.session_state.config['categorical_features'] = valid_cat_features
        sync_all_widgets()
        logging.info(f"Synced config features with DataFrame. Removed {len(current_num_features) - len(valid_num_features)} numerical and {len(current_cat_features) - len(valid_cat_features)} categorical features.")
        
        
    with st.container(border=True):
        st.subheader("Exploratory Data Analysis Report", divider='violet')
        
        api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))
        if st.button("Get AI Summary of EDA", key="get_eda_summary", width='stretch'):
            if not api_key:
                st.error("Please provide a Groq API key to get an AI summary.")
            else:
                with st.spinner("AI is analyzing your data..."):
                    prompt = get_eda_summary_prompt(df, st.session_state.config)
                    st.session_state.eda_summary = generate_llm_response(prompt, api_key)
        if st.session_state.eda_summary:
            with st.expander("AI Data Summary & Key Insights", expanded=True):
                st.markdown(st.session_state.eda_summary)
        
        eda_tabs = st.tabs(["Data Preview & Download", "Numerical Analysis", "Categorical Analysis", "Multicollinearity (VIF)"])
        
        with eda_tabs[0]:
            st.subheader("Data Preview")
            st.dataframe(get_dataset_sample(df).head(10))
            st.download_button("Download Processed Data (CSV)", convert_df_to_csv(df), "processed_data.csv", "text/csv")
            st.subheader("Missing Values & Data Types")
            missing = df.isnull().sum().sort_values(ascending=False)
            st.dataframe(missing[missing > 0].to_frame(name='Missing Count')) if missing.sum() > 0 else st.success("No missing values found.")
            st.dataframe(df.dtypes.astype(str).to_frame(name='Data Type'))
            
        with eda_tabs[1]:
            st.subheader("Numerical Feature Analysis")
            num_features = st.session_state.config.get('numerical_features', [])
            # Add debugging prints here
            # st.write(f"DEBUG: Features selected in config: {num_features}")
            # st.write(f"DEBUG: Columns available in df: {df.columns.tolist()}")
            # st.write(f"DEBUG: dtypes in df:\n{df.dtypes}")

            # FIX: Filter num_features to only include columns present in the current df
            num_features = [col for col in num_features if col in df.columns]
            # st.write(f"DEBUG: Filtered numerical features: {num_features}") # See what remains after filtering
            
            if num_features:
                st.dataframe(df[num_features].describe())
                target = st.session_state.config.get('target')
                if target and target in df.columns:
                    st.plotly_chart(px.histogram(df, x=target, title=f'Distribution of Target: {target}'), use_container_width=True)
                
                st.subheader("Correlation Heatmap (Interactive)")
                # Also filter default features for multiselect
                default_corr_features = [f for f in num_features[:10] if f in df.columns]
                corr_features = st.multiselect("Select features for heatmap:", num_features, default=default_corr_features)
                if len(corr_features) > 1:
                    # Ensure target is also present before adding
                    cols_for_corr = corr_features + ([target] if target and target in df.columns and target not in corr_features else [])
                    corr_df = df[cols_for_corr]
                    st.plotly_chart(px.imshow(corr_df.corr(numeric_only=True), text_auto=".2f", title="Correlation Heatmap", aspect="auto"), use_container_width=True)
                else:
                    st.info("Please select at least two numerical features to display a correlation heatmap.")
            else: st.info("No numerical features selected or available in the current dataset.")
                
        with eda_tabs[2]:
            st.subheader("Categorical Feature Analysis")
            cat_features = st.session_state.config.get('categorical_features', [])
            # FIX: Filter cat_features to only include columns present in the current df
            cat_features = [col for col in cat_features if col in df.columns]
            if cat_features:
                selected_cat_col = st.selectbox("Select a categorical feature to visualize:", cat_features, key="eda_cat_feature")
                if selected_cat_col:
                    target = st.session_state.config.get('target')
                    # Ensure target exists before using for color
                    color_arg = target if target and target in df.columns else None
                    st.plotly_chart(px.histogram(df, x=selected_cat_col, title=f'Distribution of {selected_cat_col}', color=color_arg, barmode='group'), use_container_width=True)
            else: st.info("No categorical features selected or available in the current dataset.")
        
        with eda_tabs[3]:
            st.subheader("Variance Inflation Factor (VIF)")
            st.markdown("VIF measures how much a feature is explained by other features. A score > 10 may indicate problematic multicollinearity.")
            if not STATSMODELS_AVAILABLE:
                st.warning("The `statsmodels` library is not installed. Please run `pip install statsmodels` to use the VIF feature.")
            else:
                num_features = st.session_state.config.get('numerical_features', [])
                # FIX: Filter num_features for VIF calculation as well
                num_features = [col for col in num_features if col in df.columns]
                
                if len(num_features) < 2:
                    st.info("Select at least two numerical features currently in the data to calculate VIF.")
                else:
                    with st.spinner("Calculating VIF..."):
                        try:
                            # Use the already filtered num_features list
                            vif_df = df[num_features].dropna() 
                            if vif_df.shape[0] < 2:
                                st.warning("Could not calculate VIF. Not enough data remains after dropping missing values.")
                            else:
                                vif_data = pd.DataFrame()
                                vif_data["Feature"] = vif_df.columns
                                vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]
                                st.dataframe(vif_data.sort_values('VIF', ascending=False).style.map(lambda v: 'background-color: #ffcccc' if v > 10 else ('background-color: #fff3cd' if v > 5 else ''), subset=['VIF']))
                                
                                high_vif_features = vif_data[vif_data["VIF"] > 10]
                                if not high_vif_features.empty:
                                    if st.button("Get AI Advice on Multicollinearity", width='stretch'):
                                        if not api_key:
                                            st.error("Please provide a Groq API key to get AI advice.")
                                        else:
                                            prompt = get_vif_analysis_prompt(high_vif_features)
                                            st.session_state.vif_summary = generate_llm_response(prompt, api_key, is_json=True)
                                    
                                    if st.session_state.vif_summary:
                                        try:
                                            vif_recs = json.loads(st.session_state.vif_summary)
                                            st.info(f"**AI Explanation:** {vif_recs.get('explanation')}")
                                            
                                            features_to_remove_info = vif_recs.get("features_to_remove", [])
                                            if features_to_remove_info:
                                                st.warning("AI recommends removing the following features to reduce multicollinearity:")
                                                for item in features_to_remove_info:
                                                    st.markdown(f"- **`{item['feature']}`**: *{item['rationale']}*")
                                                
                                                if st.button("Apply AI Recommendations", width='stretch'):
                                                    features_to_remove = [item['feature'] for item in features_to_remove_info]
                                                    # Operate on the already filtered list
                                                    current_num_features = num_features 
                                                    updated_features = [f for f in current_num_features if f not in features_to_remove]
                                                    
                                                    # Update the main config
                                                    st.session_state.config['numerical_features'] = [f for f in st.session_state.config['numerical_features'] if f in updated_features]
                                                    sync_all_widgets() # Sync UI
                                                    st.success(f"Successfully removed {len(current_num_features) - len(updated_features)} feature(s) from selection.")
                                                    st.rerun() # Rerun to refresh EDA with updated features
                                        except (json.JSONDecodeError, KeyError) as e:
                                            st.error(f"AI returned advice, but it could not be parsed. Error: {e}")
                                            st.text_area("Raw AI Response", value=st.session_state.vif_summary)

                        except ValueError as e:
                            st.error("VIF calculation failed: One or more features may have zero variance or be perfectly collinear.")
                            logging.error(f"VIF calculation ValueError: {e}", exc_info=True)
                            st.info("Try removing constant features or features that are exact copies of each other.")
                        except np.linalg.LinAlgError as e:
                            st.error("VIF calculation failed due to singular matrix. Features may be perfectly collinear.")
                            logging.error(f"VIF calculation LinAlgError: {e}", exc_info=True)
                        except Exception as e: 
                            st.error(f"An error occurred during VIF calculation: {e}")
                            logging.error("VIF calculation failed.", exc_info=True)

def display_results_page():
    st.title("Step 6: Model Evaluation & Deployment")
    
    results_data = st.session_state.get('results_data')
    if not results_data or not isinstance(results_data, tuple) or len(results_data) != 6:
        st.error("Results are not available. Please re-run the pipeline.")
        if st.button("Back to Configuration", width='stretch'): 
            st.session_state.view = 'configuration'
            st.rerun()
        return

    results_df, le, X_train, y_train, X_test, y_test = results_data

    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        st.error("No models were successfully trained. Cannot display results.")
        if st.button("Back to Configuration", width='stretch'): 
            st.session_state.view = 'configuration'
            st.rerun()
        return

    metric_col = st.session_state.config['primary_metric_display']
    results_df_sorted = results_df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)
    best_row = results_df_sorted.iloc[0]
    
    best_model = st.session_state.model_store.get_model(best_row['model_id'])
    if not best_model:
        st.error(f"Could not retrieve the best model ('{best_row['Model']}'). Please re-run the pipeline.")
        return

    if st.session_state.get('phase1_complete', False) and 'Voting' not in results_df['Model'].unique():
        display_ensemble_creation_panel(st.session_state.phase1_results, metric_col)

    st.header("Results Dashboard")
    
    # --- TAB CREATION LOGIC ---
    # 1. Define the standard tabs
    tabs_list = [
        "Performance Comparison", 
        "Run History", 
        "AI Summary & Actions", 
        "Model Inspector", 
        "Explainability", 
        "Drift Monitoring", 
        "Deployment"
    ]

    # 2. Insert RAG tab if available
    if st.session_state.get('rag_available', False):
        tabs_list.insert(2, "Semantic Search")

    # 3. Generate the tabs
    all_tabs = st.tabs(tabs_list)

    # 4. Map tabs to variables by index
    # Note: We increment idx after every assignment to keep track of where we are
    idx = 0
    tab1 = all_tabs[idx]; idx += 1  # Performance
    tab2 = all_tabs[idx]; idx += 1  # History

    # Conditional RAG assignment
    if st.session_state.get('rag_available', False):
        tab_rag = all_tabs[idx]; idx += 1
    else:
        tab_rag = None

    tab3 = all_tabs[idx]; idx += 1  # AI Summary
    tab4 = all_tabs[idx]; idx += 1  # Inspector
    tab5 = all_tabs[idx]; idx += 1  # Explainability
    tab6 = all_tabs[idx]; idx += 1  # Drift
    tab7 = all_tabs[idx]; idx += 1  # Deployment

    # --- TAB CONTENT ---

    # TAB 1: Performance Comparison
    with tab1:
        st.dataframe(results_df_sorted.drop(columns='model_id').style.highlight_max(axis=0, color='#d4edda', subset=[metric_col]))
    
    # TAB 2: Run History
    with tab2:
        st.subheader("Training Run History")
        if st.session_state.get('experiment_history'):
            history_df = pd.DataFrame(st.session_state.experiment_history)
            display_cols = ['timestamp', 'best_model', 'best_score', 'metric_name', 'num_features']
            st.dataframe(history_df[display_cols].tail(10))
            
            # Show comparison chart
            if len(history_df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['best_score'],
                    mode='lines+markers',
                    name='Best Score',
                    text=history_df['best_model'],
                    hovertemplate='Run %{x}<br>Model: %{text}<br>Score: %{y:.4f}<extra></extra>'
                ))
                fig.update_layout(
                    title=f'Score Improvement Over Runs',   
                    xaxis_title='Run Number',
                    yaxis_title=metric_col,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No previous runs to compare. Run the pipeline multiple times to see trends.")

    # TAB RAG: Semantic Search (Conditional)
    if st.session_state.get('rag_available', False) and tab_rag:
        with tab_rag:
            display_semantic_search_tab()

    # TAB 3: AI Summary & Actions
    with tab3:
        display_ai_summary_tab()
    
    # TAB 4: Model Inspector
    with tab4:
        display_performance_metrics_tab(results_df_sorted, X_test, y_test, le, metric_col)
    
    # TAB 5: Explainability
    with tab5:
        display_explainability_tab(best_row, best_model, X_test)
    
    # TAB 6: Drift Monitoring
    with tab6:
        display_drift_monitoring_tab()
    
    # TAB 7: Deployment
    with tab7:
        display_deployment_tab(best_model, best_row)

    # Global "New Project" button at bottom of results
    if st.button("Start New Project (Back to Configuration)", width='stretch'):
        st.session_state.view = 'configuration'
        # Reset workflow for a new run, but keep data
        st.session_state.workflow_step = 1
        st.session_state.step_completion = {key: False for key in st.session_state.step_completion}
        st.rerun()

# --- Helper Functions for Results Page ---

def display_ensemble_creation_panel(phase1_df, metric):
    with st.container(border=True):
        st.subheader("Create a Smart Ensemble")
        st.markdown("Based on the benchmark results, you can combine the best models into a powerful ensemble. The AI's suggestions are pre-selected below.")
        
        if not st.session_state.get('suggested_base_models'):
             st.session_state.suggested_base_models = suggest_base_models(phase1_df, metric)
        
        problem_type = st.session_state.config['problem_type']
        all_individual_models = [m for m in get_models_and_search_spaces(problem_type).keys() if 'Voting' not in m]

        st.multiselect(
            "Select Base Models for Ensemble",
            options=all_individual_models,
            default=st.session_state.suggested_base_models,
            key='widget_ensemble_base_models',
            on_change=update_config,
            args=('ensemble_base_models',)
        )

        validation_errors = validate_config(st.session_state.config, phase='ensemble')
        if validation_errors:
            for error in validation_errors:
                st.warning(error)

        if st.button("Train Ensemble Model", type="primary", width='stretch', disabled=bool(validation_errors)):
            with st.spinner("Training ensemble model..."):
                # Force unique config hash for ensemble
                import time
                config_for_hash = st.session_state.config.copy()
                config_for_hash['ensemble_timestamp'] = time.time()
                config_hash = generate_config_hash(config_for_hash)
                
                progress_placeholder = st.empty()
                run_pipeline.clear()
                
                logging.info(f"Starting ensemble training with base models: {st.session_state.config.get('ensemble_base_models')}")
                results = run_pipeline(get_processed_df(), config_hash, progress_placeholder, st.session_state.model_store, phase='ensemble')

                if results and results[0] is not None:
                    # Get current results_data
                    current_results_df = None
                    if st.session_state.get('results_data') and st.session_state.results_data[0] is not None:
                        current_results_df = st.session_state.results_data[0]
                    elif st.session_state.get('phase1_results') is not None:
                        current_results_df = st.session_state.phase1_results
                    
                    # Append new ensemble to existing results
                    if current_results_df is not None:
                        # Remove old ensemble if it exists
                        current_results_df = current_results_df[~current_results_df['Model'].str.contains('Voting', na=False)]
                        
                        # Combine with new ensemble
                        combined_df = pd.concat([current_results_df, results[0]], ignore_index=True)
                        st.session_state.results_data = (combined_df, *results[1:])
                        logging.info(f"Appended ensemble to existing results. Total models: {len(combined_df)}")
                    else:
                        st.session_state.results_data = results
                        
                    # Clear AI summary to regenerate with new results
                    st.session_state.ai_summary = None
                    st.rerun()

def display_pipeline_settings(df_for_config):
    """Manages UI for pipeline settings with the robust on_change callback pattern."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Problem & MLOps")
        st.radio("Problem Type", ["Classification", "Regression"], 
                 key='widget_problem_type', horizontal=True, on_change=handle_problem_type_change,
                 help="This is auto-inferred from the target, but can be overridden.")
        st.text_input("MLflow Tracking URI", value=st.session_state.config.get('mlflow_uri', ''), key='widget_mlflow_uri', on_change=update_config, args=('mlflow_uri',), help="URI for the MLflow tracking server (e.g., './mlruns' or 'http://host:port').")
        st.text_input("MLflow Experiment Name", value=st.session_state.config.get('mlflow_experiment_name', ''), key='widget_mlflow_experiment_name', on_change=update_config, args=('mlflow_experiment_name',))
        
        st.subheader("Data & Features")
        all_cols = list(df_for_config.columns) if df_for_config is not None else []
        st.selectbox("Target Variable", all_cols, key='widget_target', on_change=handle_target_change, help="Select the column you want the model to predict.")

        if df_for_config is not None:
            detected = FeatureDetector.detect_features(df_for_config, target=st.session_state.config.get('target'))
            options_num = sorted(list(set(detected.get('numerical', []) + detected.get('id', []))))
            options_cat = sorted(detected.get('categorical', []))
        else:
            options_num, options_cat = [], []
        
        st.multiselect("Numerical Features", options_num, key='widget_numerical_features', on_change=update_config, args=('numerical_features',), help="Features containing numeric data (e.g., Age, Salary).")
        st.multiselect("Categorical Features", options_cat, key='widget_categorical_features', on_change=update_config, args=('categorical_features',), help="Features containing text or category labels (e.g., Country, Gender).")
    
    with col2:
        st.subheader("Preprocessing & Tuning")
        st.slider("Test Set Size", 0.1, 0.5, step=0.05, key='widget_test_size', on_change=update_config, args=('test_size',), help="Percentage of data to hold out for final model evaluation.")
        st.number_input("CV Folds", 2, 10, key='widget_cv_folds', on_change=update_config, args=('cv_folds',), help="Number of folds for cross-validation during hyperparameter tuning.")
        st.checkbox("Handle Outliers (IQR)", key='widget_handle_outliers', on_change=update_config, args=('handle_outliers',), help="Caps extreme values to make the model more robust.")
        st.selectbox("Numerical Imputation", ["median", "mean", "most_frequent"], key='widget_imputation_strategy', on_change=update_config, args=('imputation_strategy',), help="Strategy to fill in missing numerical data.")
        st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"], key='widget_scaler', on_change=update_config, args=('scaler',), help="Method for scaling numerical features to a standard range.")

    with col3:
        st.subheader("Model Selection & Advanced")
        st.checkbox("Use Stat. Feature Selection", key='widget_use_statistical_feature_selection', on_change=update_config, args=('use_statistical_feature_selection',), help="Automatically select the 'k' best features based on statistical tests.")
        if st.session_state.config.get('use_statistical_feature_selection'):
            num_feats = len(st.session_state.config.get('numerical_features', [])) + len(st.session_state.config.get('categorical_features', []))
            if num_feats > 1:
                k_val = st.session_state.config.get('k_features', min(10, num_feats))
                st.slider("Num Features to Select", 1, num_feats, k_val, key='widget_k_features', on_change=update_config, args=('k_features',))
        
        problem_type = st.session_state.config.get('problem_type', 'Classification')
        all_individual_models = [m for m in get_models_and_search_spaces(problem_type).keys() if 'Voting' not in m]
        st.multiselect(
            "Select Individual Models to Benchmark", 
            all_individual_models, 
            key='widget_selected_models', 
            on_change=update_config, 
            args=('selected_models',)
        )
        
        if problem_type == 'Classification':
            options = ['ROC AUC', 'F1-Score']
            st.selectbox("Primary Metric", options, key='widget_primary_metric_display', on_change=update_config, args=('primary_metric_display',), help="The main score used to rank models.")
            imbalance_options = ["None", "SMOTE"]
            st.selectbox("Imbalanced Data Handling", imbalance_options, key='widget_imbalance_method', on_change=update_config, args=('imbalance_method',), help="Technique to handle datasets with a rare target class (e.g., fraud detection).")
        else:
            options = ['R2 Score', 'Negative MSE']
            st.selectbox("Primary Metric", options, key='widget_primary_metric_display', on_change=update_config, args=('primary_metric_display',), help="The main score used to rank models.")
            st.session_state.config['imbalance_method'] = "None"
        
        st.slider("Tuning Iterations", 5, 50, key='widget_bayes_iterations', on_change=update_config, args=('bayes_iterations',), help="Number of iterations for Bayesian hyperparameter optimization.")


def display_ai_summary_tab():
    st.subheader("AI Executive Summary & Action Panel")
    api_key = os.getenv('GROQ_API_KEY', st.session_state.config.get('groq_api_key', ''))

    # Check if results have changed since last summary
    results_hash = None
    if st.session_state.get('results_data') and st.session_state.results_data[0] is not None:
        results_df = st.session_state.results_data[0]
        results_hash = hashlib.md5(results_df.to_string().encode()).hexdigest()

    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
        st.session_state.last_results_hash = None

    # Regenerate if results changed
    if results_hash != st.session_state.get('last_results_hash'):
        st.session_state.ai_summary = None
        st.session_state.last_results_hash = results_hash
        logging.info("Results changed - will regenerate AI summary")

    if not st.session_state.ai_summary:
        if not api_key:
            st.warning("Provide a Groq API key in the sidebar to generate an AI summary.")
        else:
            with st.spinner("AI is generating a summary..."):
                if st.session_state.get('results_data') and st.session_state.results_data[0] is not None:
                    results_df = st.session_state.results_data[0]
                    ensemble_ran = any('Voting' in model for model in results_df['Model'])
                    prompt = f"""Summarize the results for a '{st.session_state.config['problem_type']}' problem predicting '{st.session_state.config['target']}'.
The primary metric is '{st.session_state.config['primary_metric_display']}'.
{'An ensemble model was also run.' if ensemble_ran else ''}
Here is the performance table:
{results_df.to_string()}

Provide a summary, an interpretation of the best model's performance, and 2-3 actionable, diverse recommendations for the next run.
If an ensemble was run, comment specifically on whether it outperformed the individual models.
Format the response in clear Markdown."""
                    summary = generate_llm_response(prompt, api_key)
                    st.session_state.ai_summary = summary if summary else "Could not generate AI summary."
                else:
                    st.session_state.ai_summary = "Could not generate AI summary as no model results are available."

    st.markdown(st.session_state.ai_summary or "")

    # Parse recommendations using enhanced parser
    if st.session_state.ai_summary and api_key:
        # Only parse if we haven't already parsed these results
        current_hash = hashlib.md5(st.session_state.ai_summary.encode()).hexdigest()
        if st.session_state.get('last_parsed_hash') != current_hash:
            st.session_state.parsed_actions = parse_ai_recommendations(
                st.session_state.ai_summary,
                api_key
            )
            st.session_state.last_parsed_hash = current_hash

    if st.session_state.get('parsed_actions'):
        with st.container(border=True):
            st.subheader("AI Action Panel")
            st.markdown("The AI suggests the following improvements. Click to apply them to your configuration for the next run.")

            # Display each recommendation with better formatting
            for i, action in enumerate(st.session_state.parsed_actions, 1):
                action_text = action.get('action_text', 'N/A')
                st.info(f"**Recommendation {i}:** {action_text}")

            if st.button("Apply Recommendations & Return to Configuration", width='stretch'):
                 apply_ai_recommendations()

def display_performance_metrics_tab(results_df, X_test, y_test, le, metric):
    st.subheader("Model Inspector")
    st.markdown("Select any trained model from the run to view its specific performance plots.")
    
    model_to_inspect = st.selectbox("Select model to inspect:", results_df['Model'].tolist())
    if model_to_inspect:
        model_id = results_df[results_df['Model'] == model_to_inspect]['model_id'].iloc[0]
        model = st.session_state.model_store.get_model(model_id)
        if model:
            y_pred = model.predict(X_test)
            if st.session_state.config['problem_type'] == 'Classification':
                labels = [int(l) for l in np.unique(np.concatenate((y_test, y_pred)))]
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                class_names = le.inverse_transform(labels) if le else labels
                fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=class_names, y=class_names, title=f"Confusion Matrix for {model_to_inspect}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title=f'Actual vs. Predicted for {model_to_inspect}', trendline='ols')
                st.plotly_chart(fig, use_container_width=True)

def display_explainability_tab(best_model_row, best_model, X_test):
    """Shows SHAP and Partial Dependence Plots with enhanced safety."""
    st.subheader(f"Explaining '{best_model_row['Model']}'")

    if 'Voting' in best_model_row['Model']:
        st.warning("Explainability plots are not supported for Voting ensembles. Please select an individual model to see its plots.")
        return

    if not SHAP_AVAILABLE:
        st.warning("The `shap` library is not installed. Please run `pip install shap` to enable all explainability features.")
        return
        
    explain_tabs = st.tabs(["Feature Importance (SHAP)", "Advanced Explainability (PDP)"])

    with explain_tabs[0]:
        st.markdown("SHAP (SHapley Additive exPlanations) values show the impact of each feature on individual predictions, highlighting which features are most important globally.")
        with st.spinner("Calculating SHAP values..."):
            shap_values, X_test_df = get_shap_values(best_model, X_test)
            if shap_values is not None and X_test_df is not None:
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None)
                st.pyplot(fig)
                st.session_state.shap_fig = fig

    with explain_tabs[1]:
        st.markdown("Partial Dependence Plots (PDP) show how a single feature affects the model's average prediction, isolating its impact from all other features.")
        with st.spinner("Calculating Partial Dependence Plots..."):
            shap_values, X_test_df = get_shap_values(best_model, X_test)
            if shap_values is not None and X_test_df is not None and not X_test_df.empty:
                top_features = pd.Series(np.abs(shap_values.values).mean(0), index=X_test_df.columns).sort_values(ascending=False).index[:5]
                if not top_features.empty:
                    selected_pdp_feature = st.selectbox("Select a feature to plot:", top_features)
                    if selected_pdp_feature:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        try:
                            original_feature_name = re.sub(r'^(num__|cat__|remainder__)', '', selected_pdp_feature)
                            pdp_feature_to_use = original_feature_name
                            for col in X_test.columns:
                                if original_feature_name.startswith(col):
                                    pdp_feature_to_use = col
                                    break

                            PartialDependenceDisplay.from_estimator(best_model, X_test, features=[pdp_feature_to_use], ax=ax)
                            ax.set_title(f"Partial Dependence Plot for '{pdp_feature_to_use}'")
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Could not generate PDP plot. Error: {e}")
                            logging.error(f"PDP plot generation failed for feature '{selected_pdp_feature}'.", exc_info=True)
                else:
                    st.info("No top features could be determined for PDP.")

def display_drift_monitoring_tab():
    st.subheader("Data Drift Monitoring")
    if not st.session_state.training_data_stats:
        st.warning("Run a pipeline first to establish a baseline.")
        return
    if new_file := st.file_uploader("Upload New Data for Drift Analysis", type=['csv'], key="drift_uploader"):
        new_df = load_data(new_file)
        if new_df is None:
            return

        base_stats = st.session_state.training_data_stats
        num_features = st.session_state.config.get('numerical_features', [])
        st.subheader("Numerical Feature Drift (KS Test)")
        results = []
        for col in num_features:
            if col in new_df.columns and col in base_stats['X_train_ref'].columns:
                base_series, new_series = base_stats['X_train_ref'][col].dropna(), new_df[col].dropna()
                if len(base_series) > 1 and len(new_series) > 1:
                    _, p_val = ks_2samp(base_series, new_series)
                    results.append({'Feature': col, 'P-Value': p_val, 'Drift Detected': p_val < 0.05})
        if results:
            st.dataframe(pd.DataFrame(results).style.map(lambda v: 'color: red' if v else '', subset=['Drift Detected']))

def display_deployment_tab(best_model, best_model_row):
    st.subheader("Deployment Assets")
    st.info("Download the best model by itself or as a complete, production-ready package.")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Model Only (.pkl)",
            data=pickle.dumps(best_model),
            file_name="model.pkl",
            mime="application/octet-stream",
            width='stretch',
            help="Downloads only the trained model file."
        )
    with col2:
        from .utils import generate_deployment_package
        deployment_zip = generate_deployment_package(best_model, st.session_state.config)
        st.download_button(
            label="Download Complete Package (.zip)",
            data=deployment_zip,
            file_name="ml_model_deployment.zip",
            mime="application/zip",
            width='stretch',
            type="primary",
            help="Downloads a zip file with the model, API script, Dockerfile, config.yaml, and instructions."
        )
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 15px;">
        <strong>The complete package includes:</strong>
        <ul>
            <li><code>model.pkl</code>: The trained machine learning model.</li>
            <li><code>app.py</code>: A ready-to-use FastAPI script.</li>
            <li><code>requirements.txt</code>: Required Python libraries.</li>
            <li><code>Dockerfile</code>: For easy containerization and deployment.</li>
            <li><code>config.yaml</code>: Human-readable configuration file.</li>
            <li><code>README.md</code>: Instructions on how to run and deploy the model.</li>
            <li><code>model_metadata.json</code>: Details about the model configuration.</li>
        </ul>
        <strong>To deploy with Docker:</strong> <code>unzip</code> → <code>docker build -t api .</code> → <code>docker run -p 8000:8000 api</code>
    </div>
    """, unsafe_allow_html=True)
    best_model_name = best_model_row['Model']
    html_report = generate_html_report(best_model_name)
    st.download_button(
        label="Download HTML Report", 
        data=html_report, 
        file_name="ml_report.html", 
        mime="text/html", 
        width='stretch'
    )

def display_experiment_tracking_page():
    st.title("Experiment Tracking Dashboard")
    if not st.session_state.get('experiment_history'):
        st.info("No experiments have been run yet. Run a pipeline from the 'Configuration' page to see results here.")
        if st.button("Back to Configuration", width='stretch'):
            st.session_state.view = 'configuration'
            st.rerun()
        return
    history_df = pd.DataFrame(st.session_state.experiment_history)
    
    display_cols = [
        'timestamp', 'problem_type', 'best_model', 
        'best_score', 'metric_name', 'num_features', 'ai_features_used'
    ]
    
    st.dataframe(
        history_df[display_cols].sort_values('timestamp', ascending=False),
        hide_index=True
    )
    st.download_button(
        "Download History (CSV)",
        convert_df_to_csv(history_df[display_cols]),
        "experiment_history.csv",
        "text/csv"
    )
    st.markdown("---")
    st.subheader("View Details & Reload Configuration")
    if not history_df.empty:
        experiment_options = [
            f"{exp['timestamp']} - {exp['best_model']} ({exp['best_score']:.4f})"
            for exp in reversed(st.session_state.experiment_history)
        ]
        
        selected_idx = st.selectbox(
            "Select an experiment to view details or load its configuration:", 
            range(len(experiment_options)),
            format_func=lambda x: experiment_options[x]
        )
        selected_exp = st.session_state.experiment_history[-(selected_idx + 1)]
        if st.button("Load This Configuration", type="primary", width='stretch'):
            from .utils import load_config_from_experiment
            load_config_from_experiment(selected_exp)
            st.session_state.view = 'configuration'
            st.rerun()
            
        with st.expander("Show Full Configuration for Selected Experiment"):
            st.json(selected_exp['full_config'])
    
def display_semantic_search_tab():
    st.subheader("Semantic Experiment Search")
    st.markdown("Search your ML experiment history using natural language")
    
    try:
        from .rag_system import semantic_experiment_search, is_rag_available
        
        if not is_rag_available():
            st.warning("No experiments indexed yet. Run a pipeline first to build the search index.")
            return
        
        search_query = st.text_input(
            "Ask about past experiments:",
            placeholder="e.g., 'Find classification experiments with accuracy > 0.90'",
            key="rag_search_query"
        )
        
        with st.expander("Example Queries"):
            st.markdown("""
            - "Show me my best classification models"
            - "Find experiments that used XGBoost"
            - "What were my regression models with R² above 0.85?"
            - "Show experiments with SMOTE enabled"
            """)
        
        if st.button("Search", type="primary", width='stretch') and search_query:
            with st.spinner("Searching experiments..."):
                results = semantic_experiment_search(search_query, k=3)
            
            if not results:
                st.info("No matching experiments found. Try running more experiments first.")
            else:
                st.success(f"Found {len(results)} relevant experiments")
                
                for i, result in enumerate(results, 1):
                    with st.expander(
                        f"Rank {i}: {result['experiment_id']} "
                        f"(Similarity: {result['similarity_score']:.2f})",
                        expanded=(i == 1)
                    ):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Best Model", result['best_model'])
                        col_b.metric("Score", f"{result['best_score']:.4f}")
                        
                        st.markdown("**Excerpt:**")
                        st.info(result['content_preview'])
    
    except ImportError:
        st.error("RAG module not found. Please check modules/rag_system.py exists.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Semantic search tab error: {e}", exc_info=True)
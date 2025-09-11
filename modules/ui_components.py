import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import json
import warnings

# --- Local Module Imports ---
from .utils import load_data, get_sample_datasets, FeatureDetector, generate_fastapi_script, convert_df_to_csv
from .pipeline import run_pipeline, get_models_and_search_spaces, AIFeatureEngineer
from .llm_utils import (
    generate_llm_response, get_rag_expert_context, generate_dynamic_data_context, 
    get_feature_engineering_prompt, apply_ai_recommendations
)

# --- Conditional Imports for Optional Libraries ---
try:
    from scipy.stats import ks_2samp
except ImportError:
    ks_2samp = None
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- State Initialization ---
def initialize_session_state():
    """Initializes all required session state variables."""
    st.set_page_config(page_title="AI MLOps Agent", layout="wide")
    session_vars = [
        'data_loaded', 'ai_features_approved', 'feature_recommendations', 'df', 'messages', 'results_summary',
        'ai_summary', 'config', 'view', 'guide_message', 'results_data', 'file_name', 'run_history',
        'parsed_actions', 'shap_summary', 'training_data_stats', 'ai_config_rationale', 'eda_summary'
    ]
    for var in session_vars:
        if var not in st.session_state:
            if var in ['messages', 'run_history', 'parsed_actions']: st.session_state[var] = []
            elif var in ['data_loaded', 'ai_features_approved']: st.session_state[var] = False
            elif var == 'view': st.session_state[var] = 'welcome'
            elif var == 'config': st.session_state[var] = {'problem_type': 'Classification'}
            else: st.session_state[var] = None

# --- UI Action Handlers (Callbacks) ---
def handle_auto_config(df):
    """
    Handles the AI auto-configuration logic by calling the LLM and updating session state.
    """
    if not st.session_state.config.get('groq_api_key'):
        st.error("Please provide a Groq API key in the sidebar to use this feature.")
        return

    with st.spinner("AI is analyzing your dataset and configuring the pipeline..."):
        from .llm_utils import get_ai_config_prompt, generate_llm_response
        prompt = get_ai_config_prompt(df)
        response = generate_llm_response(prompt, st.session_state.config.get('groq_api_key'), is_json=True)
        try:
            ai_config = json.loads(response)
            
            new_problem_type = ai_config.get('problem_type', 'Classification')
            if st.session_state.config.get('problem_type') != new_problem_type:
                 st.session_state.config['problem_type'] = new_problem_type
                 st.session_state.config['selected_models'] = list(get_models_and_search_spaces(new_problem_type).keys())

            st.session_state.config['target'] = ai_config.get('target_column')
            st.session_state.config['numerical_features'] = ai_config.get('numerical_features')
            st.session_state.config['categorical_features'] = ai_config.get('categorical_features')
            st.session_state.ai_config_rationale = ai_config.get('rationale')

        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"AI configuration failed. The model's response could not be parsed. Details: {e}")
            st.session_state.ai_config_rationale = None

def approve_features_callback():
    """
    Callback function to apply AI-suggested features. This is the robust way to handle state updates.
    It runs BEFORE the rest of the page rerenders, ensuring UI elements have the correct default values.
    """
    st.session_state.ai_features_approved = True
    df_new = AIFeatureEngineer(st.session_state.feature_recommendations).transform(st.session_state.df)
    target = st.session_state.config.get('target')
    detected_features = FeatureDetector.detect_features(df_new, target=target)
    
    # Update the configuration in session state with the new, complete feature list
    st.session_state.config['numerical_features'] = detected_features['numerical']
    st.session_state.config['categorical_features'] = detected_features['categorical']
    st.success("Approved! The new features have been added and automatically selected in the dropdowns below.")

def reject_features_callback():
    """Callback function to reject AI-suggested features and clear them from state."""
    st.session_state.ai_features_approved = False
    st.session_state.feature_recommendations = None

# --- UI Display Functions ---
def display_welcome_page():
    """Renders the welcome page of the application."""
    st.title("AI MLOps Agent")
    st.markdown("<div style='text-align: center;'>Your intelligent partner for solving <b>tabular classification and regression problems</b>. This agent doesn't just guide, it acts.</div>", unsafe_allow_html=True)
    st.info("This application automates the end-to-end machine learning lifecycle, from data analysis to model deployment. It leverages a Large Language Model to provide intelligent recommendations, translate complex metrics into business insights, and proactively guide you toward building better models.")

    st.header("A Technical Deep Dive", divider='blue')
    tabs = st.tabs(["Intelligent Automation", "Robust ML Pipeline", "Iterative MLOps"])

    with tabs[0]:
        st.subheader("AI-Driven Insights and Actions")
        st.markdown("""
        - **AI Auto-Configuration**: Analyzes your dataset to predict problem type, target variable, and feature columns.
        - **AI Feature Engineering**: Suggests novel features to improve model performance by uncovering new data relationships.
        - **AI-Powered Chat (Hybrid RAG)**: A powerful chatbot that grounds its answers in an expert knowledge base, your live model results, and a statistical summary of your specific dataset.
        """)

    with tabs[1]:
        st.subheader("Enterprise-Grade Modeling Capabilities")
        st.markdown("""
        - **Dynamic Pipeline Construction**: Automatically builds a preprocessing pipeline tailored to your data and selections.
        - **Expanded Model Arsenal**: Supports `XGBoost`, `LightGBM`, and configurable **Ensemble Models**.
        - **Bayesian Hyperparameter Optimization**: Intelligently finds optimal model parameters in fewer iterations than grid search.
        """)

    with tabs[2]:
        st.subheader("Designed for Continuous Improvement")
        st.markdown("""
        - **Experiment Tracking (MLflow)**: Automatically logs parameters, metrics, and models for every experiment.
        - **Data Drift Monitoring**: Uses the **Kolmogorov-Smirnov (KS) test** to detect significant changes in your data's distribution over time.
        - **Automated Deployment**: Generates a production-ready `FastAPI` script to serve your trained model.
        """)

    st.markdown("---")
    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("Get Started")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            if st.session_state.df is not None:
                st.session_state.data_loaded = True
                st.session_state.view = 'configuration'
                st.rerun()
    with c2:
        st.subheader("Or Use a Sample Dataset")
        sample_datasets = get_sample_datasets()
        dataset_choice = st.selectbox("Choose a Sample Dataset", list(sample_datasets.keys()))
        if st.button(f"Load '{dataset_choice}' and Configure Pipeline"):
            st.session_state.df = sample_datasets[dataset_choice]
            st.session_state.config['problem_type'] = "Classification" if "Classification" in dataset_choice else "Regression"
            st.session_state.data_loaded = True
            st.session_state.view = 'configuration'
            st.rerun()

def display_configuration_page():
    """Renders the main MLOps pipeline configuration page."""
    st.title("MLOps Pipeline Control Panel")
    df = st.session_state.df
    
    if st.session_state.guide_message:
        st.info(st.session_state.guide_message)
        st.session_state.guide_message = None

    with st.expander("User Guide & Settings Explained", expanded=True):
        st.markdown("""
        #### **Application Workflow**
        1.  **Load Data**: Start on the Welcome page.
        2.  **AI Auto-Configuration**: Use this for a fast start. The agent will pre-fill settings.
        3.  **AI Feature Engineering (Optional)**: Let the AI suggest new features. Approve to automatically add and select them.
        4.  **Review and Fine-Tune**: Manually adjust all settings as needed.
        5.  **Run Pipeline**: Execute the full training and evaluation lifecycle.
        6.  **Analyze & Iterate**: Review results, use the AI Action Panel, and repeat the process.

        ---
        #### **Data Cleaning & Preprocessing Explained**
        This section details the automated data preparation steps that occur before model training.
        - **1. Outlier Handling**: If enabled, this step first identifies extreme values in numerical columns using the Interquartile Range (IQR) method. Any value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier and is "capped" (brought back to the boundary value). This prevents single, extreme data points from disproportionately influencing the model.
        - **2. Imputation (Handling Missing Values)**:
            -   **Numerical Imputation**: Fills missing numerical values using either the `median` (the middle value, robust to outliers), `mean` (the average), or `most_frequent` (the mode).
            -   **Categorical Imputation**: Fills missing categorical values using either the `most_frequent` (the mode, or most common category) or with a `constant` string ('missing').
        - **3. Scaling (Numerical Features)**: This step normalizes the range of numerical features. `StandardScaler` standardizes features by removing the mean and scaling to unit variance. `MinMaxScaler` scales features to a given range, typically [0, 1]. This is essential for models sensitive to the scale of input features.
        - **4. Encoding (Categorical Features)**: Machine learning models require numerical input. `OneHotEncoder` converts categorical columns into a numerical format by creating a new binary (0/1) column for each unique category.
        - **5. Feature Selection (Optional)**: If enabled, this non-AI, statistical step uses an ANOVA F-test to score the relationship between each feature and the target, keeping only the 'k' highest-scoring features.
        - **6. SMOTE (Optional for Classification)**: If enabled for imbalanced datasets, this step generates new synthetic samples of the minority class to ensure the model does not become biased towards the majority class.
        """)
    
    st.header("Step 1: AI-Powered Configuration", divider='blue')
    st.button("Auto-Configure Pipeline with AI", on_click=handle_auto_config, args=(df,), use_container_width=True, help="Uses AI to analyze your dataset and recommend initial pipeline settings.")
    if st.session_state.ai_config_rationale:
        st.info(f"**AI Configuration Rationale:** {st.session_state.ai_config_rationale}")

    df_for_config = df.copy()
    if st.session_state.ai_features_approved:
        df_for_config = AIFeatureEngineer(st.session_state.feature_recommendations).transform(df_for_config)

    st.header("Step 2: Data Transformation (Optional)", divider='blue')
    with st.container(border=True):
        st.subheader("AI Co-Pilot: Advanced Feature Engineering")
        st.markdown("Use the AI to suggest new features to add to your dataset.")
        temp_target_fe = st.selectbox("Select Target for Feature Engineering Analysis", df.columns, index=len(df.columns)-1 if df.columns is not None else 0, key="ai_target_select_fe")
        if st.button("Get AI Feature Recommendations"):
            if not st.session_state.config.get('groq_api_key'): st.error("Please provide a Groq API key in the sidebar.")
            else:
                with st.spinner("AI Agent is analyzing your data for feature ideas..."):
                    df_info = {'shape': df.shape, 'description': df.describe().to_string()}
                    feats = FeatureDetector.detect_features(df, temp_target_fe)
                    prompt = get_feature_engineering_prompt(df_info, temp_target_fe, feats['numerical'], feats['categorical'])
                    recs = generate_llm_response(prompt, st.session_state.config.get('groq_api_key'), is_json=True)
                    try: 
                        st.session_state.feature_recommendations = json.loads(recs).get("recommendations")
                    except Exception: 
                        st.error("Failed to parse AI recommendations.")
        
        if st.session_state.feature_recommendations:
            for rec in st.session_state.feature_recommendations: 
                st.info(f"**Action:** {rec.get('technique', 'N/A').capitalize()} on **`{rec.get('column', 'N/A')}`**\n\n**Rationale:** {rec.get('rationale', 'Not provided.')}")
            
            c1, c2 = st.columns(2)
            c1.button("Approve & Apply Features", on_click=approve_features_callback)
            c2.button("Reject Features", on_click=reject_features_callback)
    
    st.subheader("Processed Data Preview")
    st.dataframe(df_for_config.head())
    st.download_button(label="Download Processed Data as CSV", data=convert_df_to_csv(df_for_config), file_name="processed_data.csv", mime="text/csv")

    st.header("Step 3: Review and Fine-Tune Pipeline", divider='blue')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Problem & MLOps")
        
        def on_problem_type_change():
            problem_type = st.session_state.problem_type_selector
            st.session_state.config['problem_type'] = problem_type
            st.session_state.config['selected_models'] = list(get_models_and_search_spaces(problem_type).keys())

        st.radio("Select Problem Type", ["Classification", "Regression"], 
                 key='problem_type_selector',
                 index=["Classification", "Regression"].index(st.session_state.config.get('problem_type', 'Classification')),
                 on_change=on_problem_type_change,
                 horizontal=True)

        if MLFLOW_AVAILABLE:
            with st.container(border=True):
                st.markdown("**MLflow Experiment Tracking**")
                st.session_state.config['mlflow_uri'] = st.text_input("MLflow Tracking URI", value=st.session_state.config.get('mlflow_uri', './mlruns'))
                st.session_state.config['mlflow_experiment_name'] = st.text_input("MLflow Experiment Name", value=st.session_state.config.get('mlflow_experiment_name', 'AI_MLOps_Agent_Experiment'))
        
        st.subheader("Data & Features")
        all_cols = list(df_for_config.columns)
        default_target_val = st.session_state.config.get('target', all_cols[-1] if all_cols else None)
        default_target_idx = all_cols.index(default_target_val) if default_target_val in all_cols else len(all_cols) - 1
        target_column = st.selectbox("Select Target Variable", all_cols, index=default_target_idx)
        
        detected = FeatureDetector.detect_features(df_for_config, target=target_column)
        options_num = sorted(list(set(detected['numerical'] + detected['id'])))
        options_cat = sorted(detected['categorical'])
        
        default_num = st.session_state.config.get('numerical_features', detected['numerical'])
        default_cat = st.session_state.config.get('categorical_features', detected['categorical'])

        num_features = st.multiselect("Numerical Features", options=options_num, default=default_num)
        cat_features = st.multiselect("Categorical Features", options=options_cat, default=default_cat)

    with col2:
        st.subheader("Preprocessing & Tuning")
        test_size = st.slider("Test Set Size", 0.1, 0.5, st.session_state.config.get('test_size', 0.2), 0.05)
        cv_folds = st.number_input("Cross-Validation Folds", 2, 10, st.session_state.config.get('cv_folds', 3))
        handle_outliers = st.checkbox("Handle Numerical Outliers (IQR Method)", value=st.session_state.config.get('handle_outliers', True))
        imputation = st.selectbox("Numerical Imputation", ["median", "mean", "most_frequent"], index=0)
        cat_imputation_strategy = st.selectbox("Categorical Imputation", ["most_frequent", "constant"], index=0)
        scaler = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"], index=0)
    
    with col3:
        st.subheader("Model Selection & Feature Selection")
        use_statistical_feature_selection = st.checkbox("Use Statistical Feature Selection", value=st.session_state.config.get('use_statistical_feature_selection', False))
        k_features = 10
        total_features = len(num_features) + len(cat_features)
        if use_statistical_feature_selection and total_features > 1:
            k_features = st.slider("Number of Features to Select", 2, total_features, min(10, total_features))
        
        problem_type = st.session_state.config['problem_type']
        all_models = list(get_models_and_search_spaces(problem_type).keys())
        default_models = [model for model in st.session_state.config.get('selected_models', all_models) if model in all_models]
        selected_models = st.multiselect("Select Models to Train", options=all_models, default=default_models)
        
        if problem_type == 'Classification':
            primary_metric_display = st.selectbox("Primary Optimization Metric", ['ROC AUC', 'F1-Score'], index=0)
            metric_map = {'ROC AUC': 'roc_auc', 'F1-Score': 'f1_weighted'}
            imbalance = st.selectbox("Imbalanced Data Handling", ["None", "SMOTE"], disabled=not IMBLEARN_AVAILABLE)
        else: # Regression
            primary_metric_display = st.selectbox("Primary Optimization Metric", ['R2 Score', 'Negative MSE'], index=0)
            metric_map = {'R2 Score': 'r2', 'Negative MSE': 'neg_mean_squared_error'}
            imbalance = "None"
        
        bayes_iter = st.slider("Hyperparameter Tuning Iterations", 5, 50, st.session_state.config.get('bayes_iterations', 15))

    ensemble_model_name = 'Voting Classifier' if problem_type == 'Classification' else 'Voting Regressor'
    ensemble_base_models = []
    if ensemble_model_name in selected_models:
        with st.container(border=True):
            st.subheader("Ensemble Model Configuration")
            base_model_options = [m for m in all_models if 'Voting' not in m]
            ensemble_base_models = st.multiselect("Select Base Models for Ensemble", options=base_model_options, default=st.session_state.config.get('ensemble_base_models', [m for m in selected_models if m in base_model_options]))
    
    # This final update captures all widget states just before running the pipeline
    st.session_state.config.update({
        'target': target_column, 'numerical_features': num_features, 'categorical_features': cat_features,
        'test_size': test_size, 'cv_folds': cv_folds, 'primary_metric': metric_map[primary_metric_display], 
        'primary_metric_display': primary_metric_display, 'imbalance_method': imbalance, 
        'imputation_strategy': imputation, 'cat_imputation_strategy': cat_imputation_strategy, 
        'scaler': scaler, 'handle_outliers': handle_outliers, 
        'use_statistical_feature_selection': use_statistical_feature_selection, 'k_features': k_features,
        'selected_models': selected_models, 'bayes_iterations': bayes_iter,
        'ensemble_base_models': ensemble_base_models
    })
    
    st.markdown("---")
    if st.button("Generate EDA Report", use_container_width=True):
        display_comprehensive_eda(df_for_config, num_features, cat_features, target_column)
    
    if st.button("Run Full MLOps Lifecycle", type="primary", use_container_width=True):
        if not target_column or not (num_features or cat_features) or not selected_models:
            st.error("Please configure all required settings: Target Variable, Features, and at least one Model.")
        else:
            st.session_state.results_data = run_pipeline(st.session_state.df, st.session_state.config)
            st.session_state.view = 'results'
            st.rerun()

def display_results_page():
    """Renders the results page after a pipeline run."""
    st.title("MLOps Pipeline Results")
    if st.button("<< Back to Main Configuration"):
        st.session_state.view = 'configuration'
        st.rerun()

    if not st.session_state.results_data:
        st.error("No results found. Please run the pipeline.")
        return

    results_df, le, X_train, y_train, X_test, y_test = st.session_state.results_data
    st.header("Model Results & Interpretation", divider='blue')
    problem_type = st.session_state.config['problem_type']
    primary_metric_col = st.session_state.config['primary_metric_display']
    
    results_df_sorted = results_df.sort_values(by=primary_metric_col, ascending=False).reset_index(drop=True)
    best_model_row = results_df_sorted.iloc[0]
    best_model = best_model_row['model_object']
    st.session_state.results_summary = results_df_sorted.drop(columns='model_object').to_string()
    
    run_summary = {"Run": len(st.session_state.run_history) + 1, "Best Model": best_model_row['Model'], primary_metric_col: best_model_row[primary_metric_col]}
    if not any(d['Run'] == run_summary['Run'] for d in st.session_state.run_history):
        st.session_state.run_history.append(run_summary)

    tabs = st.tabs(["Run Comparison", "AI Summary & Actions", "Performance Metrics", "Business Impact", "Drift Monitoring", "Model Explainability (SHAP)", "Deployment"])

    with tabs[0]:
        st.subheader("Experiment Comparison Dashboard")
        if len(st.session_state.run_history) > 0:
            history_df = pd.DataFrame(st.session_state.run_history).set_index("Run")
            st.dataframe(history_df.style.highlight_max(axis=0, color='#d4edda', subset=[primary_metric_col]))
            if len(history_df) > 1:
                fig = px.line(history_df, y=primary_metric_col, title=f"Trend of {primary_metric_col} Across Runs", markers=True)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("AI Executive Summary")
        summary_prompt = f"""Act as a data scientist. Summarize these results for a '{problem_type}' problem predicting '{st.session_state.config['target']}'. The primary metric was '{primary_metric_col}'. Results:\n{st.session_state.results_summary}\nProvide a summary, interpretation, and actionable recommendations in Markdown."""
        summary = generate_llm_response(summary_prompt, st.session_state.config.get('groq_api_key'))
        st.session_state.ai_summary = summary
        st.markdown(summary)
        
        parser_prompt = f"""Analyze the following text and extract actionable recommendations. The 'type' must be one of the following exact keywords: 'increase_tuning', 'try_smote', 'try_feature_selection'.
        Example: "Enhance GradientBoosting's performance, consider experimenting with different hyperparameters" -> {{"action_text": "Increase Hyperparameter Tuning Iterations", "type": "increase_tuning"}}
        Return ONLY a single JSON object with a key "actions" containing a list of these structured objects. Text to analyze: {st.session_state.ai_summary}"""
        parsed_actions_str = generate_llm_response(parser_prompt, st.session_state.config.get('groq_api_key'), is_json=True)
        try: st.session_state.parsed_actions = json.loads(parsed_actions_str).get("actions", [])
        except (json.JSONDecodeError, AttributeError): st.session_state.parsed_actions = []

        if st.session_state.parsed_actions:
            with st.container(border=True):
                st.subheader("AI Action Panel")
                st.markdown("The AI Agent suggests the following improvements. Click to apply these settings and return to the configuration page.")
                for action in st.session_state.parsed_actions:
                    st.info(f"**Recommendation:** {action.get('action_text', 'N/A')}")
                st.button("Apply Recommendations & Return to Configuration", on_click=apply_ai_recommendations)
    
    with tabs[2]:
        st.subheader("Performance Metrics")
        st.dataframe(results_df_sorted.drop(columns='model_object').style.highlight_max(axis=0, color='#d4edda', subset=[primary_metric_col]))
        y_pred = best_model.predict(X_test)
        if problem_type == 'Classification':
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=le.classes_, y=le.classes_, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else: # Regression
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Actual vs. Predicted Values', trendline='ols', trendline_color_override='red')
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Business Impact Simulator")
        if problem_type == 'Classification' and len(np.unique(y_train)) == 2:
            st.info("Adjust the classification threshold to see how it impacts business metrics.")
            threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adjusted).ravel()
                st.metric("Customers Targeted for Retention (TP + FP)", f"{tp+fp}")
                st.metric("Potential Churners Missed (FN)", f"{fn}", delta=f"{-fn}", delta_color="inverse")
                st.metric("Loyal Customers Incorrectly Targeted (FP)", f"{fp}", delta=f"{-fp}", delta_color="inverse")
        else:
            st.info("Business impact simulator is only available for binary classification problems.")
            
    with tabs[4]:
        display_drift_monitoring_tab()

    with tabs[5]:
        st.subheader("Model Explainability (SHAP)")
        if SHAP_AVAILABLE:
            with st.spinner("Calculating SHAP values..."):
                try:
                    if 'Voting' in best_model_row['Model']:
                        st.warning("SHAP plots for Voting ensembles are not directly supported.")
                    else:
                        model_key = 'classifier' if problem_type == 'Classification' else 'regressor'
                        explainer_model = best_model.named_steps.get(model_key)
                        transformation_pipeline = Pipeline(steps=best_model.steps[:-1])
                        X_test_transformed = transformation_pipeline.transform(X_test)
                        
                        try: feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
                        except Exception: feature_names = None

                        X_test_transformed_df = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed, columns=feature_names)
                        
                        explainer = shap.Explainer(explainer_model, X_test_transformed_df)
                        shap_values = explainer(X_test_transformed_df)
                        
                        fig, ax = plt.subplots()
                        plt.title(f'SHAP Feature Impact for {best_model_row["Model"]}')
                        shap.summary_plot(shap_values, X_test_transformed_df, show=False)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate SHAP plot. Error: {e}")

    with tabs[6]:
        st.subheader("Deployment Assets")
        st.info("Download the best model and a ready-to-use FastAPI script for deployment.")
        c1, c2 = st.columns(2)
        with c1:
            pkl_model = pickle.dumps(best_model)
            st.download_button(label="Download Best Model (.pkl)", data=pkl_model, file_name="best_model.pkl")
        with c2:
            fastapi_script = generate_fastapi_script(st.session_state.config['numerical_features'], st.session_state.config['categorical_features'], problem_type)
            st.download_button(label="Download FastAPI Script (.py)", data=fastapi_script, file_name="fastapi_app.py")

def display_comprehensive_eda(df, numerical_features, categorical_features, target):
    """Renders the multi-tab Exploratory Data Analysis section."""
    st.header("Comprehensive Exploratory Data Analysis", divider='violet')
    eda_tabs = st.tabs(["Data Quality Audit", "Numerical Feature Analysis", "Categorical Feature Analysis"])
    eda_summary_parts = []
    
    with eda_tabs[0]:
        st.subheader("Data Preview"); st.dataframe(df.head(10))
        eda_summary_parts.append(f"Data Preview shows the first 10 rows. Columns are: {', '.join(df.columns)}.")
        st.subheader("Missing Values Analysis")
        missing_df = df.isnull().sum().to_frame('Missing Values')
        missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
        if not missing_df.empty:
            st.dataframe(missing_df); fig = px.bar(missing_df, x=missing_df.index, y='Percentage', title='Percentage of Missing Values'); st.plotly_chart(fig, use_container_width=True)
            eda_summary_parts.append(f"Missing Values found in: {', '.join(missing_df.index)}.")
        else:
            st.success("No missing values found."); eda_summary_parts.append("No missing values found.")
        st.subheader("Data Types"); st.dataframe(df.dtypes.astype(str).to_frame(name='Data Type'))

    with eda_tabs[1]:
        st.subheader("Descriptive Statistics")
        if numerical_features:
            desc_stats = df[numerical_features].describe(); st.dataframe(desc_stats)
            eda_summary_parts.append("Descriptive stats were generated.")
        else: st.info("No numerical features selected.")
        
        st.subheader("Target Distribution")
        if target and target in df.columns:
            fig = px.histogram(df, x=target, title=f'Distribution of Target: {target}'); st.plotly_chart(fig, use_container_width=True)
            eda_summary_parts.append(f"Distribution of target '{target}' was plotted.")
        
        st.subheader("Correlation Heatmap")
        if numerical_features and len(numerical_features) > 1:
            corr_matrix = df[numerical_features + ([target] if target in df.columns else [])].corr(numeric_only=True)
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r'); st.plotly_chart(fig, use_container_width=True)
            eda_summary_parts.append("A correlation heatmap was generated.")

    with eda_tabs[2]:
        st.subheader("Categorical Feature Analysis")
        if not categorical_features: st.info("No categorical features selected.")
        else:
            for col in categorical_features:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}', color=target if target in df.columns else None, barmode='group'); st.plotly_chart(fig, use_container_width=True)
            eda_summary_parts.append(f"Distributions for categorical features were plotted.")

    st.session_state.eda_summary = " ".join(eda_summary_parts)

def display_drift_monitoring_tab():
    """Renders the Data Drift monitoring tab."""
    st.subheader("Data Drift Monitoring")
    if not st.session_state.get('training_data_stats'):
        st.warning("Run a pipeline first to establish a baseline for drift detection.")
        return
    if not ks_2samp:
        st.error("Scipy library not found. Please install it (`pip install scipy`) for drift monitoring.")
        return

    st.info("Upload a new dataset to compare its distribution against the training data.")
    new_data_file = st.file_uploader("Upload New Data for Drift Analysis", type=['csv'], key="drift_uploader")

    if new_data_file:
        new_df = load_data(new_data_file)
        st.dataframe(new_df.head())
        baseline_stats = st.session_state.training_data_stats
        
        st.subheader("Numerical Feature Drift")
        for col in st.session_state.config['numerical_features']:
            if col in new_df.columns and col in baseline_stats['X_train_ref'].columns:
                baseline_series = baseline_stats['X_train_ref'][col].dropna()
                new_series = new_df[col].dropna()
                
                if len(baseline_series) > 1 and len(new_series) > 1:
                    ks_stat, p_value = ks_2samp(baseline_series, new_series)
                    st.markdown(f"**Feature: `{col}`**")
                    if p_value < 0.05: st.error(f"Drift Detected! (Kolmogorov-Smirnov p-value: {p_value:.4f})")
                    else: st.success(f"No significant drift detected. (p-value: {p_value:.4f})")
                    
                    fig_data = pd.DataFrame({'value': pd.concat([baseline_series, new_series]), 'source': ['Baseline'] * len(baseline_series) + ['New Data'] * len(new_series)})
                    fig = px.histogram(fig_data, x='value', color='source', barmode='overlay', opacity=0.7, title=f"Distribution Comparison for '{col}'"); st.plotly_chart(fig, use_container_width=True)

def display_sidebar_chat():
    """Renders the AI Co-Pilot chat interface in the sidebar."""
    with st.sidebar:
        st.title("AI Co-Pilot")
        st.write("Ask questions about your data, model, or results. Answers are grounded in MLOps best practices and your specific context.")
        st.session_state.config['groq_api_key'] = st.text_input("Groq API Key", type="password", key="sidebar_groq_api_key", value=st.session_state.config.get('groq_api_key', ''))
        st.markdown("---")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("Ask about your data, model, or results..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Agent is thinking..."):
                    expert_context = get_rag_expert_context()
                    results_context = st.session_state.ai_summary or "No model has been run yet."
                    data_context = generate_dynamic_data_context(st.session_state.df, st.session_state.config, st.session_state.eda_summary)
                    
                    rag_prompt = f"""You are an expert AI business analyst. Your goal is to translate ML results and data into actionable business insights.
                    --- START EXPERT KNOWLEDGE BASE ---
                    {expert_context}
                    --- END EXPERT KNOWLEDGE BASE ---
                    --- START DYNAMIC DATASET & EDA CONTEXT ---
                    {data_context}
                    --- END DYNAMIC DATASET & EDA CONTEXT ---
                    --- START CURRENT EXPERIMENT RESULTS ---
                    - Problem Type: {st.session_state.config.get('problem_type', 'Not specified')}
                    - Model Performance Summary: {results_context}
                    --- END CURRENT EXPERIMENT RESULTS ---
                    User's Question: {prompt}
                    Your Answer (Synthesize all context to provide a thorough, business-focused answer):"""
                    
                    response = generate_llm_response(rag_prompt, st.session_state.config.get('groq_api_key'))
                    message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


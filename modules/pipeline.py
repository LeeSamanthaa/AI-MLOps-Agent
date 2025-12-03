# modules/pipeline.py
# FIX: Removed the ProgressTracker class and all related logic to resolve the 'cannot pickle _thread.lock' error.
# Progress updates are now handled in the main thread after parallel execution completes.

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import time
import logging
from joblib import Parallel, delayed
from collections import Counter
import os
import json

# --- Core ML & Data Libraries ---
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor, VotingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# --- Advanced MLOps & Modeling Libraries ---
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import custom transformers from their dedicated, stable module
from .transformers import SafeLabelEncoder, OutlierHandler, AIFeatureEngineer

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FIX: The ProgressTracker class has been completely removed to prevent pickling errors.

# --- Core Pipeline Logic ---

def apply_manual_hyperparameters(model, model_name, config):
    """
    Apply user-specified manual hyperparameters to a model if manual tuning is enabled.

    Args:
        model: The sklearn model instance
        model_name: Name of the model (e.g., 'LogisticRegression')
        config: The session config dictionary

    Returns:
        model: The model with updated hyperparameters
    """
    if not config.get('manual_tuning_enabled', False):
        return model

    manual_params = config.get('manual_params', {}).get(model_name, {})
    if not manual_params:
        logging.info(f"No manual parameters found for {model_name}")
        return model

    # Create a new model instance to avoid modifying the original
    new_model = clone(model)
    
    # Filter for parameters that actually exist on the model
    valid_params = {p: v for p, v in manual_params.items() if hasattr(new_model, p)}
    
    if valid_params:
        try:
            new_model.set_params(**valid_params)
            logging.info(f"Applied manual hyperparameters to {model_name}: {valid_params}")
        except Exception as e:
            logging.warning(f"Could not set manual params for {model_name}. Error: {e}")
            return model # Return original model on failure
    
    return new_model

def get_models_and_search_spaces(problem_type='Classification'):
    """Returns a dictionary of models and their corresponding Bayesian search spaces."""
    models = {}
    if problem_type == 'Classification':
        models = {
            'LogisticRegression': {'model': LogisticRegression(random_state=42, max_iter=1000)},
            'RandomForest': {'model': RandomForestClassifier(random_state=42)},
            'GradientBoosting': {'model': GradientBoostingClassifier(random_state=42)},
        }
        if IMBLEARN_AVAILABLE: models['BalancedRandomForest'] = {'model': BalancedRandomForestClassifier(random_state=42)}
        if XGB_AVAILABLE: models['XGBoost'] = {'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')}
        if LGBM_AVAILABLE: models['LightGBM'] = {'model': LGBMClassifier(random_state=42)}
        if len(models) > 1: models['Voting Classifier'] = {'model': None}
    else:  # Regression
        models = {
            'LinearRegression': {'model': LinearRegression()},
            'RandomForest': {'model': RandomForestRegressor(random_state=42)},
            'GradientBoosting': {'model': GradientBoostingRegressor(random_state=42)},
        }
        if XGB_AVAILABLE: models['XGBoost'] = {'model': XGBRegressor(random_state=42)}
        if LGBM_AVAILABLE: models['LightGBM'] = {'model': LGBMRegressor(random_state=42)}
        if len(models) > 1: models['Voting Regressor'] = {'model': None}
    
    for name, info in models.items():
        if 'Voting' in name or info.get('model') is None:
            continue
        key = 'classifier' if problem_type == 'Classification' else 'regressor'
        if 'LogisticRegression' in name: info['params_bayes'] = {f'{key}__C': Real(1e-3, 1e+2, prior='log-uniform')}
        elif 'RandomForest' in name: info['params_bayes'] = {f'{key}__n_estimators': Integer(100, 500), f'{key}__max_depth': Integer(10, 100)}
        elif 'GradientBoosting' in name: info['params_bayes'] = {f'{key}__n_estimators': Integer(100, 300), f'{key}__learning_rate': Real(0.01, 0.3)}
        elif 'XGBoost' in name: info['params_bayes'] = {f'{key}__n_estimators': Integer(100, 500), f'{key}__learning_rate': Real(0.01, 0.3), f'{key}__max_depth': Integer(3, 15)}
        elif 'LightGBM' in name: info['params_bayes'] = {f'{key}__n_estimators': Integer(100, 500), f'{key}__learning_rate': Real(0.01, 0.3), f'{key}__num_leaves': Integer(20, 100)}

    return models

def suggest_base_models(results_df, metric):
    """Suggests top 2-3 diverse models for an ensemble."""
    if results_df.empty:
        return []
    
    top_performers = results_df.sort_values(by=metric, ascending=False)
    suggestions = top_performers['Model'].head(3).tolist()
    
    return suggestions

# --- SHAP Integration ---

# Import SHAP within the file for dependency check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_shap_values(model, X_test):
    """
    Calculates SHAP values for the trained model.
    This safely extracts the final estimator from the pipeline and transforms the data.
    """
    if not SHAP_AVAILABLE:
        logging.warning("SHAP is not available. Skipping calculation.")
        return None, None
    
    # Extract the final estimator and preprocessed data
    try:
        # 1. Get the preprocessor step
        preprocessor = model.named_steps['preprocessor']
        
        # 2. Transform the test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # 3. Handle features for column names (Get feature names after one-hot encoding)
        feature_names = preprocessor.get_feature_names_out(X_test.columns)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        # 4. Get the final estimator (e.g., LogisticRegression, XGBoost, etc.)
        final_estimator = None
        for step_name in model.named_steps:
            if step_name not in ['preprocessor', 'feature_selection', 'smote', 'ensemble']:
                final_estimator = model.named_steps[step_name]
                break
        
        if final_estimator is None:
            # Handle Voting Classifier/Regressor where the name is 'ensemble'
            if 'ensemble' in model.named_steps:
                final_estimator = model.named_steps['ensemble']
            else:
                logging.error("Could not find final estimator in pipeline steps.")
                return None, None

        # 5. Calculate SHAP values
        model_name = final_estimator.__class__.__name__
        if 'RandomForest' in model_name or 'XGBoost' in model_name or 'LightGBM' in model_name:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(X_test_df)
            
            # For multi-class classification, shap_values is a list of arrays.
            # Use the first class's values or average them for the summary plot.
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # We often use the second class (index 1) for binary classification importance
                # or average the absolute values for multi-class global importance.
                # Returning the primary output array:
                shap_output = shap_values[1] if len(shap_values) == 2 else np.abs(np.array(shap_values)).mean(axis=0)
            else:
                 shap_output = shap_values
            
        elif hasattr(final_estimator, 'predict_proba') or hasattr(final_estimator, 'predict'):
            # Use KernelExplainer or other appropriate explainer for non-tree models
            # Since KernelExplainer is very slow, we recommend using a smaller sample of data
            # For simplicity, we use Explainer and rely on the model type detection
            try:
                # Use a small sample for KernelExplainer if necessary
                sample_data = X_test_df.head(100)
                explainer = shap.Explainer(final_estimator, sample_data)
                shap_output = explainer(X_test_df)
            except Exception as e:
                logging.error(f"Failed using generic SHAP Explainer: {e}")
                return None, None
        else:
            logging.warning("SHAP calculation skipped: Model type not supported or recognized.")
            return None, None
        
        # Ensure we return the SHAP Values object and the data used
        return shap_output, X_test_df.reset_index(drop=True)

    except Exception as e:
        logging.error(f"Error calculating SHAP values: {e}", exc_info=True)
        return None, None


# FIX: Modified function signature to remove the 'progress_tracker' parameter.
def train_single_model(name, model_info, X_train, y_train, X_test, y_test, preprocessor, config, run_id):
    """Function to train a single model, designed to be run in parallel."""
    try:
        problem_type = config['problem_type']
        key = 'classifier' if problem_type == 'Classification' else 'regressor'
        
        if 'Voting' in name:
            base_models_for_ensemble = config.get('ensemble_base_models', [])
            all_models_info = get_models_and_search_spaces(problem_type)
            estimators = [(m_name, clone(all_models_info[m_name]['model'])) for m_name in base_models_for_ensemble if m_name in all_models_info]
            
            if not estimators:
                logging.warning(f"Skipping {name}: No valid base models selected.")
                return None
            
            Ensemble = VotingClassifier if problem_type == 'Classification' else VotingRegressor
            ensemble_params = {'estimators': estimators}
            if problem_type == 'Classification':
                ensemble_params['voting'] = 'soft'
            
            model = Ensemble(**ensemble_params)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ensemble', model)])
            pipeline.fit(X_train, y_train)
            best_model, best_score = pipeline, 0.0
        else:
            steps = [('preprocessor', preprocessor)]
            if config.get('use_statistical_feature_selection'):
                k = min(config.get('k_features', 10), X_train.shape[1])
                if k < 1:
                    logging.warning(f"Skipping feature selection for {name} as k={k} is less than 1.")
                else:
                    selector = SelectKBest(f_classif if problem_type == 'Classification' else f_regression, k=k)
                    steps.append(('feature_selection', selector))
            
            if problem_type == 'Classification' and config.get('imbalance_method') == 'SMOTE' and IMBLEARN_AVAILABLE and 'Balanced' not in name:
                steps.append(('smote', SMOTE(random_state=42)))
            
            model_instance = apply_manual_hyperparameters(
                clone(model_info['model']), 
                name, 
                config
            )
            steps.append((key, model_instance))
            pipeline = ImbPipeline(steps=steps)

            is_manual_tuning = config.get('manual_tuning_enabled', False) and name in config.get('manual_params', {})
            
            if is_manual_tuning:
                manual_params = config['manual_params'][name]
                manual_params = {k: v for k, v in manual_params.items() if v is not None}
                prefixed_params = {f"{key}__{p_name}": p_val for p_name, p_val in manual_params.items()}
                
                pipeline.set_params(**prefixed_params)
                pipeline.fit(X_train, y_train)
                best_model, best_score = pipeline, 0.0
            elif config.get('manual_tuning_enabled', False):
                pipeline.fit(X_train, y_train)
                best_model, best_score = pipeline, 0.0
            elif SKOPT_AVAILABLE and 'params_bayes' in model_info:
                bayes_iter = 5 if config.get('quick_test') else config['bayes_iterations']
                cv_folds = 2 if config.get('quick_test') else config['cv_folds']
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if problem_type == 'Classification' else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                opt = BayesSearchCV(pipeline, model_info['params_bayes'], n_iter=bayes_iter, cv=cv, scoring=config['primary_metric'], random_state=42, n_jobs=1)
                opt.fit(X_train, y_train)
                best_model, best_score = opt.best_estimator_, opt.best_score_
            else:
                pipeline.fit(X_train, y_train)
                best_model, best_score = pipeline, 0.0
        
        model_id = f"{run_id}-{name}".replace(" ", "_")
        
        y_pred = best_model.predict(X_test)
        metrics = {'Model': name, 'Best_Score_CV': best_score, 'model_id': model_id}
        
        if problem_type == 'Classification':
            y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            metrics.update({'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0), 'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)})
            if y_prob is not None:
                avg = 'weighted' if len(np.unique(y_train)) > 2 else 'binary'
                metrics['F1-Score'] = f1_score(y_test, y_pred, average=avg, zero_division=0)
                if len(np.unique(y_train)) > 2:
                    metrics['ROC_AUC'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                else:
                    metrics['ROC_AUC'] = roc_auc_score(y_test, y_prob[:, 1])
        else:
            metrics.update({'R2_Score': r2_score(y_test, y_pred), 'MSE': mean_squared_error(y_test, y_pred), 'MAE': mean_absolute_error(y_test, y_pred)})
        
        return (model_id, best_model, metrics)
    except ValueError as e:
        logging.error(f"Data-related error training model {name}: {e}", exc_info=True)
        st.error(f"Error training {name}: Check if your data, especially the target variable, is suitable for the selected model.")
        return None
    except TypeError as e:
        logging.error(f"Type error training model {name}: {e}", exc_info=True)
        st.error(f"Error training {name}: A data type mismatch may have occurred. Ensure features are correctly identified as numerical/categorical.")
        return None
    except Exception as e:
        logging.error(f"A general error occurred while training model {name}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while training {name}. Check logs for details.")
        return None
    # FIX: Removed the 'finally' block that called the progress tracker.

@st.cache_data(show_spinner=False)
def run_pipeline(_df, config_hash, _progress_placeholder, _model_store, phase='benchmark'):
    """Executes the ML pipeline with parallel processing and phase-awareness."""
    config = st.session_state.config
    run_timestamp = int(time.time())
    
    progress_bar = _progress_placeholder.progress(0, text="Initializing pipeline...")

    run = None
    if MLFLOW_AVAILABLE and config.get('mlflow_uri'):
        try:
            if mlflow.active_run():
                mlflow.end_run()
            mlflow.set_tracking_uri(config['mlflow_uri'])
            mlflow.set_experiment(config.get('mlflow_experiment_name', 'AI_MLOps_Agent_Experiment'))
            run = mlflow.start_run()
            mlflow.log_params({k: v for k, v in config.items() if k != 'groq_api_key'})
        except Exception as e:
            st.warning(f"Could not connect to MLflow tracking server. Running without tracking. Error: {e}")
            logging.warning(f"MLflow connection failed: {e}")
            run = None
    
    run_id = run.info.run_id if run else f'local_{run_timestamp}'

    try:
        progress_bar.progress(0, text="Initializing pipeline...")
        df, problem_type, target = _df.copy(), config['problem_type'], config['target']
        
        all_features = [f for f in config['numerical_features'] + config['categorical_features'] if f in df.columns]
        X, y_raw = df[all_features], df[target]
        not_na_mask = y_raw.notna()
        X, y_raw = X.loc[not_na_mask], y_raw[not_na_mask]
        
        progress_bar.progress(5, text="Encoding target variable...")
        le = SafeLabelEncoder() if problem_type == 'Classification' else None
        y = le.fit(y_raw).transform(y_raw) if le else y_raw
        
        progress_bar.progress(10, text="Splitting data into training and test sets...")
        stratify = y if problem_type == 'Classification' and pd.Series(y).nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42, stratify=stratify)
        st.session_state.training_data_stats = {'X_train_ref': X_train, 'X_test_ref': X_test, 'y_test_ref': y_test, 'le_ref': le}
        
        progress_bar.progress(15, text="Building preprocessing pipeline...")
        num_features = [col for col in X_train.select_dtypes(include=np.number).columns.tolist() if col in config['numerical_features']]
        cat_features = [col for col in X_train.select_dtypes(exclude=np.number).columns.tolist() if col in config['categorical_features']]
        
        num_transformer_steps = [('imputer', SimpleImputer(strategy=config['imputation_strategy'])), ('scaler', StandardScaler() if config['scaler'] == 'StandardScaler' else MinMaxScaler())]
        if config['handle_outliers']:
            num_transformer_steps.insert(1, ('outlier_handler', OutlierHandler()))
        num_transformer = Pipeline(steps=num_transformer_steps)
        
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first' if problem_type == 'Classification' else None))])
        
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)], remainder='passthrough')
        
        all_models = get_models_and_search_spaces(problem_type)
        
        if phase == 'benchmark':
            models_to_run = {name: info for name, info in all_models.items() if 'Voting' not in name and name in config['selected_models']}
        else:
            model_name = 'Voting Classifier' if problem_type == 'Classification' else 'Voting Regressor'
            models_to_run = {model_name: all_models[model_name]}

        logging.info(f"Starting pipeline run for phase '{phase}' with models: {list(models_to_run.keys())}")
        progress_bar.progress(20, text=f"Training {len(models_to_run)} model(s) in parallel...")
        
        # FIX: Removed ProgressTracker and callback logic.
        # FIX: Modified the delayed() call to remove the progress_tracker argument.
        results_list = Parallel(n_jobs=-1)(
            delayed(train_single_model)(
                name, info, X_train, y_train, X_test, y_test, preprocessor, config, run_id
            )
            for name, info in models_to_run.items()
        )
        
        # FIX: Update progress bar in the main thread AFTER all parallel jobs are complete.
        progress_bar.progress(85, text="All models trained. Consolidating results...")
        
        results = []
        for result in results_list:
            if result:
                model_id, best_model, metrics = result
                _model_store.add_model(model_id, best_model)
                results.append(metrics)
        
        progress_bar.progress(90, text="Finalizing results...")
        
        if not results:
            st.error("No models were successfully trained. Please check your configuration, data, and logs for errors.")
            _progress_placeholder.empty()
            return None, None, None, None, None, None

        results_df = pd.DataFrame(results).rename(columns={'Best_Score_CV': 'Best Score (CV)', 'ROC_AUC': 'ROC AUC', 'F1-Score': 'F1-Score', 'R2_Score': 'R2 Score'})
        
        if results_df.empty:
            st.error("Could not generate a results summary because all models failed during training or evaluation.")
            _progress_placeholder.empty()
            return None, None, None, None, None, None

        best_model_row = results_df.sort_values(by=config['primary_metric_display'], ascending=False).iloc[0]
        if run:
            progress_bar.progress(95, text="Logging best model to MLflow...")
            best_model_obj = _model_store.get_model(best_model_row['model_id'])
            if best_model_obj:
                try:
                    mlflow.sklearn.log_model(best_model_obj, "best_model")

                    if st.session_state.get('feature_recommendations'):
                        feature_artifact_path = "feature_engineering_config.json"
                        with open(feature_artifact_path, 'w') as f:
                            json.dump(st.session_state.feature_recommendations, f, indent=2)
                        mlflow.log_artifact(feature_artifact_path)
                        os.remove(feature_artifact_path)
                        logging.info("Logged feature engineering config to MLflow")
                except Exception as e:
                    st.warning(f"Failed to log model to MLflow: {e}")
                    logging.warning(f"MLflow model logging failed: {e}")

        progress_bar.progress(100, text="Pipeline complete!")
        time.sleep(1)
        _progress_placeholder.empty()

        # Save experiment to history for comparison
        try:
            from .utils import save_experiment_run
            save_experiment_run(config, results_df, run_id)
            logging.info(f"Experiment {run_id} saved to history")
        except Exception as e:
            logging.warning(f"Could not save experiment to history: {e}")
                    # Try to add experiment to RAG (completely safe - won't crash)
        try:
            from .rag_system import add_experiment_to_rag
            experiment_id = f"exp_{run_id}_{phase}"
            ai_summary = st.session_state.get('ai_summary', '')
            add_experiment_to_rag(experiment_id, results_df, config, ai_summary)
        except:
            pass  # RAG not available, that's fine
        # ===== STOP =====
        
        return results_df, le, X_train, y_train, X_test, y_test
    
    
    except Exception as e:
        st.error(f"A critical error occurred in the MLOps pipeline: {e}")
        logging.critical(f"MLOps pipeline failed: {e}", exc_info=True)
        if _progress_placeholder: _progress_placeholder.empty()
        return None, None, None, None, None, None
    finally:
        if run and mlflow.active_run():
            mlflow.end_run()
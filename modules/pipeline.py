import streamlit as st
import pandas as pd
import numpy as np
import warnings
import mlflow

# --- Core ML & Data Libraries ---
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor, VotingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# --- Advanced MLOps Libraries ---
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
    from sklearn.pipeline import Pipeline as ImbPipeline # Fallback
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

warnings.filterwarnings('ignore')

# --- Custom Transformers ---
class OutlierHandler(BaseEstimator, TransformerMixin):
    """A transformer to handle outliers by capping them at a specified IQR factor."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_numeric = X.select_dtypes(include=np.number)
        for col in X_numeric.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_numeric = X_copy.select_dtypes(include=np.number)
        for col in X_numeric.columns:
            if col in self.lower_bounds_:
                X_copy[col] = X_copy[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_copy

class AIFeatureEngineer(BaseEstimator, TransformerMixin):
    """A transformer to apply feature engineering techniques recommended by the AI."""
    def __init__(self, recommendations=None):
        self.recommendations = recommendations if recommendations is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not self.recommendations:
            return X_transformed
        for rec in self.recommendations:
            try:
                technique = rec.get("technique")
                column = rec.get("column")
                if column not in X_transformed.columns:
                    continue
                if technique == "binning":
                    X_transformed[f"{column}_binned"] = pd.qcut(X_transformed[column], q=rec.get("bins", 4), labels=False, duplicates='drop')
                elif technique == "polynomial":
                    poly = PolynomialFeatures(degree=rec.get("degree", 2), include_bias=False)
                    poly_feats = poly.fit_transform(X_transformed[[column]])
                    for i in range(1, poly_feats.shape[1]):
                        X_transformed[f'{column}_pow{i+1}'] = poly_feats[:, i]
                elif technique == "log_transform":
                    X_transformed[f"{column}_log"] = np.log1p(X_transformed[column])
                elif technique == "interaction":
                    other_col = rec.get("other_column")
                    if other_col and other_col in X_transformed.columns:
                        X_transformed[f"{column}_{other_col}_interaction"] = X_transformed[column] * X_transformed[other_col]
            except Exception as e:
                st.warning(f"Could not apply '{technique}' on '{column}'. Error: {e}")
        return X_transformed

# --- Core Pipeline Logic ---
def get_models_and_search_spaces(problem_type='Classification'):
    """Returns a dictionary of models and their hyperparameter search spaces."""
    models = {}
    if problem_type == 'Classification':
        models = {
            'LogisticRegression': {'model': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)},
            'RandomForest': {'model': RandomForestClassifier(random_state=42)},
            'GradientBoosting': {'model': GradientBoostingClassifier(random_state=42)},
            'Voting Classifier': {'model': None}
        }
        if IMBLEARN_AVAILABLE: models['BalancedRandomForest'] = {'model': BalancedRandomForestClassifier(random_state=42)}
        if XGB_AVAILABLE: models['XGBoost'] = {'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')}
        if LGBM_AVAILABLE: models['LightGBM'] = {'model': LGBMClassifier(random_state=42)}
        
        for name in models:
            if 'LogisticRegression' in name: models[name]['params_bayes'] = {'classifier__C': Real(1e-3, 1e+2, prior='log-uniform')}
            elif 'RandomForest' in name: models[name]['params_bayes'] = {'classifier__n_estimators': Integer(50, 250), 'classifier__max_depth': Integer(5, 50)}
            elif 'GradientBoosting' in name: models[name]['params_bayes'] = {'classifier__n_estimators': Integer(50, 200), 'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform')}
            elif 'XGBoost' in name: models[name]['params_bayes'] = {'classifier__n_estimators': Integer(50, 250), 'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'), 'classifier__max_depth': Integer(3, 10)}
            elif 'LightGBM' in name: models[name]['params_bayes'] = {'classifier__n_estimators': Integer(50, 250), 'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'), 'classifier__num_leaves': Integer(20, 50)}
    else: # Regression
        models = {
            'LinearRegression': {'model': LinearRegression()},
            'RandomForest': {'model': RandomForestRegressor(random_state=42)},
            'GradientBoosting': {'model': GradientBoostingRegressor(random_state=42)},
            'Voting Regressor': {'model': None}
        }
        if XGB_AVAILABLE: models['XGBoost'] = {'model': XGBRegressor(random_state=42, objective='reg:squarederror')}
        if LGBM_AVAILABLE: models['LightGBM'] = {'model': LGBMRegressor(random_state=42)}

        model_key = 'regressor'
        for name in models:
            if 'RandomForest' in name: models[name]['params_bayes'] = {f'{model_key}__n_estimators': Integer(50, 250), f'{model_key}__max_depth': Integer(5, 50)}
            elif 'GradientBoosting' in name: models[name]['params_bayes'] = {f'{model_key}__n_estimators': Integer(50, 200), f'{model_key}__learning_rate': Real(0.01, 0.3, prior='log-uniform')}
            elif 'XGBoost' in name: models[name]['params_bayes'] = {f'{model_key}__n_estimators': Integer(50, 250), f'{model_key}__learning_rate': Real(0.01, 0.3, prior='log-uniform'), 'classifier__max_depth': Integer(3, 10)}
            elif 'LightGBM' in name: models[name]['params_bayes'] = {f'{model_key}__n_estimators': Integer(50, 250), f'{model_key}__learning_rate': Real(0.01, 0.3, prior='log-uniform'), f'{model_key}__num_leaves': Integer(20, 50)}
    return models

@st.cache_resource(show_spinner="Running the full MLOps pipeline...")
def run_pipeline(_df, _config):
    """
    Executes the end-to-end machine learning pipeline based on the provided configuration.
    This includes preprocessing, model training, hyperparameter tuning, and evaluation.
    """
    config = _config.copy()
    
    if mlflow.active_run():
        mlflow.end_run()

    run = None
    try:
        # MLflow Setup
        if config.get('mlflow_uri'):
            try:
                mlflow.set_tracking_uri(config['mlflow_uri'])
                mlflow.set_experiment(config.get('mlflow_experiment_name', 'AI_MLOps_Agent_Experiment'))
                run = mlflow.start_run(run_name=f"Run_{len(st.session_state.run_history)+1}")
                loggable_params = {k: v for k, v in config.items() if k not in ['groq_api_key', 'ensemble_base_models'] and isinstance(v, (str, int, float, bool))}
                mlflow.log_params(loggable_params)
            except Exception as e:
                st.warning(f"Could not connect to MLflow Tracking URI. Check the path/URL. Error: {e}")
                if run and mlflow.active_run(): mlflow.end_run()
        
        # Data Preparation
        df = _df.copy()
        problem_type = config['problem_type']
        target = config['target']
        
        if st.session_state.ai_features_approved:
            df = AIFeatureEngineer(st.session_state.feature_recommendations).transform(df)
        
        all_features = config['numerical_features'] + config['categorical_features']
        X = df[all_features]
        y = df[target]
        
        # Drop rows where target is NaN
        y = y.dropna()
        X = X.loc[y.index]
        
        le = None
        if problem_type == 'Classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        stratify_option = y if problem_type == 'Classification' and pd.Series(y).value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42, stratify=stratify_option)
        
        st.session_state.training_data_stats = {
            'X_train_ref': X_train
        }

        # Preprocessing Pipelines
        numerical_steps = [('imputer', SimpleImputer(strategy=config['imputation_strategy']))]
        if config['handle_outliers']:
            numerical_steps.append(('outlier_handler', OutlierHandler()))
        numerical_steps.append(('scaler', StandardScaler() if config['scaler'] == 'StandardScaler' else MinMaxScaler()))
        numerical_transformer = Pipeline(steps=numerical_steps)

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=config['cat_imputation_strategy'], fill_value='missing' if config['cat_imputation_strategy'] == 'constant' else None)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first' if problem_type == 'Classification' else None))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, config['numerical_features']),
            ('cat', categorical_transformer, config['categorical_features'])
        ], remainder='passthrough')
        
        # Model Training and Evaluation Loop
        all_models = get_models_and_search_spaces(problem_type)
        results = []
        models_to_run = {name: all_models[name] for name in config['selected_models'] if name in all_models}
        
        for name, model_info in models_to_run.items():
            with st.spinner(f"Training {name}..."):
                is_ensemble = 'Voting' in name
                if is_ensemble:
                    base_estimators_config = get_models_and_search_spaces(problem_type)
                    estimators = []
                    for model_name in config.get('ensemble_base_models', []):
                        if model_name in base_estimators_config and 'Voting' not in model_name:
                            estimators.append((model_name, clone(base_estimators_config[model_name]['model'])))
                    if not estimators: continue
                    
                    EnsembleModel = VotingClassifier if problem_type == 'Classification' else VotingRegressor
                    ensemble_model = EnsembleModel(estimators=estimators, voting='soft' if problem_type == 'Classification' else 'hard')
                    best_model = Pipeline(steps=[('preprocessor', preprocessor), ('ensemble', ensemble_model)]).fit(X_train, y_train)
                    best_score = 0.0
                else:
                    PipelineClass = ImbPipeline
                    steps = [('preprocessor', preprocessor)]
                    
                    feature_selection_func = f_classif if problem_type == 'Classification' else f_regression
                    if config['use_statistical_feature_selection']: 
                        k = min(config['k_features'], len(all_features))
                        if k > 0:
                            steps.append(('feature_selection', SelectKBest(feature_selection_func, k=k)))

                    if problem_type == 'Classification' and config['imbalance_method'] == 'SMOTE' and IMBLEARN_AVAILABLE and 'Balanced' not in name:
                        steps.append(('smote', SMOTE(random_state=42)))
                    
                    model_key = 'classifier' if problem_type == 'Classification' else 'regressor'
                    steps.append((model_key, clone(model_info['model'])))
                    pipeline = PipelineClass(steps=steps)

                    if SKOPT_AVAILABLE and 'params_bayes' in model_info:
                        cv_splitter = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=42) if problem_type == 'Classification' else KFold(n_splits=config['cv_folds'], shuffle=True, random_state=42)
                        optimizer = BayesSearchCV(pipeline, model_info['params_bayes'], n_iter=config['bayes_iterations'], cv=cv_splitter, scoring=config['primary_metric'], random_state=42, n_jobs=-1)
                        optimizer.fit(X_train, y_train)
                        best_model, best_score = optimizer.best_estimator_, optimizer.best_score_
                    else:
                        pipeline.fit(X_train, y_train)
                        best_model, best_score = pipeline, 0.0

                y_pred = best_model.predict(X_test)
                metrics = {'Model': name, 'Best_Score_CV': best_score}
                
                if problem_type == 'Classification':
                    y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
                    metrics.update({'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0), 'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)})
                    if y_pred_proba is not None:
                        if len(np.unique(y_train)) > 2:
                            metrics.update({'ROC_AUC': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 'F1_Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)})
                        else:
                            metrics.update({'ROC_AUC': roc_auc_score(y_test, y_pred_proba[:, 1]), 'F1_Score': f1_score(y_test, y_pred, zero_division=0)})
                else: # Regression
                    metrics.update({'R2_Score': r2_score(y_test, y_pred), 'MSE': mean_squared_error(y_test, y_pred), 'MAE': mean_absolute_error(y_test, y_pred)})
                
                metrics['model_object'] = best_model
                results.append(metrics)
                if run and mlflow.active_run():
                    with mlflow.start_run(run_id=run.info.run_id, nested=True) as nested_run:
                        mlflow.set_tag("Model", name)
                        mlflow_metrics = {k.replace(' ', '_').replace('(', '').replace(')', ''): v for k, v in metrics.items() if k not in ['Model', 'model_object']}
                        mlflow.log_metrics(mlflow_metrics)

        results_df = pd.DataFrame(results)
        results_df.rename(columns={'Best_Score_CV': 'Best Score (CV)', 'ROC_AUC': 'ROC AUC', 'F1_Score': 'F1-Score', 'R2_Score': 'R2 Score'}, inplace=True)
        primary_metric_col_display = config['primary_metric_display']
        best_model_row = results_df.sort_values(by=primary_metric_col_display, ascending=False).iloc[0]

        if run and mlflow.active_run():
            mlflow.sklearn.log_model(best_model_row['model_object'], "best_model")

        return results_df, le, X_train, y_train, X_test, y_test

    finally:
        if run and mlflow.active_run():
            mlflow.end_run()
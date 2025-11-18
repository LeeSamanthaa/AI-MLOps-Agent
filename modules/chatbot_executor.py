# modules/chatbot_executor.py
import streamlit as st
import re
from typing import Tuple, List
import pandas as pd

class ChatbotExecutor:
    """
    Parses natural language commands to control the MLOps application state.
    It uses regex to match user input and executes corresponding handler methods.
    """
    def __init__(self):
        # Maps regex patterns to their handler methods
        self.command_patterns = {
            # Configuration Management
            r'set (?:problem type|problem) to (classification|regression)': self.set_problem_type,
            r'set target (?:to|variable|as) [\"\']?([\w\s_]+)[\"\']?': self.set_target,
            r'(?:add|include) model[s]? (.+)': self.add_models,
            r'remove model[s]? (.+)': self.remove_models,
            r'set test size to (\d+\.?\d*)%?': self.set_test_size,
            r'(?:enable|use|turn on) smote': lambda: self.set_imbalance_method('SMOTE'),
            r'(?:disable|turn off) smote': lambda: self.set_imbalance_method('None'),
            r'set (?:tuning|bayes|bayesian) iterations to (\d+)': self.set_bayes_iterations,
            r'set cv folds to (\d+)': self.set_cv_folds,
            
            # Data Manipulation
            r'drop (?:the)? column[s]? [\"\']?([\w\s_]+)[\"\']?': self.drop_column,
            r'rename column [\"\']?([\w\s_]+)[\"\']? to [\"\']?([\w\s_]+)[\"\']?': self.rename_column,
            r'fill missing values in [\"\']?([\w\s_]+)[\"\']? with (the )?(median|mean|mode)': self.fill_missing,
            
            # Feature Engineering Commands (NEW)
            r'create (?:a )?polynomial (?:feature|features) (?:for|from) [\"\']?([\w\s_]+)[\"\']?(?: with)?(?: degree)?(\d+)?': self.create_polynomial_feature,
            r'create (?:an? )?interaction (?:feature|features)? (?:between|from) [\"\']?([\w\s_]+)[\"\']? and [\"\']?([\w\s_]+)[\"\']?': self.create_interaction_feature,
            r'create (?:a )?log transform(?:ation)? (?:for|of) [\"\']?([\w\s_]+)[\"\']?': self.create_log_transform,
            r'bin (?:the )?(?:feature|column) [\"\']?([\w\s_]+)[\"\']?(?: into (\d+) bins)?': self.create_binning,
            r'get (?:ai )?feature (?:engineering )?(?:recommendation|suggestion)s?': self.request_ai_features,
            r'apply (?:all )?(?:ai )?features?': self.apply_ai_features,
            
            # Help and Guidance (NEW)
            r'how (?:do i|can i) use (?:this )?(?:app|tool|application)': self.show_app_guide,
            r'(?:show|display) (?:app |application )?(?:guide|help|tutorial)': self.show_app_guide,
            r'what (?:can i|should i) do (?:in )?step (\d+)': self.explain_step,
            
            # Workflow & Status
            r'show(?: me)?(?: the)? current status': self.show_status,
            r'what should i do next\??': self.provide_guidance,
            r'am i ready to train\??': self.check_readiness_to_train
        }
        
    def parse_and_execute(self, user_input: str) -> Tuple[bool, str]:
        """
        Iterates through command patterns, executes the first match, and returns the result.

        Args:
            user_input: The natural language query from the user.

        Returns:
            A tuple containing (bool: if a command was executed, str: response message).
        """
        for pattern, handler in self.command_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                try:
                    # Call the handler with captured groups from the regex
                    response_message = handler(*match.groups())
                    st.rerun() # Refresh UI to reflect state changes
                    return True, response_message
                except Exception as e:
                    return True, f"[ERROR] Error executing command: {e}"
        
        # If no command pattern matched, indicate fallback to RAG
        return False, "Not a command."

    # --- Status and Guidance ---
    
    def get_status_summary(self) -> str:
        """Generates a markdown summary of the current pipeline configuration."""
        config = st.session_state.config
        df = st.session_state.df
        
        status_lines = [
            "**Current Pipeline Status:**",
            f"- **Data Loaded**: {'Yes' if df is not None else 'No'} ({df.shape if df is not None else 'N/A'})",
            f"- **Problem Type**: {config.get('problem_type', 'Not set')}",
            f"- **Target Variable**: {config.get('target', 'Not set')}",
            f"- **Features**: {len(config.get('numerical_features', []))} numerical, {len(config.get('categorical_features', []))} categorical",
            f"- **Models Selected**: {len(config.get('selected_models', []))} ({', '.join(config.get('selected_models', [])[:3])}...)",
            f"- **Training Complete**: {'Yes' if st.session_state.get('phase1_complete') else 'No'}"
        ]
        return "\n".join(status_lines)

    def show_status(self) -> str:
        """Handler to return the current status summary."""
        return self.get_status_summary()
        
    def provide_guidance(self) -> str:
        """Provides context-aware guidance on the next step."""
        if not st.session_state.get('data_loaded'):
            return "Your first step is to upload a dataset or load a sample one from the welcome page."
        if not st.session_state.config.get('target'):
            return "You have data loaded. The next logical step is to set your target variable. You can tell me: `set target to YourColumnName`."
        if not st.session_state.config.get('selected_models'):
            return "You've configured your features. Now it's time to select some models to train. Try: `add models RandomForest and XGBoost`."
        
        return "You seem ready to go! Review your configuration in the main panel and then proceed to the 'Training & Validation' step to run the pipeline."

    def check_readiness_to_train(self) -> str:
        """Checks if the configuration is valid for training."""
        from .utils import validate_config
        errors = validate_config(st.session_state.config, phase='training')
        if not errors:
            return "[SUCCESS] Yes, your configuration appears valid. You are ready to train."
        else:
            error_str = "\n".join([f"- {e}" for e in errors])
            return f"[WARNING] Not quite. Please address these issues first:\n{error_str}"

    # --- Configuration Handlers ---
    
    def set_problem_type(self, problem_type: str) -> str:
        """Sets the problem type (Classification/Regression)."""
        from .ui_components import handle_problem_type_change
        problem_type = problem_type.strip().capitalize()
        if problem_type not in ['Classification', 'Regression']:
            return f"[ERROR] Invalid problem type '{problem_type}'. Must be 'Classification' or 'Regression'."
        
        st.session_state.config['problem_type'] = problem_type
        handle_problem_type_change() # Resets dependent configs and syncs UI
        return f"[SUCCESS] Problem type set to **{problem_type}**."

    def set_target(self, column_name: str) -> str:
        """Sets the target variable."""
        from .ui_components import handle_target_change
        df = st.session_state.df
        column_name = column_name.strip()
        
        if df is None:
            return "[ERROR] Please load data before setting a target."
        if column_name not in df.columns:
            return f"[ERROR] Column '{column_name}' not found in the dataset."
        
        st.session_state.config['target'] = column_name
        handle_target_change() # Auto-infers problem type and syncs UI
        return f"[SUCCESS] Target variable set to **{column_name}**. Problem type auto-detected."

    def add_models(self, models_str: str) -> str:
        """Adds one or more models to the selection list."""
        from .pipeline import get_models_and_search_spaces
        from .ui_components import sync_all_widgets
        
        available_models = list(get_models_and_search_spaces(st.session_state.config['problem_type']).keys())
        requested = [m.strip() for m in re.split(r',| and ', models_str)]
        
        added, not_found = [], []
        for req_model in requested:
            found_match = None
            for avail_model in available_models:
                if req_model.lower() in avail_model.lower():
                    found_match = avail_model
                    break
            
            if found_match and found_match not in st.session_state.config['selected_models']:
                st.session_state.config['selected_models'].append(found_match)
                added.append(found_match)
            elif not found_match:
                not_found.append(req_model)
        
        sync_all_widgets()
        
        messages = []
        if added: messages.append(f"[SUCCESS] Added models: **{', '.join(added)}**.")
        if not_found: messages.append(f"[WARNING] Models not found or invalid for this problem type: {', '.join(not_found)}.")
        return " ".join(messages) if messages else "No new models were added."

    def remove_models(self, models_str: str) -> str:
        """Removes one or more models from the selection list."""
        from .ui_components import sync_all_widgets
        
        current_selection = st.session_state.config['selected_models']
        requested_to_remove = [m.strip() for m in re.split(r',| and ', models_str)]
        removed_models = []

        for rem_model_req in requested_to_remove:
            model_to_remove = None
            for sel_model in current_selection:
                if rem_model_req.lower() in sel_model.lower():
                    model_to_remove = sel_model
                    break
            
            if model_to_remove:
                current_selection.remove(model_to_remove)
                removed_models.append(model_to_remove)
                
        sync_all_widgets()
        return f"[SUCCESS] Removed models: **{', '.join(removed_models)}**." if removed_models else "No models were removed."

    def set_test_size(self, size_str: str) -> str:
        """Sets the test set size."""
        from .ui_components import sync_all_widgets
        try:
            size = float(size_str)
            if size >= 1: size /= 100.0 # Handle inputs like "25" for 25%
            
            if not (0.01 < size < 0.99):
                return "[ERROR] Test size must be between 1% and 99%."
                
            st.session_state.config['test_size'] = size
            sync_all_widgets()
            return f"[SUCCESS] Test set size set to **{size:.0%}**."
        except ValueError:
            return f"[ERROR] Invalid number '{size_str}' for test size."

    def set_imbalance_method(self, method: str) -> str:
        """Enables or disables SMOTE."""
        from .ui_components import sync_all_widgets
        if st.session_state.config['problem_type'] != 'Classification':
            return "[WARNING] Imbalance handling is only applicable for Classification problems."
            
        st.session_state.config['imbalance_method'] = method
        sync_all_widgets()
        return f"[SUCCESS] Imbalanced data handling set to **{method}**."

    def set_bayes_iterations(self, iterations_str: str) -> str:
        """Sets the number of Bayesian tuning iterations."""
        from .ui_components import sync_all_widgets
        try:
            iterations = int(iterations_str)
            if not (1 <= iterations <= 100):
                return "[ERROR] Iterations must be between 1 and 100."
                
            st.session_state.config['bayes_iterations'] = iterations
            sync_all_widgets()
            return f"[SUCCESS] Tuning iterations set to **{iterations}**."
        except ValueError:
            return f"[ERROR] Invalid integer '{iterations_str}' for iterations."

    def set_cv_folds(self, folds_str: str) -> str:
        """Sets the number of cross-validation folds."""
        from .ui_components import sync_all_widgets
        try:
            folds = int(folds_str)
            if not (2 <= folds <= 20):
                return "[ERROR] CV folds must be between 2 and 20."
            
            st.session_state.config['cv_folds'] = folds
            sync_all_widgets()
            return f"[SUCCESS] Cross-validation folds set to **{folds}**."
        except ValueError:
            return f"[ERROR] Invalid integer '{folds_str}' for CV folds."
    
    # --- Data Manipulation Handlers ---

    def drop_column(self, column_name: str) -> str:
        """Drops a column from the dataframe and updates feature lists."""
        from .utils import get_processed_df
        from .ui_components import sync_all_widgets

        df = st.session_state.df
        column_name = column_name.strip()
        
        if df is None: return "[ERROR] No data loaded."
        if column_name not in df.columns: return f"[ERROR] Column '{column_name}' not found."
            
        st.session_state.df.drop(columns=[column_name], inplace=True)
        
        # Also remove from feature selections if present
        for key in ['numerical_features', 'categorical_features']:
            if column_name in st.session_state.config[key]:
                st.session_state.config[key].remove(column_name)
        
        get_processed_df.clear() # Invalidate cache
        sync_all_widgets()
        return f"[SUCCESS] Dropped column **{column_name}** and updated feature lists."

    def rename_column(self, old_name: str, new_name: str) -> str:
        """Renames a column in the dataframe and updates feature lists."""
        from .utils import get_processed_df
        from .ui_components import sync_all_widgets

        df = st.session_state.df
        old_name, new_name = old_name.strip(), new_name.strip()
        
        if df is None: return "[ERROR] No data loaded."
        if old_name not in df.columns: return f"[ERROR] Column '{old_name}' not found."
            
        st.session_state.df.rename(columns={old_name: new_name}, inplace=True)

        for key in ['numerical_features', 'categorical_features', 'target']:
            if st.session_state.config.get(key) == old_name:
                st.session_state.config[key] = new_name
            elif isinstance(st.session_state.config.get(key), list) and old_name in st.session_state.config[key]:
                st.session_state.config[key] = [new_name if item == old_name else item for item in st.session_state.config[key]]

        get_processed_df.clear()
        sync_all_widgets()
        return f"[SUCCESS] Renamed column **{old_name}** to **{new_name}**."

    def fill_missing(self, column_name: str, _, strategy: str) -> str:
        """Fills missing values in a specified column."""
        from .utils import get_processed_df
        df = st.session_state.df
        column_name = column_name.strip()
        strategy = strategy.strip()

        if df is None: return "[ERROR] No data loaded."
        if column_name not in df.columns: return f"[ERROR] Column '{column_name}' not found."
        
        try:
            if strategy == 'median': fill_value = df[column_name].median()
            elif strategy == 'mean': fill_value = df[column_name].mean()
            else: fill_value = df[column_name].mode()[0]
        except Exception as e:
            return f"[ERROR] Could not calculate {strategy} for '{column_name}'. Is it numeric? Error: {e}"
            
        st.session_state.df[column_name].fillna(fill_value, inplace=True)
        get_processed_df.clear()
        return f"[SUCCESS] Filled missing values in **{column_name}** with the {strategy} (Value: {fill_value})."

    # --- Helper for Feature Engineering (NEW) ---
    
    def _validate_feature_command(self, column_names: List[str], feature_type: str) -> Tuple[bool, str]:
        """Validates if columns exist and are of the correct type for feature engineering."""
        df = st.session_state.df
        if df is None:
            return False, "[ERROR] No data loaded. Please load data first."
        
        numerical_features = st.session_state.config.get('numerical_features', [])
        
        for col in column_names:
            col = col.strip()
            if col not in df.columns:
                return False, f"[ERROR] Column '{col}' not found."
            if feature_type == 'numerical' and col not in numerical_features:
                # Check if it's numeric but just not in the list
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.session_state.config['numerical_features'].append(col)
                    from .ui_components import sync_all_widgets
                    sync_all_widgets()
                    st.info(f"Auto-added '{col}' to numerical features list.")
                else:
                    return False, f"[ERROR] Column '{col}' is not a numerical feature. Please add it to 'Numerical Features' in Step 4."
        return True, ""

    # --- Feature Engineering Handlers (FIXED) ---

    def create_polynomial_feature(self, column_name: str, degree_str: str = None) -> str:
        """Creates a polynomial feature recommendation and applies it."""
        from .ui_components import handle_approve_features
        
        is_valid, err_msg = self._validate_feature_command([column_name], 'numerical')
        if not is_valid: return err_msg
        
        degree = int(degree_str) if degree_str else 2
        rec = {
            "technique": "polynomial",
            "column": column_name.strip(),
            "degree": degree,
            "confidence_score": 90, # User-driven
            "rationale_long": f"User-requested polynomial feature (degree {degree}) for {column_name}."
        }
        st.session_state.feature_recommendations.append(rec)
        handle_approve_features()
        return f"[SUCCESS] Created and applied **degree {degree} polynomial feature** for **{column_name}**."

    def create_interaction_feature(self, col1: str, col2: str) -> str:
        """Creates an interaction feature recommendation and applies it."""
        from .ui_components import handle_approve_features
        
        is_valid, err_msg = self._validate_feature_command([col1, col2], 'numerical')
        if not is_valid: return err_msg
        
        rec = {
            "technique": "interaction",
            "column_1": col1.strip(),
            "column_2": col2.strip(),
            "confidence_score": 90, # User-driven
            "rationale_long": f"User-requested interaction feature for {col1} and {col2}."
        }
        st.session_state.feature_recommendations.append(rec)
        handle_approve_features()
        return f"[SUCCESS] Created and applied **interaction feature** for **{col1} x {col2}**."

    def create_log_transform(self, column_name: str) -> str:
        """Creates a log transform feature recommendation and applies it."""
        from .ui_components import handle_approve_features
        
        is_valid, err_msg = self._validate_feature_command([column_name], 'numerical')
        if not is_valid: return err_msg
        
        rec = {
            "technique": "log_transform",
            "column": column_name.strip(),
            "confidence_score": 90, # User-driven
            "rationale_long": f"User-requested log transform for {column_name}."
        }
        st.session_state.feature_recommendations.append(rec)
        handle_approve_features()
        return f"[SUCCESS] Created and applied **log transform** for **{column_name}**."

    def create_binning(self, column_name: str, bins_str: str = None) -> str:
        """Creates a binned feature recommendation and applies it."""
        from .ui_components import handle_approve_features
        
        is_valid, err_msg = self._validate_feature_command([column_name], 'numerical')
        if not is_valid: return err_msg

        bins = int(bins_str) if bins_str else 4
        rec = {
            "technique": "binning",
            "column": column_name.strip(),
            "bins": bins,
            "confidence_score": 90, # User-driven
            "rationale_long": f"User-requested binning ({bins} bins) for {column_name}."
        }
        st.session_state.feature_recommendations.append(rec)
        handle_approve_features()
        return f"[SUCCESS] Created and applied **binning ({bins} bins)** for **{column_name}**."

    def request_ai_features(self) -> str:
        """Triggers the AI to generate feature recommendations."""
        from .ui_components import handle_feature_recommendations
        
        if st.session_state.df is None: return "[ERROR] No data loaded."
        if not st.session_state.config.get('target'): return "[ERROR] Please set a target variable first."
            
        handle_feature_recommendations(st.session_state.df)
        return "[SUCCESS] AI feature recommendations have been generated. Review them in **Step 3: Feature Engineering**."

    def apply_ai_features(self) -> str:
        """Applies all pending AI-generated features."""
        from .ui_components import handle_approve_features
        
        if not st.session_state.feature_recommendations:
            return "[WARNING] No pending AI feature recommendations to apply."
            
        handle_approve_features()
        return "[SUCCESS] All pending AI features have been applied to the dataset."

    # --- Help & Guidance Handlers (FIXED) ---
    
    def show_app_guide(self) -> str:
        """Shows a simple guide on how to use the app."""
        st.session_state.view = 'welcome'
        return "[SUCCESS] Navigating to the Welcome Page for the full application guide."


    def explain_step(self, step_str: str) -> str:
        """Explains the purpose of a specific workflow step."""
        try:
            step = int(step_str)
            if step == 1: 
                st.session_state.workflow_step = 1
                return "Navigating to **Step 1**. Here, you load your data and can use 'Auto-Configure' to let AI detect your target variable and feature types."
            if step == 2: 
                st.session_state.workflow_step = 2
                return "Navigating to **Step 2**. Here, you explore your data. Check the plots and VIF analysis to find issues like multicollinearity or skewed data."
            if step == 3: 
                st.session_state.workflow_step = 3
                return "Navigating to **Step 3**. Here, you create new features. Click 'Get AI Feature Recommendations' for smart suggestions, or tell me to create them (e.g., `create polynomial feature for Age`)."
            if step == 4: 
                st.session_state.workflow_step = 4
                return "Navigating to **Step 4**. Here, you configure the pipeline. Choose your models, set the test size, and decide on tuning iterations."
            if step == 5: 
                st.session_state.workflow_step = 5
                return "Navigating to **Step 5**. Here, you run the training! Click 'Benchmark Individual Models' to start."
            if step == 6: 
                st.session_state.view = 'results'
                return "Navigating to **Step 6**. Here, you review everything. See which model won, get an AI summary, and download your deployment package."
            return "[ERROR] Please enter a step number from 1 to 6."
        except ValueError:
            return f"[ERROR] Invalid step '{step_str}'. Please enter a number from 1 to 6."
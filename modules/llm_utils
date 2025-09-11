import streamlit as st
import json

# Attempt to import Groq, handle if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Attempt to import imblearn for SMOTE check
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

def generate_llm_response(prompt, api_key, is_json=False):
    """
    Generates a response from the Groq LLM API.
    """
    if not api_key:
        return "Please provide a Groq API key in the sidebar to use AI features."
    if not GROQ_AVAILABLE:
        return "Groq library not found. Please install it (`pip install groq`)."
    try:
        client = Groq(api_key=api_key)
        response_format = {"type": "json_object"} if is_json else None
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format=response_format
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI response. Error: {e}"

def get_feature_engineering_prompt(df_info, target, numerical_features, categorical_features):
    """
    Creates a prompt for the LLM to suggest feature engineering techniques.
    """
    return f"""
You are an expert data scientist. Analyze this dataset and recommend 2-4 specific feature engineering techniques to improve model performance for predicting {target}.
Dataset Info: Target={target}, Shape={df_info['shape']}, Numerics={numerical_features}, Categoricals={categorical_features}, Stats={df_info['description']}
Return your response as a JSON object: {{"recommendations": [{{"technique": "binning|polynomial|log_transform|interaction", "column": "col_name", "rationale": "Why this helps.", "bins": 4, "degree": 2, "other_column": "other_col"}}]}}.
Crucially, ensure every recommendation object in the list has a non-empty value for 'technique', 'column', and 'rationale'."""

def get_ai_config_prompt(df):
    """
    Creates a prompt for the LLM to auto-configure the pipeline based on the dataset.
    """
    prompt = f"""
As an expert data scientist, analyze the following dataset to determine the optimal initial configuration for a machine learning pipeline.

**Dataset Information:**
- **Columns:** {list(df.columns)}
- **Data Head:**
{df.head().to_string()}
- **Data Types:**
{df.dtypes.to_string()}
- **Unique Values per Column:**
{df.nunique().to_string()}

**Your Task:**
Based on the data, determine the following:
1.  **`problem_type`**: Is this a 'Classification' or 'Regression' problem? A continuous, non-integer target suggests regression. A target with a small number of distinct values suggests classification.
2.  **`target_column`**: Which column is the most likely target for prediction? This is usually the column you are trying to predict (e.g., 'Churn', 'Price').
3.  **`numerical_features`**: Which columns are numerical features? Exclude identifiers (like CustomerID) and the target column.
4.  **`categorical_features`**: Which columns are categorical features? Exclude identifiers and the target column.
5.  **`rationale`**: Briefly explain your reasoning for these choices.

**Return your response as a single, valid JSON object with the following exact keys:**
`{{"problem_type": "...", "target_column": "...", "numerical_features": [], "categorical_features": [], "rationale": "..."}}`
"""
    return prompt

def get_rag_expert_context():
    """
    Returns a static knowledge base for the RAG system about ML concepts.
    """
    return """
**Expert Knowledge Base on Interpreting ML Models for Business**

* **On Classification Metrics (e.g., Customer Churn):**
    * **Precision**: Of all the customers we *predicted* would churn, what percentage actually did? **Business Impact**: High precision is critical when the cost of intervention is high. You want to avoid spending resources on customers who were never going to leave.
    * **Recall (Sensitivity)**: Of all the customers who *actually* churned, what percentage did our model successfully identify? **Business Impact**: High recall is critical when the cost of losing a customer is high. You want to cast a wide net and catch as many at-risk customers as possible, even if it means incorrectly flagging a few loyal ones.
    * **F1-Score**: The harmonic mean of Precision and Recall. **Business Impact**: A great single metric to optimize when the business cost of a false positive (wasting resources) and a false negative (losing a customer) are roughly equal.
    * **ROC AUC**: Measures the model's ability to distinguish between the positive and negative classes. An AUC of 0.5 is random guessing; 1.0 is a perfect model. **Business Impact**: A high AUC indicates a reliable model that consistently ranks at-risk customers higher than loyal ones.

* **On Regression Metrics (e.g., House Price Prediction):**
    * **R² Score (Coefficient of Determination)**: What percentage of the variance in the target variable (e.g., Price) can be explained by our model's features? **Business Impact**: An R² of 0.75 means our features can explain 75% of the price movements. It's a measure of the model's overall explanatory power. It does not indicate prediction error.
    * **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values. **Business Impact**: Punishes large errors more heavily than small ones. If a few very wrong predictions are particularly bad for the business (e.g., underpricing a luxury home by a huge margin), MSE is a key metric to minimize.
    * **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values. **Business Impact**: Directly interpretable in the target's units. An MAE of $5,000 means our price predictions are, on average, off by $5,000. This is an excellent, easy-to-explain measure of typical prediction error.

* **On Model Explainability (SHAP):**
    * **SHAP Values**: Show how much each feature contributed to pushing a specific prediction away from the average. **Business Impact**: SHAP moves beyond *what* the model predicts to *why*. If 'Age' is the top feature for a churn model, it provides a data-driven reason to focus business strategy on age-related customer segments.

* **On Data Drift:**
    * **Data Drift**: Occurs when the statistical properties of the data in production change from the data the model was trained on. **Business Impact**: This is a silent killer of model performance. If the average age of new customers suddenly drops, a model trained on older customers will become unreliable. Monitoring for drift is essential to know when a model needs to be retrained.
"""

def generate_dynamic_data_context(df, config, eda_summary):
    """
    Creates a dynamic textual summary of the user's dataset and EDA findings for the RAG prompt.
    """
    target = config.get('target')
    if not target or df is None:
        return "No data loaded or target selected."

    context = f"**Dataset Shape:** {df.shape}\n\n"
    numerical_features = config.get('numerical_features', [])
    if numerical_features:
        context += "**Descriptive Statistics for Numerical Features:**\n" + df[numerical_features].describe().to_string() + "\n\n"
    
    categorical_features = config.get('categorical_features', [])
    if categorical_features:
        context += "**Categorical Feature Value Counts:**\n"
        for col in categorical_features:
            context += f"- **{col}**:\n{df[col].value_counts().to_string()}\n"
    
    if target in df.columns and numerical_features:
        numeric_df = df[numerical_features + [target]]
        if len(numeric_df.columns) > 1:
            try:
                corr_matrix = numeric_df.corr(numeric_only=True)
                if not corr_matrix.empty and target in corr_matrix:
                    target_correlations = corr_matrix[target].sort_values(ascending=False).drop(target)
                    context += "\n**Top 5 Feature Correlations with Target:**\n"
                    context += target_correlations.head(5).to_string()
            except Exception:
                pass # Ignore correlation calculation errors
    
    if eda_summary:
        context += "\n\n--- START EDA SUMMARY ---\n" + eda_summary + "\n--- END EDA SUMMARY ---"

    return context

def apply_ai_recommendations():
    """
    Applies configuration changes suggested by the AI Action Panel to the session state.
    """
    config_updates = {}
    messages = []
    for action in st.session_state.parsed_actions:
        action_type = action.get('type')
        if action_type == 'increase_tuning':
            current_iterations = st.session_state.config.get('bayes_iterations', 15)
            new_iterations = min(50, current_iterations + 10) # Cap at 50
            config_updates['bayes_iterations'] = new_iterations
            messages.append(f"Increased **Hyperparameter Tuning Iterations** to {new_iterations}.")
        elif action_type == 'try_smote' and IMBLEARN_AVAILABLE:
            config_updates['imbalance_method'] = 'SMOTE'
            messages.append("Enabled **SMOTE** for imbalanced data handling.")
        elif action_type == 'try_feature_selection':
            config_updates['use_statistical_feature_selection'] = True
            messages.append("Enabled **Statistical Feature Selection**.")
            
    st.session_state.config.update(config_updates)
    st.session_state.guide_message = "The AI Agent has automatically updated the following settings for you. Review them and re-run the pipeline.\n\n" + "\n".join([f"- {msg}" for msg in messages])
    st.session_state.view = 'configuration'
    st.rerun()


# ============================================================================
# FILE 1: modules/llm_utils.py
# CHANGES: Implemented robust, user-facing retry logic with exponential
# backoff, model fallback for 503 errors, and API status checks.
# Strengthened error handling and added security/validation checks.
# ============================================================================
# modules/llm_utils.py
# ENHANCED VERSION - Added confidence scoring, more robust JSON parsing, and improved prompt engineering.
# Original functionality preserved, added new keys to expected JSON outputs and refined prompts for reliability.

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import re
import time
import os
import requests
from functools import wraps

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Library Availability Check ---
try:
    from groq import Groq, APIError, AuthenticationError, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def sanitize_for_prompt(text, max_length=100):
    """Sanitize text to prevent prompt injection."""
    if not isinstance(text, str):
        text = str(text)
    # Remove common injection patterns
    text = text.replace("Ignore all previous", "")
    text = text.replace("ignore previous", "")
    text = text.replace("disregard", "")
    return text[:max_length]

@st.cache_data(ttl=300) # Cache for 5 minutes
def check_groq_status():
    """Pings the Groq API to check for service availability."""
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", timeout=5)
        if response.status_code == 200:
            return True, "Groq API is operational."
        elif response.status_code == 503:
            return False, "Groq API is currently experiencing high load (503 Service Unavailable)."
        else:
            return False, f"Groq API returned an unexpected status: {response.status_code}."
    except requests.RequestException as e:
        return False, f"Could not connect to Groq API: {e}"

def generate_llm_response(prompt, api_key, is_json=False, check_api_status=True):
    """
    Generates a response from the Groq LLM API with enhanced error handling,
    retry logic with exponential backoff, and model fallback.
    """
    # Initialize token counter and error state in session state
    if 'llm_token_usage' not in st.session_state:
        st.session_state.llm_token_usage = {'total_tokens': 0, 'estimated_cost': 0.0}
    st.session_state.last_api_error = None

    if not api_key:
        st.error("Error: Please provide a Groq API key in the sidebar to use AI features.")
        return None
    if not api_key.startswith('gsk_'):
        st.error("Error: Invalid Groq API Key format. It should start with 'gsk_'.")
        return None

    if not GROQ_AVAILABLE:
        st.error("Error: The 'groq' library is not installed. Please run `pip install groq`.")
        return None
        
    if check_api_status:
        is_available, message = check_groq_status()
        if not is_available:
            st.warning(f"Heads up: {message} AI features may be slow or unavailable.")

    # Get config for retry and models
    config = st.session_state.config
    max_retries = config.get('api_retry_attempts', 3)
    backoff_factor = config.get('api_retry_backoff', 5)
    primary_model = config.get('groq_model', 'llama-3.1-8b-instant')
    fallback_model = config.get('groq_fallback_model', 'llama-3.1-70b-versatile')
    current_model = primary_model

    for attempt in range(max_retries):
        try:
            client = Groq(api_key=api_key)
            response_format = {"type": "json_object"} if is_json else None
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=current_model,
                response_format=response_format
            )
            response_content = chat_completion.choices[0].message.content

            # Track token usage
            if hasattr(chat_completion, 'usage'):
                tokens_used = chat_completion.usage.total_tokens
                st.session_state.llm_token_usage['total_tokens'] += tokens_used
                cost_per_1k = 0.0001
                st.session_state.llm_token_usage['estimated_cost'] += (tokens_used / 1000) * cost_per_1k

            # Sanitize and clean JSON output before parsing
            if is_json and response_content:
                clean_response = re.sub(r"^```json|```$", "", response_content.strip(), flags=re.MULTILINE)
                clean_response = clean_response.strip()
                
                # FIX: Use a more robust regex to remove trailing commas
                # This finds any comma that is followed by optional whitespace and a closing } or ]
                clean_response = re.sub(r',(?=\s*[}\]])', '', clean_response)
                
                try:
                    # Validate JSON structure
                    parsed = json.loads(clean_response)
                    logging.info(f"Successfully parsed JSON with keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
                    return clean_response
                except json.JSONDecodeError as e:
                    # Enhanced error logging
                    error_position = getattr(e, 'pos', 0)
                    context_start = max(0, error_position - 50)
                    context_end = min(len(clean_response), error_position + 50)
                    error_context = clean_response[context_start:context_end]
                    
                    logging.error(f"JSON Parse Error: {e}")
                    logging.error(f"Error at position {error_position}")
                    logging.error(f"Context: ...{error_context}...")
                    logging.error(f"Full response (first 500 chars): {clean_response[:500]}")
                    
                    st.error(f"AI returned invalid JSON. Error: {str(e)}")
                    with st.expander("Show AI Response for Debugging"):
                        st.code(clean_response, language="json")
                    return None
            return response_content

        except AuthenticationError:
            st.error("Groq API Error: Authentication failed. Please check if your API key is correct and active.")
            logging.error("Groq API authentication failed.")
            st.session_state.last_api_error = "Authentication failed. Check your API key."
            return None
        except RateLimitError as e:
            st.error("Groq API Error: Rate limit exceeded. Please wait and try again later.")
            logging.error(f"Groq API rate limit error: {e}")
            st.session_state.last_api_error = "Rate limit exceeded."
            return None
        except APIError as e:
            wait_time = backoff_factor ** attempt
            st.session_state.last_api_error = f"API Error: {e.status_code} - {e.message}"
            logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries}) with status {e.status_code}. Retrying in {wait_time}s...")
            
            # Specific handling for 503 Service Unavailable (overloaded)
            if e.status_code == 503:
                st.warning(f"Groq API is overloaded. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                if current_model == primary_model and fallback_model:
                    current_model = fallback_model
                    st.info(f"Switching to fallback model ({fallback_model}) for the next attempt.")
            # FIX: Handle 400 error from log
            elif e.status_code == 400:
                st.error(f"Groq API Error (400): The AI failed to generate a valid response (e.g., bad JSON). Check logs for 'failed_generation' details.")
                logging.error(f"Groq API 400 Bad Request: {e.message}")
                return None
            else:
                st.warning(f"API Error ({e.status_code}). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")

            if attempt == max_retries - 1:
                st.error(f"Groq API Error: Could not generate AI response after {max_retries} attempts. Error: {e}")
                logging.error(f"Groq API error after max retries: {e}")
                return None
            
            time.sleep(wait_time)
        except Exception as e:
            st.error(f"An unexpected error occurred while contacting the AI. Error: {e}")
            logging.error(f"Unexpected error in generate_llm_response: {e}", exc_info=True)
            st.session_state.last_api_error = f"An unexpected error occurred: {e}"
            return None
    return None


def parse_ai_recommendations(summary_text, api_key):
    """
    Enhanced parser that extracts actionable recommendations from AI summaries.
    Handles diverse recommendation formats and maps them to executable actions.
    
    Args:
        summary_text: The AI-generated summary containing recommendations
        api_key: Groq API key for the parsing LLM call
        
    Returns:
        list: List of action dictionaries with type and parameters
    """
    parser_prompt = f"""Analyze the following ML results summary and extract ALL actionable recommendations.
For each recommendation, determine the action type and parameters.

Action Types:
1. "increase_tuning" - Increase hyperparameter tuning iterations
2. "try_smote" - Enable SMOTE for imbalanced data
3. "try_feature_selection" - Enable statistical feature selection
4. "add_model" - Add a specific model (include "model_name" key)
5. "enable_regularization" - Enable regularization (include "model_name" and "param_name")
6. "create_polynomial_features" - Create polynomial features (include "degree")
7. "try_ensemble_strategy" - Try different ensemble strategies (include "strategy": "voting", "stacking", or "bagging")
8. "investigate_interactions" - Recommend feature interaction analysis
9. "adjust_test_size" - Change test/train split (include "new_size")
10. "increase_cv_folds" - Increase cross-validation folds (include "new_folds")

Summary Text:
{summary_text}

Return a JSON object with this exact structure:
{{
  "actions": [
    {{
      "action_text": "Brief description of the recommendation",
      "type": "one of the action types above",
      "model_name": "ModelName (if applicable)",
      "param_name": "parameter_name (if applicable)",
      "param_value": "value (if applicable)",
      "degree": 2 (if applicable),
      "strategy": "strategy_name (if applicable)",
      "new_size": 0.25 (if applicable),
      "new_folds": 10 (if applicable)
    }}
  ]
}}

Example:
If the text says "Try Random Forest", return:
{{
  "actions": [
    {{
      "action_text": "Add Random Forest model to improve performance",
      "type": "add_model",
      "model_name": "RandomForest"
    }}
  ]
}}

If the text says "Increase tuning iterations to 50", return:
{{
  "actions": [
    {{
      "action_text": "Increase Bayesian tuning iterations to 50",
      "type": "increase_tuning",
      "param_value": 50
    }}
  ]
}}

IMPORTANT: Extract ALL recommendations, even if they're phrased differently."""

    response = generate_llm_response(parser_prompt, api_key, is_json=True)
    
    if not response:
        logging.warning("AI recommendation parser returned empty response")
        return []
    
    try:
        parsed = json.loads(response)
        actions = parsed.get("actions", [])
        logging.info(f"Successfully parsed {len(actions)} recommendations")
        return actions
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to parse AI recommendations: {e}")
        logging.error(f"Raw response: {response[:500]}")
        return []

def execute_data_transformation(df, user_query, api_key):
    """
    Uses an LLM to generate and execute Python code to transform a DataFrame.
    FIXED: Now returns tuple (df_modified, code_executed) in ALL cases.
    
    Args:
        df: Input DataFrame
        user_query: Natural language description of transformation
        api_key: Groq API key
        
    Returns:
        tuple: (df_modified or None, code_executed or None)
    """
    if df is None or df.empty:
        st.error("Cannot execute transformation on an empty or non-existent dataset.")
        return None, None
    if not user_query or not api_key:
        st.error("Cannot execute transformation. Query or API key is missing.")
        return None, None
    
    prompt = f"""Given a pandas DataFrame named `df`, write a Python script to perform the following transformation: '{user_query}'.
The final transformed DataFrame must be assigned to a variable named `df_modified`.
Do not include any code outside of the transformation logic. For example, no `import pandas as pd` or `df = pd.read_csv(...)`.
The DataFrame `df` is already in scope.
Your code must only output the python code block with NO markdown formatting."""
    
    code_to_execute = generate_llm_response(prompt, api_key, is_json=False)
    
    if not code_to_execute:
        st.error("AI could not generate the required code. Please try rephrasing your request.")
        logging.error("LLM returned empty response for data transformation")
        return None, "ERROR: No code generated"
        
    # Remove markdown fences if present
    code_to_execute = code_to_execute.strip()
    if code_to_execute.startswith("```"):
        lines = code_to_execute.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code_to_execute = '\n'.join(lines)
        
    forbidden_keywords = ['import', 'os', 'sys', 'subprocess', 'eval', 'execfile', 'open', '__']
    if any(kw in code_to_execute for kw in forbidden_keywords):
        st.error("Generated code contains forbidden operations for security reasons.")
        logging.error(f"Generated code failed security check: {code_to_execute[:100]}")
        return None, "ERROR: Security violation"
        
    safe_globals = {'df': df.copy(), 'pd': pd, 'np': np}
    
    try:
        with st.spinner("Executing transformation..."):
            exec(code_to_execute, safe_globals)
            
            if 'df_modified' in safe_globals:
                return safe_globals['df_modified'], code_to_execute
            else:
                st.error("Execution failed: The generated code did not produce a `df_modified` variable.")
                return None, None
                
    except Exception as e:
        st.error(f"A runtime error occurred during code execution: {e}")
        logging.error(f"Error executing LLM-generated code:\n{code_to_execute}\n\nError: {e}", exc_info=True)
        return None, None

def get_feature_engineering_prompt(df_info, target, numerical_features, categorical_features):
    """Creates a structured prompt for the LLM to recommend feature engineering techniques."""
    # ENHANCED PROMPT: Now asks for confidence score and uses a more robust JSON structure.
    return f"""
You are an expert data scientist. Analyze the provided dataset information and suggest 2-4 specific feature engineering techniques. For each recommendation, provide a detailed rationale and a confidence score.

Dataset Info:
- Target Variable: "{target}"
- Shape: {df_info['shape']}
- Numerical Features: {[sanitize_for_prompt(f) for f in numerical_features]}
- Categorical Features: {[sanitize_for_prompt(f) for f in categorical_features]}
- Descriptive Statistics:\n{df_info['description']}

Your task is to return a valid JSON object with a single key "recommendations". The value should be a list of objects.
Each object must have the keys "technique", "column", "rationale_long", and "confidence_score".
- `confidence_score` should be an integer between 0 and 100 representing your certainty in the recommendation's utility.
For the 'interaction' technique, use 'column_1' and 'column_2' instead of 'column'.
The "rationale_long" must be a detailed, multi-sentence paragraph.

Example Format:
{{
  "recommendations": [
    {{
      "technique": "binning",
      "column": "Age",
      "rationale_long": "The 'Age' feature's descriptive statistics show a wide range. Binning this variable into groups could help the model capture non-linear relationships with the target variable, especially if the target's behavior changes abruptly at certain age thresholds.",
      "confidence_score": 85
    }}
  ]
}}
Ensure the 'technique' is one of: "binning", "polynomial", "log_transform", "interaction", "target_encode".
"""

def get_ai_config_prompt(df):
    """Creates a structured prompt for the LLM to auto-configure the pipeline."""
    # ENHANCED PROMPT: Now asks for confidence scores and gives explicit rules for discrete numericals.
    prompt = f"""
As an expert data scientist, analyze the following dataset to determine the optimal initial configuration for a machine learning pipeline. Provide a detailed, multi-sentence rationale for your choices.

Dataset Information:
- Columns: {[sanitize_for_prompt(str(col)) for col in df.columns]}
- Data Head:\n{df.head().to_string()}
- Data Types:\n{df.dtypes.to_string()}
- Unique Values per Column:\n{df.nunique().to_string()}

Your Task:
Based on the data, determine the following:
1.  `problem_type`: Is this a 'Classification' or 'Regression' problem?
2.  `target_column`: Which column is the most likely target variable?
3.  `numerical_features`: Which columns are numerical features? Exclude identifiers and the target.
4.  `categorical_features`: Which columns are categorical features? Exclude identifiers and the target.
5.  `confidence_score`: An integer (0-100) representing your overall confidence in this configuration.
6.  `rationale`: A detailed, multi-sentence paragraph explaining your reasoning.

**CRITICAL RULES for classification:**
-   **Numerical vs. Categorical:** Pay close attention to `df.nunique()` and `df.dtypes`.
-   **Identifiers:** Columns with high unique values (e.g., `CustomerID`, `OrderID`) are **not features**. Exclude them.
-   **Discrete Numerical:** Columns that are `int64` and have a *small* number of unique values (e.g., 2-25) are tricky. **USE THE COLUMN NAME as a strong hint.**
    -   `'Bedrooms'`, `'Bathrooms'`, `'NumOfProducts'`, `'Tenure'`, `'Quantity'`, `'YearBuilt'` are almost always **NUMERICAL features**.
    -   `'ProductCode'`, `'ZipCode'`, `'GroupID'`, `'StatusFlag'` are almost always **CATEGORICAL features**, even if they are numbers.
-   **Target:** The target variable is often the goal (e.g., 'Price', 'Churn', 'Survived').

Return your response as a single, valid JSON object with these exact keys:
{{"problem_type": "...", "target_column": "...", "numerical_features": [], "categorical_features": [], "confidence_score": ..., "rationale": "..."}}
"""
    return prompt

def get_eda_summary_prompt(df, config):
    """Creates a prompt for the LLM to summarize the EDA."""
    return f"""
You are an expert data scientist providing a high-level summary of a dataset for a colleague.
Based on the data profile below, provide a concise summary in Markdown format.

**Data Profile:**
- **Shape:** {df.shape}
- **Target Variable:** {config.get('target', 'Not Set')}
- **Numerical Features:** {config.get('numerical_features', [])}
- **Categorical Features:** {config.get('categorical_features', [])}
- **Missing Values:** {df.isnull().sum().sum()} total missing values.
- **Data Head:**
{df.head().to_string()}

**Your Task:**
Write a brief summary covering:
1.  **Overall Data Quality:** Comment on the size of the data and the presence of missing values.
2.  **Key Feature Observations:** Point out any interesting distributions or potential identifiers you notice from the head of the data.
3.  **Potential Challenges:** Mention 1-2 potential challenges for modeling (e.g., "The presence of identifiers like CustomerID will need to be handled," or "The wide range in 'Balance' suggests scaling will be important.").
"""

def get_vif_analysis_prompt(high_vif_df):
    """Creates a prompt for the LLM to analyze VIF scores and recommend actions."""
    # ENHANCED PROMPT: Asks for a simple, structured response to make parsing more reliable.
    return f"""
You are an expert data scientist explaining multicollinearity.
The following features have a high Variance Inflation Factor (VIF > 10).

**High VIF Features:**
{high_vif_df.to_string(index=False)}

**Your Task:**
1.  Provide a simple, one-sentence explanation of what high VIF means.
2.  Recommend which features to **remove** to fix this issue.
3.  Provide a brief rationale for each removal.

Return your response as a single, valid JSON object with the following structure:
{{
    "explanation": "Your one-sentence explanation here.",
    "features_to_remove": [
        {{
            "feature": "FeatureName1",
            "rationale": "Reason for removing FeatureName1."
        }},
        {{
            "feature": "FeatureName2",
            "rationale": "Reason for removing FeatureName2."
        }}
    ]
}}
"""

def get_ensemble_guidance_prompt(results_df, primary_metric):
    """Creates a prompt for the LLM to recommend base models for an ensemble."""
    top_performers = results_df.sort_values(by=primary_metric, ascending=False)
    top_performers = top_performers[~top_performers['Model'].str.contains("Voting")]
    
    return f"""
You are an AI assistant guiding a user on how to build a powerful ensemble model.
Based on the model performance results below, recommend the best 2-3 models to include in a 'Voting' ensemble.

**Model Performance (sorted by {primary_metric}):**
{top_performers[['Model', primary_metric]].to_string(index=False)}

**Your Task:**
1.  Identify the top 2-3 individual models from the table.
2.  Write a brief recommendation explaining that ensembles often work best by combining a diverse set of strong models.
3.  List the models you recommend selecting in the 'Base Models for Ensemble' multi-select box.
"""

def get_rag_expert_context():
    """Provides a static knowledge base for the RAG system."""
    return """
**Expert Knowledge on Interpreting ML Models for Business**

* **Classification Metrics (e.g., Customer Churn):**
    * **Precision**: Of all customers we *predicted* would churn, what % actually did? **Business Impact**: High precision is critical when intervention is expensive. Avoid spending resources on customers who were never going to leave.
    * **Recall**: Of all customers who *actually* churned, what % did we identify? **Business Impact**: High recall is critical when losing a customer is expensive. Cast a wide net to catch as many at-risk customers as possible.
    * **ROC AUC**: Measures the model's ability to distinguish between classes. **Business Impact**: A high AUC indicates a reliable model that consistently ranks at-risk customers higher than loyal ones.

* **Regression Metrics (e.g., House Price Prediction):**
    * **R² Score**: What percentage of the variance in price can be explained by our model's features? **Business Impact**: An R² of 0.75 means our features explain 75% of price movements. It's a measure of explanatory power, not prediction error.
    * **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values. **Business Impact**: Directly interpretable. An MAE of $5,000 means our price predictions are, on average, off by $5,000. It's an excellent measure of typical error.

* **On Data Drift:**
    * **Data Drift**: Occurs when production data's statistical properties change from the training data. **Business Impact**: A silent killer of model performance. If the average age of new customers suddenly drops, a model trained on older customers will become unreliable. Monitoring for drift is essential to know when to retrain.
"""

def generate_dynamic_data_context(df, config, eda_summary):
    """Creates a textual summary of the dataset and EDA for the RAG prompt."""
    if df is None or not config:
        return "No data loaded or configuration set."

    target = config.get('target')
    context = f"**Dataset Shape:** {df.shape}\n"
    numerical_features = config.get('numerical_features', [])
    if numerical_features:
        valid_num_features = [f for f in numerical_features if f in df.columns]
        if valid_num_features:
            context += "**Numerical Features Stats:**\n" + df[valid_num_features].describe().to_string() + "\n\n"
    
    if target and target in df.columns and numerical_features:
        try:
            numeric_cols_for_corr = [col for col in numerical_features if col in df.columns]
            if len(numeric_cols_for_corr) > 1:
                numeric_df = df[numeric_cols_for_corr + [target]]
                corr_matrix = numeric_df.corr(numeric_only=True)
                if not corr_matrix.empty and target in corr_matrix:
                    target_correlations = corr_matrix[target].sort_values(ascending=False).drop(target)
                    context += "\n**Top 5 Feature Correlations with Target:**\n"
                    context += target_correlations.head(5).to_string()
        except Exception as e:
            logging.warning(f"Could not generate correlation matrix for context. Error: {e}")
            pass
            
    if eda_summary:
        context += "\n\n--- EDA SUMMARY ---\n" + eda_summary + "\n--- END EDA SUMMARY ---"
    return context

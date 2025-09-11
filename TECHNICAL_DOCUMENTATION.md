# Technical Documentation: AI MLOps Agent

## 1. Modular Architecture

The application has been refactored from a single-file script into a modular, package-based structure to promote separation of concerns, maintainability, and code reuse. The core logic is encapsulated within the `modules/` directory.

-   **`app.py`**: This is the main entry point and acts as a simple controller. Its sole responsibilities are to initialize the session state and route the user to the correct "view" (Welcome, Configuration, or Results) based on `st.session_state.view`. It imports and calls high-level functions from `modules.ui_components`.

-   **`modules/`**: This directory is a Python package containing the application's core logic.
    -   **`ui_components.py`**: Contains all functions that render Streamlit widgets and compose the user interface. This includes the main page layouts (`display_welcome_page`, `display_configuration_page`, `display_results_page`), complex UI elements like the EDA and results tabs, and the sidebar chat. It is the bridge between the user and the backend logic.
    -   **`pipeline.py`**: Encapsulates the entire machine learning workflow. It contains the custom `scikit-learn` transformers (`OutlierHandler`, `AIFeatureEngineer`), the `get_models_and_search_spaces` function which defines the model arsenal, and the main `run_pipeline` function. This function is decorated with `@st.cache_resource` to prevent re-running on simple UI interactions and to persist the trained models.
    -   **`llm_utils.py`**: Centralizes all communication with the Large Language Model (Groq). It handles API calls (`generate_llm_response`), contains all prompt engineering templates (`get_ai_config_prompt`, etc.), and builds the context for the Retrieval-Augmented Generation (RAG) system (`get_rag_expert_context`, `generate_dynamic_data_context`).
    -   **`utils.py`**: A collection of general-purpose, stateless helper functions. This includes data loading (`load_data`), sample data generation (`get_sample_datasets`), and deployment script creation (`generate_fastapi_script`).

## 2. State Management

A flawless user experience is maintained through careful management of Streamlit's session state (`st.session_state`).

-   **Initialization**: `initialize_session_state` in `ui_components.py` is called once at the start of `app.py` to create all necessary keys in the session state dictionary. This prevents `KeyError` exceptions.

-   **Preserving User Input**: All interactive widgets on the configuration page (sliders, checkboxes, multiselects) are designed to be "state-aware." Their `value` or `default` arguments are read directly from `st.session_state.config`. At the end of the configuration page render, the current values of all these widgets are written back to `st.session_state.config`. This ensures that if the page re-renders (e.g., after the AI suggests features), no user selections are lost.

-   **Bug Fix: `StreamlitAPIException`**: The previous exception when changing the "Problem Type" has been resolved. The `on_change` callback for the problem type radio button (`on_problem_type_change`) now performs two critical actions:
    1.  It updates `st.session_state.config['problem_type']`.
    2.  It immediately resets `st.session_state.config['selected_models']` to the default list of models for the *newly selected* problem type.
    This ensures the `st.multiselect` for model selection never receives a `default` list containing models that are not in its `options` list, which was the root cause of the bug.

-   **Elimination of Disruptive Reruns**: `st.rerun()` is used judiciously and only for page navigation (e.g., moving from 'configuration' to 'results'). It is not used within component callbacks on the configuration page, preventing unexpected state resets and ensuring a smooth user workflow.

## 3. ML Pipeline and MLOps

-   **Dynamic Pipeline Construction**: The `run_pipeline` function in `pipeline.py` dynamically constructs a `scikit-learn` Pipeline. Steps like `OutlierHandler`, `SelectKBest`, and `SMOTE` are conditionally added based on the user's configuration in `st.session_state.config`.

-   **Robust MLflow Integration**:
    -   The connection to the MLflow server is wrapped in a `try...except` block to gracefully handle connection errors without crashing the application.
    -   The main run and all nested model runs are managed with `try...finally` and `with` statements, respectively, ensuring that `mlflow.end_run()` is always called to prevent orphaned processes.
    -   **Bug Fix: `MlflowException`**: Metric names are sanitized before logging. The code `k.replace(' ', '_').replace('(', '').replace(')', '')` removes spaces and parentheses (e.g., `Best Score (CV)` becomes `Best_Score_CV`), which prevents the "invalid metric name" error.

## 4. AI and RAG System

-   **Hybrid RAG**: The AI Co-Pilot chat in `ui_components.py` implements a hybrid RAG system. Before calling the LLM, it constructs a detailed prompt containing three distinct context types, which are generated in `llm_utils.py`:
    1.  **Static Expert Knowledge**: `get_rag_expert_context` provides a pre-written knowledge base on MLOps best practices and metric interpretation.
    2.  **Dynamic Data Context**: `generate_dynamic_data_context` creates a real-time text summary of the user's currently loaded data, including shape, descriptive statistics, and correlations. It also injects the summary of any EDA the user has generated.
    3.  **Dynamic Results Context**: The `ai_summary` from the most recent model run is included to provide context on current performance.

This multi-faceted context allows the LLM to provide answers that are not only theoretically correct but also deeply grounded in the user's specific project state.
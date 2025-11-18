## Quick orientation

This repository is a Streamlit-based MLOps demo composed of reusable modules under the `modules/` folder. Key responsibilities:
- `modules/ui_components.py` — Streamlit UI, widget <-> `st.session_state` orchestration, and user flows.
- `modules/pipeline.py` — Model training orchestration (parallel runs, BayesSearchCV hooks, model metrics).
- `modules/transformers.py` — Custom sklearn-style transformers (SafeLabelEncoder, OutlierHandler, AIFeatureEngineer).
- `modules/llm_utils.py` — Prompt builders and Groq LLM client wrapper; LLM responses are expected as JSON when `is_json=True`.
- `modules/utils.py` — session initialization, data loading, ModelStore, feature detection, sample datasets, and helper generators (FastAPI script, HTML report).

Read these files together to understand the data flow: upload/load -> `get_processed_df()` (applies AI features when approved) -> `run_pipeline()` (uses `st.session_state.config`) -> results stored in `st.session_state.model_store` and `st.session_state.results_data`.

## Important structural patterns and conventions (do not change without checking callers)
- Central state container: a single `st.session_state['config']` dict drives the pipeline. Widget values are mirrored with `widget_` prefixes (e.g. `widget_selected_models`) and synchronized using `sync_all_widgets()`.
- Cache usage: several helpers are decorated with `@st.cache_data` (e.g. `get_processed_df`, `profile_data`, `run_pipeline`). To invalidate a cached helper in code use its `.clear()` method (the UI code already calls `get_processed_df.clear()` in places).
- Model identity: trained models are stored with a model id formatted as `f"{run_id}-{name}"` (spaces replaced with `_`). Use `st.session_state.model_store.get_model(model_id)` to retrieve.
- LLM integration: `modules/llm_utils.py` builds prompts that explicitly request JSON. Call sites (e.g. `handle_auto_config`) call `generate_llm_response(..., is_json=True)` and immediately `json.loads(...)` the result. If you change prompts, keep the JSON schema exact.
- Feature engineering pipeline: AI recommendations are a list of dicts consumed by `AIFeatureEngineer` (see `transformers.AIFeatureEngineer.transform`). The expected keys include `technique` and either `column` or `column_1`/`column_2` for interactions.
- Preprocessing contract: `pipeline.run_pipeline` builds a `ColumnTransformer` with numeric and categorical transformers. Model pipelines expect the final estimator step named either `classifier` or `regressor` (used as prefixes for BayesSearchCV param grids).

## External integrations and optional dependencies
These libraries are used conditionally; code checks availability with try/except. When modifying model orchestration or CI, be explicit about optional installs:
- Groq LLM: `groq` (used in `llm_utils.generate_llm_response`) — API key prefixed `gsk_` stored in `st.session_state.config['groq_api_key']`.
- MLflow: `mlflow` (optional experiment tracking in `run_pipeline`). Start UI with `mlflow ui --backend-store-uri ./mlruns`.
- Model libraries: `xgboost`, `lightgbm` (pipeline includes toggles for them), `imblearn` (SMOTE, BalancedRandomForest), `skopt` (BayesSearchCV), `shap`, `statsmodels`.

Install suggestions (PowerShell):
```powershell
python -m pip install streamlit scikit-learn pandas numpy plotly scikit-optimize imbalanced-learn xgboost lightgbm mlflow shap groq
```

If you only need to run basic flows (samples + baseline models) omit optional extras.

## Developer workflows & quick commands (what actually works from the code)
- Run the Streamlit UI: `streamlit run <entrypoint.py>` — this repo contains `modules/` only; the app entrypoint is expected to import `modules.*`. If you don't have an entrypoint, run a small wrapper that imports `modules.ui_components` and calls Streamlit UI functions.
- Use sample data: `modules.utils.get_sample_datasets()` provides quick datasets (Bank Customer Churn, House Price Regression) used by the UI. The UI code uses these when you press the sample dataset button.
- MLflow: the pipeline will set `mlflow.set_tracking_uri(config['mlflow_uri'])` when `mlflow` is available; default `mlruns` is local. Start MLflow UI with:
```powershell
mlflow ui --backend-store-uri ./mlruns
```
- FastAPI: `generate_fastapi_script(...)` creates a simple `fastapi_app.py` that expects `best_model.pkl` next to it. Run with:
```powershell
uvicorn fastapi_app:app --reload
```

## Where to look when things fail (fast triage)
- LLM problems: check `st.session_state.config['groq_api_key']` and `GROQ_AVAILABLE` flag in `llm_utils.py`. LLM functions sanitize markdown fences before JSON parse — keep that logic if you change responses.
- Model training failures: `pipeline.train_single_model` logs ValueError/TypeError explicitly and uses `st.toast` to surface errors. Look at `st.session_state.training_data_stats` (contains X_train_ref, X_test_ref, y_test_ref, le_ref) to replay failing inputs.
- Missing models in reports: `generate_html_report` looks up models by `best_model_name` in `st.session_state.results_data` and then fetches from `st.session_state.model_store`. Ensure model ids match the stored keys.
- Caches: if UI shows stale data call `.clear()` on the cached helper (for example `get_processed_df.clear()`).

## Small examples to follow when editing code
- To add a new model to Bayes optimization, add to `get_models_and_search_spaces()` in `modules/pipeline.py` and include a `params_bayes` dict with the estimator-prefixed parameter names (e.g. `'classifier__C'` for logistic regression).
- If you change the LLM JSON schema, update both the prompt builder in `llm_utils.py` and the calling code in `ui_components.py` (`handle_auto_config`, `handle_feature_recommendations`) because they `json.loads()` responses directly.
- When adding a new session-state-config key, add default in `get_default_config()` (modules/utils.py) and initialize a `widget_{key}` entry in `initialize_session_state()` so UI widgets sync correctly.

## Notes and constraints for AI agents
- Preserve function names and `st.session_state` keys — the UI logic depends on exact keys (e.g., `'config'`, `'model_store'`, `'df'`, `'feature_recommendations'`).
- Do not assume synchronous availability of optional packages — follow the existing try/except import patterns and feature gates (`*_AVAILABLE` flags).
- Keep prompt->JSON contracts stable. Example: `get_ai_config_prompt()` expects the LLM to return a JSON object with keys: `problem_type`, `target_column`, `numerical_features`, `categorical_features`, `confidence_score`, `rationale`.

If any of the above is unclear or you'd like me to expand a short section (run/debug scripts, example entrypoint, or tests), tell me which area and I'll update the file. 

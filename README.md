

# AI MLOps Agent

## Overview

The AI MLOps Agent is an enterprise-grade Streamlit application designed to automate and intelligently guide users through the end-to-end machine learning lifecycle for tabular data. It serves as an interactive partner for solving classification and regression problems, moving beyond a simple tool to an active agent that provides recommendations, automates configuration, and translates complex results into actionable business insights.

This project is built with a robust, modular architecture to ensure maintainability, scalability, and a seamless user experience.

## Key Features

### Intelligent Automation
- **AI Auto-Configuration**: Upon loading a dataset, the agent analyzes its schema and values to automatically determine the problem type (Classification vs. Regression), identify the target variable, and select appropriate feature columns.
- **AI-Powered Feature Engineering**: The agent creatively suggests novel features (e.g., interaction terms, polynomial features, binning) by analyzing the statistical properties of the data, helping to uncover hidden patterns and improve model performance.
- **Proactive AI Action Panel**: After each pipeline run, the AI analyzes the results and provides an executive summary along with concrete, actionable recommendations (e.g., "Enable SMOTE for imbalance," "Increase hyperparameter tuning"). These can be applied with a single click.

### Enterprise-Grade MLOps Pipeline
- **Expanded Model Support**: Train and evaluate a suite of models, including `Logistic/Linear Regression`, `RandomForest`, `GradientBoosting`, `XGBoost`, and `LightGBM`.
- **Ensemble Modeling**: Combine the predictions of multiple base models using a `VotingClassifier` or `VotingRegressor` to improve performance and robustness.
- **Bayesian Hyperparameter Optimization**: Employs `BayesSearchCV` for intelligent and efficient hyperparameter tuning, yielding better results in fewer iterations than traditional grid search.
- **Comprehensive Preprocessing**: Features a dynamic, customizable preprocessing pipeline including outlier handling (IQR method), multiple imputation strategies, feature scaling, and one-hot encoding.

### Iterative MLOps and Governance
- **Experiment Tracking with MLflow**: Integrates seamlessly with an MLflow Tracking Server. Every run automatically logs parameters, performance metrics for all models, and saves the best-performing model as a versioned artifact.
- **Data Drift Monitoring**: Detects changes in data distribution between the training set and new data using the Kolmogorov-Smirnov (KS) test, providing a critical signal for when a model may need retraining.
- **Automated Deployment Assets**: Instantly generates a production-ready `FastAPI` script and a pickled model file (`best_model.pkl`), dramatically reducing the time from experimentation to deployment.
- **Context-Aware AI Co-Pilot (Hybrid RAG)**: A sidebar chatbot that leverages Retrieval-Augmented Generation. It answers specific user questions by dynamically combining a static expert knowledge base with the live context of your dataset, EDA findings, and model results.

## Project Structure
.
├── app.py                  # Main application controller/entry point
├── modules/
│   ├── init.py         # Makes 'modules' a Python package
│   ├── llm_utils.py        # All LLM interactions (Groq API, RAG, prompts)
│   ├── pipeline.py         # Core ML pipeline, models, and custom transformers
│   ├── ui_components.py    # All Streamlit UI rendering functions
│   └── utils.py            # Data loading, helper functions, script generation
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── TECHNICAL_DOCUMENTATION.md # In-depth architectural details
└── ci.yaml                 # Basic CI workflow for code formatting

## Getting Started

### Prerequisites
- Python 3.9+
- An API key from [Groq](https://console.groq.com/keys) for AI features.
- An MLflow Tracking Server (optional, can use local filesystem).

### Installation
1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1.  Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser to the URL provided by Streamlit.
3.  Enter your Groq API key in the sidebar to enable all AI features.
4.  Upload a CSV file or select a sample dataset to begin.
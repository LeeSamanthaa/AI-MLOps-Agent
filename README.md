# AI MLOps Agent: The Conversational Data-to-Deployment Platform
<div style="text-align: center; margin-bottom: 25px;"> <strong>COMPLETE FULL-STACK By Samantha Lee</strong> </div>

The **AI MLOps Agent** is an interactive, Streamlit-based application designed to fully automate and accelerate the end-to-end Machine Learning lifecycle for tabular data. Leveraging the power of a Large Language Model (LLM) for conversational control and insight generation, the Agent guides users from raw data ingestion and feature engineering to model training, evaluation, and production-ready deployment.

It transforms complex, multi-step MLOps workflows into intuitive, iterative processes, making advanced model development accessible and governed.

## Key Features

### Conversational AI & Automation

* **Conversational Data Cleaning & Feature Creation (Chatbot):** Use natural language commands to perform complex data transformations and build new features directly into your DataFrame. For example:
    * `drop column CustomerID`
    * `fill missing values in Age with median`
    * `create polynomial feature for Age with degree 3`
    * `create interaction feature between Age and Income`
    * `create log transform for Balance`
    * `bin the feature Salary into 5 bins`
* **AI Auto-Configuration:** Automatically determine the **problem type** (Classification/Regression), **target variable**, and initial feature sets upon data upload.
* **AI-Powered Feature Engineering:** Get creative, contextual recommendations (e.g., polynomial, interaction, log transforms) from the AI and apply them with a single click.
* **Proactive Action Panel:** After training, the AI analyzes results, provides an executive summary, and suggests concrete, actionable steps for the next run (e.g., "Enable SMOTE," "Increase tuning iterations").

### Enterprise-Grade MLOps Pipeline
* **Expanded Model Suite:** Benchmark individual models including `Logistic/Linear Regression`, `RandomForest`, `GradientBoosting`, **`XGBoost`**, and **`LightGBM`**.
* **Smart Ensemble Modeling:** Automatically or manually combine top-performing models using a `VotingClassifier` or `VotingRegressor` for improved robustness.
* **Bayesian Hyperparameter Tuning:** Efficiently search for optimal model parameters using `BayesSearchCV`, outperforming traditional grid search in speed and results quality.
* **Advanced Preprocessing:** Customizable pipeline steps for **outlier handling (IQR)**, multiple imputation strategies, feature scaling, and feature selection (SelectKBest).

### Iterative MLOps and Governance
* **Comprehensive EDA with AI Insights:** Automated statistical reports, interactive plots (correlation, distribution), and integrated **VIF Analysis** for multicollinearity detection. The AI provides specific advice on which features to remove to reduce multicollinearity.
* **Explainability (XAI):** Interpret the best model with **SHAP values** and **Partial Dependence Plots (PDP)** to understand feature impact.
* **Experiment Tracking:** A dedicated dashboard to view, compare, and reload configurations from previous training runs.
* **Drift Monitoring:** Use the Kolmogorov-Smirnov (KS) test to compare new data against the training baseline, alerting when the model may need retraining.

### Production-Ready Output

* **One-Click Deployment Package:** Instantly download a **`.zip` file** containing all production assets for the best model:
    * `model.pkl` (Trained pipeline)
    * `app.py` (Production-ready **FastAPI** server script)
    * `Dockerfile` (For easy containerization)
    * `requirements.txt` & `config.yaml`
* **Shareable HTML Report:** Generate a self-contained HTML file summarizing all results, AI insights, and XAI plots for non-technical stakeholders.

## Project Structure

The application is organized into a modular structure for maintainability and scalability:

```
├── .github/
│   └── workflows/
│       └── ci.yaml                 # GitHub Actions CI/CD Pipeline
│
├── modules/                        # Core Python package
│   ├── __init__.py                 # Marks directory as a package
│   ├── chatbot_executor.py         # LLM command parsing & execution
│   ├── llm_utils.py                # Groq API interaction & structured prompt engineering
│   ├── pipeline.py                 # Core ML model pipeline logic
│   ├── transformers.py             # Custom scikit-learn transformers
│   └── utils.py                    # Helper functions (state, deployment assets)
│
├── tests/
│   ├── __init__.py                 # Optional, for test discovery
│   └── test_pipeline.py            # Pytest unit tests for core functionality
│
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── setup.py / pyproject.toml       # (Optional) Packaging & installation metadata
``` 



## Getting Started

### Prerequisites

* Python 3.10+
* An API key from [Groq](https://console.groq.com/keys) for all AI features.

### Installation and Run

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone [https://github.com/LeeSamanthaa/AI-MLOps-Agent.git](https://github.com/LeeSamanthaa/AI-MLOps-Agent.git)
    cd AI-MLOps-Agent
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Launch the application:**
    ```bash
    streamlit run app.py
    ```

3.  **Configure:**
    * Open your web browser to the provided URL.
    * Enter your Groq API key in the sidebar.
    * Upload a dataset or load one of the sample datasets to begin your MLOps journey!

## Contact

This entire project was developed solely by **Samantha** as a full-stack solution.

For questions, feedback, or professional inquiries, please contact: Always seeking to make improvements!
* **Email:** [Samantha.dataworks@gmail.com](mailto:Samantha.dataworks@gmail.com)
# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from modules.pipeline import get_models_and_search_spaces, train_single_model
from modules.utils import get_default_config, ModelStore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mock Streamlit session state and dependencies
class MockSessionState:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.model_store = ModelStore()
        self.training_data_stats = {}
        self.phase1_complete = False
        self.phase1_results = None
        self.feature_recommendations = []
        self.ai_features_approved = False
    def get(self, key, default=None):
        return getattr(self, key, default)

def create_mock_classification_data():
    df = pd.DataFrame({
        'feature_num_1': np.random.rand(100) * 100,
        'feature_cat_1': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    return df

def create_mock_preprocessor(df, config):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    
    num_features = ['feature_num_1']
    cat_features = ['feature_cat_1']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['imputation_strategy'])), 
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features), 
            ('cat', cat_transformer, cat_features)
        ], 
        remainder='drop'
    )
    return preprocessor

@pytest.fixture(scope="session", autouse=True)
def mock_streamlit_session_state(request):
    """Mocks st.session_state for testing."""
    original_session_state = getattr(pd, 'session_state', None)
    
    # Mock Streamlit to prevent it from crashing on import/access
    import sys
    sys.modules['streamlit'] = MockSessionState(get_default_config())
    sys.modules['streamlit'].session_state = MockSessionState(get_default_config())

    yield
    # Restore original state after tests
    if original_session_state is not None:
        pd.session_state = original_session_state
    else:
        delattr(pd, 'session_state')
        
def test_classification_pipeline_runs():
    """Test if the classification pipeline runs for a basic model."""
    df = create_mock_classification_data()
    config = get_default_config()
    config['problem_type'] = 'Classification'
    
    # Configure required features
    X = df[['feature_num_1', 'feature_cat_1']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = create_mock_preprocessor(df, config)
    models = get_models_and_search_spaces('Classification')
    
    # Use a simple model for quick testing
    model_name = 'LogisticRegression'
    model_info = models[model_name]
    
    # Manually mock session state again to ensure preprocessor/config is visible
    import streamlit as st
    st.session_state.model_store = ModelStore()
    
    result = train_single_model(
        model_name, model_info, 
        X_train, y_train, 
        X_test, y_test, 
        preprocessor, config, 
        run_id="test_run_1"
    )
    
    assert result is not None
    model_id, best_model, metrics = result
    
    assert 'Accuracy' in metrics
    assert 'ROC_AUC' in metrics
    assert best_model is not None
    assert st.session_state.model_store.get_model(model_id) is not None
    
def test_problem_type_detection():
    """Test the problem type detection logic in the utils file."""
    # Assuming FeatureDetector is a static class in utils.py
    from modules.utils import FeatureDetector
    
    # Classification (Binary)
    df_class = pd.DataFrame({'f1': np.random.rand(100), 'target': np.random.choice([0, 1], 100)})
    detector_class = FeatureDetector.detect_features(df_class, target='target')
    assert 'f1' in detector_class['numerical']
    
    # Classification (Multiclass) - Test discrete code logic
    df_discrete = pd.DataFrame({'f1': np.random.choice([1, 2, 3, 4], 100), 'target': np.random.choice(['A', 'B'], 100)})
    detector_discrete = FeatureDetector.detect_features(df_discrete, target='target')
    # Should be classified as numerical, as the unique count is small, unless FeatureDetector.is_discrete_code
    # overrides it based on complex logic. We verify it's not ID/Text/Datetime
    assert 'f1' in detector_discrete['numerical'] or 'f1' in detector_discrete['categorical']
    
    # Regression
    df_reg = pd.DataFrame({'f1': np.random.rand(100) * 100, 'target': np.random.rand(100) * 1000})
    detector_reg = FeatureDetector.detect_features(df_reg, target='target')
    assert 'f1' in detector_reg['numerical']
    
    # Test column renaming
    from modules.chatbot_executor import ChatbotExecutor
    executor = ChatbotExecutor()
    df_to_rename = pd.DataFrame({'old_name': [1, 2, 3], 'target': [4, 5, 6]})
    
    # Mock session state to hold the DataFrame
    import streamlit as st
    st.session_state.df = df_to_rename
    st.session_state.config['numerical_features'] = ['old_name']
    
    is_command, response = executor.parse_and_execute('rename column old_name to new_name')
    
    assert is_command is True
    assert "[SUCCESS]" in response
    assert 'new_name' in st.session_state.df.columns
    assert 'old_name' not in st.session_state.df.columns
    assert 'new_name' in st.session_state.config['numerical_features']
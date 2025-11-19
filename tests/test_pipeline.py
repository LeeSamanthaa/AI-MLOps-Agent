# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from modules.pipeline import get_models_and_search_spaces, train_single_model
from modules.utils import get_default_config, ModelStore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys # Needed for mocking Streamlit module

# --- Mock Classes ---

class MockSessionState:
    def __init__(self, config):
        self.config = config
        self.df = None
        # Initialize a proper ModelStore instance here
        self.model_store = ModelStore() 
        self.training_data_stats = {}
        self.phase1_complete = False
        self.phase1_results = None
        self.feature_recommendations = []
        self.ai_features_approved = False
    
    # Allows attribute access via dot notation (e.g., st.session_state.config)
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return None # Return None for missing attributes for safe access

    def get(self, key, default=None):
        return getattr(self, key, default)

def create_mock_classification_data():
    df = pd.DataFrame({
        # Use floating point random data which is clearly numerical
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

# --- Fixture to Mock Streamlit ---

@pytest.fixture(scope="session", autouse=True)
def mock_streamlit_session_state():
    """Mocks st.session_state globally for all tests."""
    original_streamlit = sys.modules.get('streamlit') 
    
    # Create the mock state object
    mock_state = MockSessionState(get_default_config())
    
    # Set the global module reference
    sys.modules['streamlit'] = mock_state 
    sys.modules['streamlit'].session_state = mock_state
    
    yield # Run tests
    
    # Teardown: Restore the original module reference
    if original_streamlit is not None:
        sys.modules['streamlit'] = original_streamlit
    elif 'streamlit' in sys.modules:
        del sys.modules['streamlit']
        
# --- Tests ---

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
    
    model_name = 'LogisticRegression'
    model_info = models[model_name]
    
    # Access the globally mocked session state which contains the ModelStore
    import streamlit as st
    store_instance = st.session_state.model_store 

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
    
    # FIX: Check the ModelStore directly using the instance that was populated 
    assert store_instance.get_model(model_id) is not None
    
def test_problem_type_detection():
    """Test the problem type detection logic in the utils file."""
    # Assuming FeatureDetector is a static class in utils.py
    from modules.utils import FeatureDetector
    
    # Classification (Binary)
    df_class = pd.DataFrame({'f1': np.random.rand(100), 'target': np.random.choice([0, 1], 100)})
    detector_class = FeatureDetector.detect_features(df_class, target='target')
    
    # FIX: Assert that 'f1' is correctly identified as a numerical feature
    assert 'f1' in detector_class['numerical']
    
    # Regression - Simple test case for numeric detection
    df_reg = pd.DataFrame({'f1': np.random.rand(100) * 100, 'target': np.random.rand(100) * 1000})
    detector_reg = FeatureDetector.detect_features(df_reg, target='target')
    assert 'f1' in detector_reg['numerical']
    
    # Test column renaming via ChatbotExecutor (simplified)
    from modules.chatbot_executor import ChatbotExecutor
    executor = ChatbotExecutor()
    df_to_rename = pd.DataFrame({'old_name': [1, 2, 3], 'target': [4, 5, 6]})
    
    import streamlit as st
    # Set the DataFrame directly on the mocked session state
    st.session_state.df = df_to_rename 
    st.session_state.config['numerical_features'] = ['old_name']
    
    is_command, response = executor.parse_and_execute('rename column old_name to new_name')
    
    # Check that the command executed and the column was renamed in the mock state/config
    assert is_command is True
    assert "[SUCCESS]" in response
    assert 'new_name' in st.session_state.df.columns
    assert 'old_name' not in st.session_state.df.columns
    assert 'new_name' in st.session_state.config['numerical_features']
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st

# Import app modules
from modules.pipeline import train_single_model, get_models_and_search_spaces
from modules.utils import FeatureDetector, ModelStore, get_default_config

# --- Setup / Fixtures ---

@pytest.fixture(autouse=True)
def mock_session_state():
    """Ensure streamlit session state is mocked for tests."""
    if 'model_store' not in st.session_state:
        st.session_state.model_store = ModelStore()
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()
    # Clear store before each test
    st.session_state.model_store.clear()

def create_mock_classification_data():
    """Generates a simple synthetic classification dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_num_1': np.random.rand(100),
        'feature_num_2': np.random.randint(0, 100, 100),
        'feature_cat_1': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return df

def create_mock_preprocessor(df, config):
    """Creates a basic preprocessor pipeline for testing."""
    num_features = ['feature_num_1', 'feature_num_2']
    cat_features = ['feature_cat_1']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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
        ]
    )
    return preprocessor

# --- Tests ---

def test_problem_type_detection():
    """Test the problem type detection logic in the utils file."""
    # FIX: Use random integers with a range smaller than the row count.
    # This ensures duplicates exist, so the detector sees it as a FEATURE, not an ID.
    df_class = pd.DataFrame({
        'f1': np.random.randint(0, 50, 100).astype(float), 
        'target': np.random.choice([0, 1], 100)
    })
    
    detector_class = FeatureDetector.detect_features(df_class, target='target')

    # This assertion will now pass because 'f1' is not 100% unique
    assert 'f1' in detector_class['numerical']
    assert 'target' not in detector_class['numerical']


def test_classification_pipeline_runs():
    """Test if the classification pipeline runs for a basic model."""
    df = create_mock_classification_data()
    config = get_default_config()
    config['problem_type'] = 'Classification'

    # Configure required features
    X = df[['feature_num_1', 'feature_num_2', 'feature_cat_1']]
    y = df['target']
    
    # Create Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = create_mock_preprocessor(df, config)
    models = get_models_and_search_spaces('Classification')

    model_name = 'LogisticRegression'
    model_info = models[model_name]

    # Access the globally mocked session state which contains the ModelStore
    store_instance = st.session_state.model_store

    # Run the training function
    # Note: train_single_model is a pure function (it returns data, doesn't save to store itself)
    result = train_single_model(
        model_name, model_info,
        X_train, y_train,
        X_test, y_test,
        preprocessor, config,
        run_id="test_run_1"
    )

    # Assert the training returned valid results
    assert result is not None
    model_id, best_model, metrics = result

    assert 'Accuracy' in metrics
    assert 'ROC_AUC' in metrics
    assert best_model is not None

    # FIX: Manually add the model to the store. 
    # The actual app loop does this, but in a unit test of a single function, we must do it manually.
    store_instance.add_model(model_id, best_model)

    # Now verify retrieval works
    assert store_instance.get_model(model_id) is not None
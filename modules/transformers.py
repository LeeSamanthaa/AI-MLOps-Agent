### 2. `modules/transformers.py` (Modified)

# modules/transformers.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import logging # Added for logging warnings

# --- Custom Transformers for the Pipeline ---

class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    A robust label encoder that can handle unseen labels during transform,
    mapping them to a specific 'unknown' value (-1).
    """
    def __init__(self):
        self.classes_ = {}
        self.class_map_ = {}

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_map_ = {label: i for i, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.vectorize(self.class_map_.get)(y, -1)
    
    def inverse_transform(self, y):
        inv_map = {i: label for label, i in self.class_map_.items()}
        return np.vectorize(inv_map.get)(y)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """A custom transformer to cap outliers in numerical columns using the IQR method."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_numeric = X_df.select_dtypes(include=np.number)
        for col in X_numeric.columns:
            Q1, Q3 = X_numeric[col].quantile(0.25), X_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame): X_copy = pd.DataFrame(X_copy)
        for col in self.lower_bounds_:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_copy


class AIFeatureEngineer(BaseEstimator, TransformerMixin):
    """A custom transformer to apply feature engineering techniques."""
    def __init__(self, recommendations=None):
        self.recommendations = recommendations or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not self.recommendations: 
            return X_transformed
            
        for rec in self.recommendations:
            try:
                technique = rec.get("technique")
                
                if technique == "interaction":
                    col1, col2 = rec.get("column_1"), rec.get("column_2")
                    if col1 in X_transformed.columns and col2 in X_transformed.columns:
                        new_col_name = f"{col1}_x_{col2}"
                        
                        # FIX: Ensure both columns are numeric before multiplication
                        val1 = pd.to_numeric(X_transformed[col1], errors='coerce')
                        val2 = pd.to_numeric(X_transformed[col2], errors='coerce')
                        
                        # Perform multiplication and EXPLICITLY set dtype
                        interaction_result = (val1 * val2).astype('float64')
                        
                        # Handle NaN values
                        if interaction_result.isna().any():
                            median_val = interaction_result.median()
                            if pd.isna(median_val):
                                interaction_result = interaction_result.fillna(0)
                                logging.warning(f"Interaction feature {new_col_name} had all NaN values, filled with 0")
                            else:
                                interaction_result = interaction_result.fillna(median_val)
                        # Create a proper Series with name attribute
                        X_transformed[new_col_name] = pd.Series(
                            interaction_result.values, 
                            index=X_transformed.index, 
                            name=new_col_name,
                            dtype='float64'
                 )  

                    logging.info(f"Created interaction feature {new_col_name} with dtype {X_transformed[new_col_name].dtype}")
                    continue
                
                column = rec.get("column")
                if column not in X_transformed.columns: 
                    continue

                if technique == "binning":
                    # FIX: Ensure numeric input for binning
                    numeric_col = pd.to_numeric(X_transformed[column], errors='coerce')
                    X_transformed[f"{column}_binned"] = pd.qcut(
                        numeric_col, 
                        q=rec.get("bins", 4), 
                        labels=False, 
                        duplicates='drop'
                    ).astype('float64')  # Explicitly set as float
                    
                elif technique == "polynomial":
                    degree = rec.get("degree", 2)
                    # FIX: Ensure numeric input
                    numeric_col = pd.to_numeric(X_transformed[column], errors='coerce').fillna(0)
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_feats = poly.fit_transform(numeric_col.values.reshape(-1, 1))
                    
                    for i in range(1, poly_feats.shape[1]):
                        new_col = f'{column}_pow{i+1}'
                        X_transformed[new_col] = poly_feats[:, i].astype('float64')
                        logging.info(f"Created polynomial feature {new_col} with dtype {X_transformed[new_col].dtype}")
                        
                elif technique == "log_transform":
                    # FIX: Ensure numeric input
                    numeric_col = pd.to_numeric(X_transformed[column], errors='coerce')
                    if numeric_col.min() <= 0:
                        X_transformed[f"{column}_log"] = np.log1p(
                            numeric_col - numeric_col.min()
                        ).astype('float64')
                    else:
                        X_transformed[f"{column}_log"] = np.log1p(numeric_col).astype('float64')

            except Exception as e:
                col_name = rec.get('column', rec.get('column_1', 'N/A'))
                st.warning(f"Could not apply feature '{technique}' on '{col_name}'. Error: {e}")
                logging.error(f"Feature engineering failed for {technique} on {col_name}: {e}", exc_info=True)
                
        return X_transformed
# modules/rag_system.py
"""
Advanced RAG (Retrieval-Augmented Generation) System with Vector Store
Demonstrates: LangChain, FAISS, prompt engineering, semantic search

This module is OPTIONAL - app will work without it, but RAG features will be disabled.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json
import os

# Safe imports with graceful degradation
LANGCHAIN_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
    FAISS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain/FAISS not available: {e}")
    logging.info("RAG features disabled. To enable: pip install langchain langchain-community faiss-cpu sentence-transformers")

logging.basicConfig(level=logging.INFO)

class MLExperimentVectorStore:
    """
    Manages vector embeddings of ML experiments for semantic search and retrieval.
    Uses FAISS for efficient similarity search and HuggingFace embeddings.
    
    Gracefully handles missing dependencies - will not crash the app.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store with embeddings model.
        
        Args:
            embedding_model: HuggingFace model name for generating embeddings
        
        Raises:
            ImportError: Only if LANGCHAIN_AVAILABLE is False (caught by initialize_rag_system)
        """
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            raise ImportError("LangChain and FAISS are required for RAG features")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.vectorstore: Optional[FAISS] = None
            self.metadata_store: Dict[str, Dict] = {}
            
            logging.info(f"Initialized vector store with model: {embedding_model}")
        except Exception as e:
            logging.error(f"Failed to initialize embeddings model: {e}")
            raise
    
    def create_experiment_document(self, 
                                   experiment_id: str,
                                   results_df: pd.DataFrame, 
                                   config: Dict,
                                   ai_summary: Optional[str] = None) -> Document:
        """
        Convert experiment results into a rich text document for embedding.
        
        Args:
            experiment_id: Unique identifier for this experiment
            results_df: DataFrame with model performance metrics
            config: Experiment configuration dictionary
            ai_summary: Optional AI-generated summary text
            
        Returns:
            LangChain Document object with metadata
        """
        # Extract key information
        problem_type = config.get('problem_type', 'Unknown')
        target = config.get('target', 'Unknown')
        best_model = results_df.iloc[0]['Model'] if not results_df.empty else 'None'
        primary_metric = config.get('primary_metric_display', 'Score')
        best_score = results_df.iloc[0][primary_metric] if not results_df.empty else 0.0
        
        num_features = len(config.get('numerical_features', [])) + len(config.get('categorical_features', []))
        feature_names = config.get('numerical_features', []) + config.get('categorical_features', [])
        
        # Create rich text representation
        text_content = f"""
Experiment ID: {experiment_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROBLEM DESCRIPTION:
- Type: {problem_type}
- Target Variable: {target}
- Number of Features: {num_features}
- Feature List: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}

MODEL PERFORMANCE:
- Best Model: {best_model}
- Primary Metric ({primary_metric}): {best_score:.4f}

ALL MODELS TESTED:
{self._format_results_table(results_df, primary_metric)}

CONFIGURATION:
- Test Size: {config.get('test_size', 0.2):.1%}
- CV Folds: {config.get('cv_folds', 5)}
- Hyperparameter Tuning Iterations: {config.get('bayes_iterations', 25)}
- Imbalance Handling: {config.get('imbalance_method', 'None')}
- Feature Selection: {'Enabled' if config.get('use_statistical_feature_selection') else 'Disabled'}

AI INSIGHTS:
{ai_summary if ai_summary else 'No AI summary generated.'}
"""
        
        # Store metadata for later retrieval
        metadata = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'problem_type': problem_type,
            'target': target,
            'best_model': best_model,
            'best_score': float(best_score),
            'num_features': num_features,
            'primary_metric': primary_metric
        }
        
        self.metadata_store[experiment_id] = {
            'metadata': metadata,
            'full_config': config,
            'results_df': results_df.to_dict('records')
        }
        
        return Document(page_content=text_content, metadata=metadata)
    
    def _format_results_table(self, df: pd.DataFrame, primary_metric: str) -> str:
        """Format results DataFrame as readable text table."""
        if df.empty:
            return "No results available."
        
        lines = []
        for idx, row in df.iterrows():
            model = row['Model']
            score = row.get(primary_metric, 'N/A')
            lines.append(f"  {idx+1}. {model}: {score}")
        
        return '\n'.join(lines)
    
    def add_experiment(self, 
                      experiment_id: str,
                      results_df: pd.DataFrame,
                      config: Dict,
                      ai_summary: Optional[str] = None):
        """
        Add a new experiment to the vector store.
        
        Args:
            experiment_id: Unique identifier
            results_df: Model performance results
            config: Experiment configuration
            ai_summary: Optional AI summary
        """
        try:
            doc = self.create_experiment_document(experiment_id, results_df, config, ai_summary)
            
            if self.vectorstore is None:
                # Create new vector store
                self.vectorstore = FAISS.from_documents([doc], self.embeddings)
                logging.info("Created new FAISS vector store")
            else:
                # Add to existing store
                self.vectorstore.add_documents([doc])
                logging.info(f"Added experiment {experiment_id} to vector store")
        except Exception as e:
            logging.error(f"Failed to add experiment to vector store: {e}")
            # Don't raise - allow app to continue without RAG
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 3,
                       filter_by: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Perform semantic search across all experiments.
        
        Args:
            query: Natural language search query
            k: Number of results to return
            filter_by: Optional metadata filters (e.g., {'problem_type': 'Classification'})
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vectorstore is None:
            logging.warning("Vector store is empty. No experiments to search.")
            return []
        
        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Apply metadata filters if provided
        if filter_by:
            filtered_results = []
            for doc, score in results:
                match = all(doc.metadata.get(key) == value for key, value in filter_by.items())
                if match:
                    filtered_results.append((doc, score))
            return filtered_results
        
        return results
    
    def get_retrieval_qa_chain(self, llm_client, query: str) -> str:
        """
        Create a RetrievalQA chain for question answering over experiments.
        
        Args:
            llm_client: LLM client (e.g., Groq, OpenAI)
            query: User question
            
        Returns:
            Generated answer based on retrieved context
        """
        if self.vectorstore is None:
            return "No experiment data available yet. Run a pipeline first."
        
        # Define prompt template
        template = """You are an AI assistant helping analyze machine learning experiments.
        
Context from previous experiments:
{context}

User Question: {question}

Instructions:
1. Answer based on the experiment data provided in the context
2. Reference specific experiment IDs, models, and metrics when relevant
3. If the answer isn't in the context, say so clearly
4. Provide actionable insights when possible

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Format final prompt
        final_prompt = prompt.format(context=context, question=query)
        
        return final_prompt  # Return prompt for LLM client to process
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """
        Compare multiple experiments side-by-side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'experiments': [],
            'common_features': [],
            'performance_diff': {}
        }
        
        all_features = []
        for exp_id in experiment_ids:
            if exp_id in self.metadata_store:
                exp_data = self.metadata_store[exp_id]
                comparison['experiments'].append({
                    'id': exp_id,
                    'timestamp': exp_data['metadata']['timestamp'],
                    'best_model': exp_data['metadata']['best_model'],
                    'best_score': exp_data['metadata']['best_score'],
                    'problem_type': exp_data['metadata']['problem_type']
                })
                
                # Collect features
                config = exp_data['full_config']
                features = config.get('numerical_features', []) + config.get('categorical_features', [])
                all_features.append(set(features))
        
        # Find common features
        if all_features:
            comparison['common_features'] = list(set.intersection(*all_features))
        
        return comparison
    
    def export_vectorstore(self, path: str):
        """Export vector store to disk for persistence."""
        if self.vectorstore:
            try:
                # Create directory if it doesn't exist
                os.makedirs(path, exist_ok=True)
                
                self.vectorstore.save_local(path)
                # Save metadata separately
                with open(f"{path}/metadata.json", 'w') as f:
                    # Convert numpy types to native Python for JSON serialization
                    serializable_metadata = {}
                    for key, value in self.metadata_store.items():
                        serializable_metadata[key] = {
                            'metadata': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value['metadata'].items()},
                            'full_config': value['full_config'],
                            'results_df': value['results_df']
                        }
                    json.dump(serializable_metadata, f, indent=2)
                logging.info(f"Exported vector store to {path}")
            except Exception as e:
                logging.error(f"Failed to export vector store: {e}")
    
    def load_vectorstore(self, path: str):
        """Load vector store from disk."""
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            logging.warning("Cannot load vector store - dependencies not available")
            return
            
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            # Load metadata
            metadata_path = f"{path}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
                logging.info(f"Loaded vector store from {path}")
            else:
                logging.warning(f"Metadata file not found at {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to load vector store: {e}")


class AdvancedPromptTemplates:
    """
    Collection of advanced prompt engineering templates.
    Demonstrates: Few-shot learning, chain-of-thought, role-based prompting
    """
    
    @staticmethod
    def feature_engineering_with_cot(df_summary: str, target: str) -> str:
        """
        Chain-of-Thought prompt for feature engineering recommendations.
        """
        return f"""You are an expert data scientist. Use chain-of-thought reasoning to recommend features.

Dataset Summary:
{df_summary}

Target Variable: {target}

Think step-by-step:
1. First, analyze the data types and distributions
2. Then, identify potential relationships between features
3. Next, consider domain-specific feature engineering techniques
4. Finally, prioritize recommendations by expected impact

For each recommendation, provide:
- Feature type (polynomial, interaction, binning, etc.)
- Specific columns involved
- Reasoning (2-3 sentences explaining WHY this will help)
- Confidence score (0-100)

Format your response as JSON:
{{
  "reasoning_steps": ["step 1...", "step 2...", ...],
  "recommendations": [
    {{
      "technique": "interaction",
      "columns": ["col1", "col2"],
      "reasoning": "...",
      "confidence": 85
    }}
  ]
}}"""
    
    @staticmethod
    def model_debugging_prompt(model_name: str, metrics: Dict, expected_baseline: float) -> str:
        """
        Few-shot prompt for debugging underperforming models.
        """
        return f"""You are an ML debugging expert. Analyze why this model is underperforming.

Model: {model_name}
Current Performance: {metrics}
Expected Baseline: {expected_baseline}

Here are examples of common issues and solutions:

Example 1:
Problem: RandomForest accuracy = 0.65, expected 0.80
Diagnosis: Class imbalance (90% majority class)
Solution: Apply SMOTE or use BalancedRandomForest

Example 2:
Problem: Linear Regression R² = 0.35, expected 0.70
Diagnosis: Non-linear relationships in data
Solution: Try polynomial features or switch to tree-based model

Example 3:
Problem: XGBoost overfitting (train=0.95, test=0.70)
Diagnosis: Model too complex, insufficient regularization
Solution: Increase regularization (alpha, lambda), reduce max_depth

Now analyze the current model:
1. What is the most likely cause of underperformance?
2. What specific action should be taken?
3. What metric improvement can we expect?

Provide your answer as JSON:
{{
  "diagnosis": "...",
  "root_cause": "...",
  "recommended_action": "...",
  "expected_improvement": "...",
  "priority": "high/medium/low"
}}"""
    
    @staticmethod
    def experiment_comparison_prompt(exp1: Dict, exp2: Dict) -> str:
        """
        Prompt for comparing two experiments and explaining differences.
        """
        return f"""Compare these two ML experiments and explain the performance difference.

Experiment 1:
{json.dumps(exp1, indent=2)}

Experiment 2:
{json.dumps(exp2, indent=2)}

Provide a detailed comparison covering:
1. Configuration Differences: What changed between experiments?
2. Performance Impact: Which changes led to better/worse results?
3. Feature Analysis: How did feature sets differ?
4. Recommendations: What should be tried next?

Structure your response in Markdown format."""


# Integration with Streamlit session state
def initialize_rag_system():
    """
    Initialize RAG system in session state.
    Safe to call - will gracefully disable if dependencies missing.
    
    Returns:
        MLExperimentVectorStore instance or None if unavailable
    """
    if 'rag_vectorstore' not in st.session_state:
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            st.session_state.rag_vectorstore = None
            st.session_state.rag_available = False
            logging.info("RAG system disabled - LangChain/FAISS not available")
            return None
        
        try:
            st.session_state.rag_vectorstore = MLExperimentVectorStore()
            st.session_state.rag_available = True
            logging.info("Initialized RAG vector store")
        except Exception as e:
            st.session_state.rag_vectorstore = None
            st.session_state.rag_available = False
            logging.error(f"Could not initialize RAG system: {e}")
            return None
    
    return st.session_state.rag_vectorstore


def add_experiment_to_rag(experiment_id: str, results_df: pd.DataFrame, 
                         config: Dict, ai_summary: str = None):
    """
    Add completed experiment to RAG vector store.
    Safe to call even if RAG is disabled - will just log and return False.
    
    Call this after each successful pipeline run.
    """
    if not st.session_state.get('rag_available', False):
        logging.debug("RAG not available - skipping experiment indexing")
        return False
    
    rag_system = st.session_state.get('rag_vectorstore')
    
    if rag_system and LANGCHAIN_AVAILABLE:
        try:
            rag_system.add_experiment(experiment_id, results_df, config, ai_summary)
            logging.info(f"Added experiment {experiment_id} to RAG system")
            return True
        except Exception as e:
            logging.error(f"Failed to add experiment to RAG: {e}")
            return False
    
    return False


def semantic_experiment_search(query: str, k: int = 3) -> List[Dict]:
    """
    Search experiments using natural language.
    Safe to call - returns empty list if RAG unavailable.
    
    Examples:
    - "Find classification experiments with accuracy above 0.90"
    - "Show me experiments that used XGBoost"
    - "What were my best regression models?"
    """
    if not st.session_state.get('rag_available', False):
        logging.debug("RAG not available - returning empty search results")
        return []
    
    rag_system = st.session_state.get('rag_vectorstore')
    
    if not rag_system or not LANGCHAIN_AVAILABLE:
        return []
    
    try:
        results = rag_system.semantic_search(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'experiment_id': doc.metadata.get('experiment_id'),
                'similarity_score': float(score),
                'best_model': doc.metadata.get('best_model'),
                'best_score': doc.metadata.get('best_score'),
                'timestamp': doc.metadata.get('timestamp'),
                'content_preview': doc.page_content[:200] + '...'
            })
        
        return formatted_results
    
    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        return []


def is_rag_available() -> bool:
    """
    Check if RAG features are available.
    Use this to conditionally show/hide RAG UI elements.
    """
    return st.session_state.get('rag_available', False)
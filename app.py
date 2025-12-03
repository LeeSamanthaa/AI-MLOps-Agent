# app.py
# Main entry point for the Streamlit application.
# UX ENHANCEMENT - Refined layout and added a welcome page for better user onboarding.
# BUG FIX - Corrected session state initialization to prevent data loss on rerun.

import streamlit as st
from modules.ui_components import (
    display_welcome_page,
    display_configuration_page,
    display_results_page,
    display_master_sidebar,
    display_experiment_tracking_page
)
from modules.utils import initialize_session_state
import logging

# --- Page Configuration ---
st.set_page_config(
    page_title="AI MLOps Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state if it's the first run
    initialize_session_state()
    
    # Initialize RAG System
    if 'rag_vectorstore' not in st.session_state:
        try:
            from modules.rag_system import initialize_rag_system, LANGCHAIN_AVAILABLE, FAISS_AVAILABLE
            
            if LANGCHAIN_AVAILABLE and FAISS_AVAILABLE:
                initialize_rag_system()
                logging.info("RAG system initialized successfully")
            else:
                logging.warning("RAG dependencies not available - RAG features disabled")
                st.session_state.rag_available = False
                
        except Exception as e:
            logging.error(f"RAG initialization failed: {e}", exc_info=True)
            st.session_state.rag_available = False

    # --- Main App Logic ---
    if not st.session_state.get('data_loaded', False):
        st.session_state.view = 'welcome'
    
    # Display the persistent master sidebar
    if st.session_state.view != 'welcome':
        display_master_sidebar()

    # --- View-based page rendering ---
    if st.session_state.view == 'welcome':
        display_welcome_page()
    elif st.session_state.view == 'configuration':
        display_configuration_page()
    elif st.session_state.view == 'results':
        display_results_page()
    elif st.session_state.view == 'experiments':
        display_experiment_tracking_page()
    else:
        st.error("Invalid view selected. Please reset the application.")
        if st.button("Reset"):
            initialize_session_state(force=True)
            st.rerun()

if __name__ == "__main__":
    main()


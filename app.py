import streamlit as st
import sys
import os

# --- Fix for ModuleNotFoundError ---
# This block adds the project's root directory to Python's path. This is crucial
# for ensuring that Streamlit can correctly locate and import the 'modules' package.
# It makes the import system robust, regardless of how the script is executed.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from modules.ui_components import (
    initialize_session_state,
    display_sidebar_chat,
    display_welcome_page,
    display_configuration_page,
    display_results_page,
)


def main():
    """
    Main function to run the Streamlit application.
    Initializes session state and routes between different views.
    """
    # Initialize session state variables if they don't exist.
    initialize_session_state()

    # Display the AI Co-Pilot chat in the sidebar.
    display_sidebar_chat()

    # Main panel routing based on the current view in the session state.
    if st.session_state.view == "welcome":
        display_welcome_page()
    elif st.session_state.view == "configuration":
        display_configuration_page()
    elif st.session_state.view == "results":
        display_results_page()


if __name__ == "__main__":
    main()

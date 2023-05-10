"""Module for utility functions and classes"""

import streamlit as st


class AppStates:
    """Class for holding possible states of the application"""

    INITIALISING = "initialising"
    OPTIMISING = "optimising"
    INTERPRETING = "interpreting"


def clear_cache() -> None:
    """Clears all caches from streamlit"""
    st.cache_data.clear()
    st.cache_resource.clear()

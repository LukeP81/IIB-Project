"""Module for utility functions and classes"""

import streamlit as st


class AppStates:  # pylint: disable= too-few-public-methods
    """Class for holding possible states of the application
    (would be an Enum but there were issues with .value and this
    class is only used to enforce st.session_state values)"""

    INITIALISING = "initialising"
    OPTIMISING = "optimising"
    INTERPRETING = "interpreting"


def clear_cache() -> None:
    """Clears all caches from streamlit"""
    st.cache_data.clear()
    st.cache_resource.clear()

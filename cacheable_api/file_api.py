"""Module for caching file information"""

import streamlit as st


@st.cache_data
def get_saved_file():
    """Caches the chosen file"""
    return st.session_state.get("saved_file", None)


@st.cache_data
def get_file_data():
    """Caches the data from the chosen file"""
    return (st.session_state.get("file_x_data", None),
            st.session_state.get("file_y_data", None))


@st.cache_data
def get_feature_names():
    """Caches the feature names"""
    return (st.session_state.get("file_feature_names", None),
            st.session_state.get("file_value_name", None))

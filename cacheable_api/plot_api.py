"""Contains API for plotting"""
import streamlit as st


# pylint:disable=too-few-public-methods
class PlotAPI:
    """Contains methods for plotting"""

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_component_figures(amount, _covariate_names, _oak):
        """Returns component plots, can vary with amount parameter"""
        with st.spinner("Generating component plots"):
            figs = _oak.plot(top_n=amount,
                             semilogy=False,
                             X_columns=_covariate_names,
                             )
        return figs

import streamlit as st

from cacheable_api.file_api import FileAPI


class PlotAPI:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_component_figures(amount, _covariate_names, _oak):
        with st.spinner("Generating component plots"):
            figs = _oak.plot(top_n=amount,
                             semilogy=False,
                             X_columns=_covariate_names,
                             )
        return figs

"""Module for optimising the model"""
import numpy as np
import pandas as pd
import streamlit as st

from optimising.model_setup import ModelSetup
from cacheable_api.oak_api import ModelAPI
from other.utilities import AppStates
from interpreting.interpret_home import Interpret


class Optimize:
    """Class containing methods for optimising the model"""

    @staticmethod
    def display():
        """Main display method for optimisation"""
        top_of_page = st.empty()
        options_frame = st.empty()
        progress_frame = st.empty()
        chart_frame = st.empty()

        with options_frame.container():
            setup_dict = ModelSetup.optimise_display()

        if setup_dict is not None:
            ModelAPI.set_test_data(x_test=setup_dict["x_test"],
                                   y_test=setup_dict["y_test"])
            options_frame.empty()

            def normalising_progress(current, total):
                current += 1
                progress_frame.progress(
                    value=current / total,
                    text=f"Normalising order {current}/{total}"
                )

            with chart_frame.container():
                st.subheader("Realtime hyperparameter values")
                dataframe = pd.DataFrame()
                chart = st.line_chart(dataframe)

            def printing(*args):
                chart.add_rows(
                    np.reshape([num.numpy() for num in args[2]], (1, -1)))

            with top_of_page:
                ModelAPI.set_oak_model(
                    x_train=setup_dict["x_train"],
                    y_train=setup_dict["y_train"],
                    _model_kwargs=setup_dict["model_kwargs"],
                    opt_callback=printing,
                    normalising_callback=normalising_progress
                )
            progress_frame.empty()
            chart_frame.empty()
            st.session_state["mode"] = AppStates.INTERPRETING
            Interpret.display()

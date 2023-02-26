"""Module for basic model methods"""

import streamlit as st

from backend.models import OakModel


class ModelUtilities:
    """Namespace for basic model methods"""

    @classmethod
    def check_optimised(cls):
        """Method for checking whether optimised model parameters are stored"""
        model_attributes = st.session_state.get("model_attributes", None)
        return False if model_attributes is None else True

    @classmethod
    def load_model(cls):
        """Method for retrieving a model using stored parameters"""

        model_attributes = st.session_state["model_attributes"]
        model = OakModel.assign(num_dims=model_attributes["dims"],
                                data=model_attributes["data"],
                                params=model_attributes["params"])
        return model
# pylint: disable=fixme
# todo progress-half
# todo demonstration files-half
# todo num_dims-maybe expand to limiting
# todo data type safety
# todo testing
# todo different plots - histogram on side
# todo feature names
# todo start parameter choices

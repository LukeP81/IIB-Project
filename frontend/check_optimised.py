import streamlit as st
from backend.models import OakModel
from frontend.post_optimisation.plot import Plot


class ModelSomething:
    @classmethod
    def check_optimised(cls):
        model_attributes = st.session_state.get("model_attributes", None)
        return False if model_attributes is None else True

    @classmethod
    def load_model(cls):
        model_attributes = st.session_state["model_attributes"]
        model = OakModel.pre_run(dims=model_attributes["dims"],
                                 data=model_attributes["data"],
                                 params=model_attributes["params"])
        return model

# todo progress-half
# todo demonstration files-half
# todo num_dims-maybe expand to limiting
# todo data type safety
# todo testing
# todo different plots - histogram on side
# todo feature names
# todo start parameter choices

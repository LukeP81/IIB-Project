"""Module for accessing essential features of the OAK model"""

import streamlit as st
import tensorflow as tf
from oak.model_utils import oak_model


@st.cache_resource
def generate_oak_model(training_x: tf.Tensor,
                       training_y: tf.Tensor,
                       model_kw: dict,
                       fit_kw: dict
                       ) -> oak_model:
    """
    Cacheable function for creating a trained model
    :param training_x: the x training data
    :param training_y: the y training data
    :param model_kw: the keywords for the model
    :param fit_kw: the keywords for the fitting
    :return: an optimised OAK model
    """

    model = oak_model(**model_kw)
    model.fit(X=training_x, Y=training_y, **fit_kw)
    return model

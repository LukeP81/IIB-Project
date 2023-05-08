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
    model = oak_model(**model_kw)
    model.fit(X=training_x, Y=training_y, **fit_kw)
    return model

"""Module for accessing essential features of the OAK model"""
from typing import Optional, Any

import streamlit as st
import tensorflow as tf

from cacheable_api.api_exceptions import NotCachedError
from oak.model_utils import oak_model


class ModelAPI:

    @staticmethod
    @st.cache_data
    def get_test_data(_x_test=None, _y_test=None):
        """Caches the data from the chosen file"""
        if _x_test is None or _y_test is None:
            raise NotCachedError
        return _x_test, _y_test

    @staticmethod
    def set_test_data(x_test, y_test):
        x_test = tf.cast(x_test, tf.float64)
        y_test = tf.cast(y_test, tf.float64)
        _ = ModelAPI.get_test_data(x_test, y_test)

    @staticmethod
    @st.cache_resource
    def get_oak_model(_model=None) -> oak_model:
        if _model is None:
            raise NotCachedError
        return _model

    @staticmethod
    def set_oak_model(x_train: tf.Tensor,
                      y_train: tf.Tensor,
                      _model_kwargs: Optional[dict] = None,
                      opt_callback: Optional[callable] = None,
                      normalising_callback: Optional[callable] = None,
                      ):
        """
        Cacheable function for creating a trained model
        :param normalising_callback:
        :param x_train: the x training data
        :param y_train: the y training data
        :param _model_kwargs: the keywords for the model
        :param opt_callback:
        :return: an optimised OAK model
        """
        with st.spinner("Creating model"):
            model = oak_model(**_model_kwargs)
        with st.spinner("Optimising model"):
            model.fit(X=x_train, Y=y_train,
                      opt_callback=opt_callback,
                      normalising_callback=normalising_callback)
        _ = ModelAPI.get_oak_model(model)

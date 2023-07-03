"""Module for holding basic calls for common compute heavy processes"""
from typing import List, Iterable, Optional

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import Tensor

from cacheable_api.api_exceptions import NotCachedError
from cacheable_api.oak_api import ModelAPI
from oak.model_utils import oak_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component


class ComputationAPI:
    """Class acting as a namespace for compute heavy methods"""

    @staticmethod
    def get_data(oak: oak_model):
        """Public method for accessing the data"""
        try:
            compute_data = ComputationAPI._get_computation_data()
        except NotCachedError:
            ComputationAPI._get_computation_data.clear()
            ComputationAPI._set_computation_data(oak)
            compute_data = ComputationAPI._get_computation_data()
        return compute_data

    @staticmethod
    def _get_performance_metrics(y_pred: Tensor, y_true: Tensor):
        """Private method for getting basic performance metrics"""
        with st.spinner("Computing performance metrics"):
            residuals = y_pred - y_true
            rss = tf.reduce_mean(tf.square(residuals))
            tss = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
            r2 = 1 - rss / tss
            rmse = tf.sqrt(rss)
        return r2, rmse

    @staticmethod
    def _get_sobol_info(oak: oak_model):
        """Private method for getting sobol indices"""

        with st.spinner("Computing sobol indices"):
            placeholder = st.empty()

            def kernel_progress(current, total):
                current += 1
                total += 1
                placeholder.progress(value=current / total,
                                     text=f"component {current}/{total}")

            oak.get_sobol(callback=kernel_progress)
            placeholder.empty()
            tuple_of_indices = oak.tuple_of_indices
            normalised_sobols = oak.normalised_sobols
        return normalised_sobols, tuple_of_indices

    @staticmethod
    def _get_component_info(clipped_tensor: Tensor,
                            normalised_sobols: List[float],
                            oak: oak_model,
                            order: Iterable,
                            y_pred: Tensor,
                            y_test: Tensor):
        """Private method for getting extra component information"""

        with st.spinner("Computing component details"):
            x_transformed = oak.transform_x(clipped_tensor)
            alpha = get_model_sufficient_statistics(oak.m, get_L=False)
            # get the predicted y for all the kernel components
            prediction_list = get_prediction_component(
                oak.m,
                alpha,
                x_transformed,  # ignore type error
            )
            constant_term = alpha.numpy().sum() * oak.m.kernel.variances[
                0].numpy()
            y_pred_component = np.ones(y_test.shape[0]) * constant_term
            cumulative_sobol = []
            rmse_component = []
            y_pred_component_transformed = None

            for i, n in enumerate(order):
                # add predictions of the terms one by one ranked by their Sobol index
                y_pred_component += prediction_list[n].numpy()
                y_pred_component_transformed = oak.scaler_y.inverse_transform(
                    y_pred_component.reshape(-1, 1)
                )
                error_component = np.sqrt(
                    tf.reduce_mean((y_pred_component_transformed - y_test) ** 2)
                )
                rmse_component.append(error_component)
                cumulative_sobol.append(normalised_sobols[n])
            cumulative_sobol = np.cumsum(cumulative_sobol)
            np.testing.assert_allclose(y_pred_component_transformed[:, 0], y_pred)
        return cumulative_sobol, rmse_component

    @staticmethod
    def _set_computation_data(oak: oak_model):
        """Private method for setting cache"""

        x_test, y_test = ModelAPI.get_test_data()
        x_min, x_max = oak.xmin, oak.xmax

        clipped_tensor = tf.clip_by_value(x_test, x_min, x_max)
        clipped_tensor = tf.cast(clipped_tensor, tf.float64)

        y_pred = oak.predict(clipped_tensor)

        r2, rmse = ComputationAPI._get_performance_metrics(y_pred, y_test[:, 0])

        normalised_sobols, tuple_of_indices = ComputationAPI._get_sobol_info(oak)

        order = np.argsort(normalised_sobols)[::-1]

        cumulative_sobol, rmse_component = ComputationAPI._get_component_info(
            clipped_tensor, normalised_sobols, oak, order, y_pred, y_test)

        nll = -oak.m.predict_log_density((oak.transform_x(clipped_tensor),
                                          oak.scaler_y.transform(y_test),)
                                         ).numpy().mean()
        ComputationAPI._get_computation_data(
            _computation_data={"cumulative_sobol": cumulative_sobol,
                               "nll": nll,
                               "normalised_sobols": normalised_sobols,
                               "order": order,
                               "r2": r2,
                               "rmse": rmse,
                               "rmse_component": rmse_component,
                               "tuple_of_indices": tuple_of_indices})

    @staticmethod
    @st.cache_data
    def _get_computation_data(_computation_data: Optional[dict] = None):
        """Cached method"""
        if _computation_data is None:
            raise NotCachedError
        return _computation_data

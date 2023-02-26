"""Module for implementing the orthogonal additive kernel """
from typing import Tuple, Optional, Union

import gpflow.models
import numpy as np
import tensorflow as tf


class PlotGP:
    """Namespace for methods implementing the Newton-Girard formulae"""

    @classmethod
    def predict_1st_order(cls,
                          model: gpflow.models.GPModel,
                          dimension: int,
                          plot_range: Optional[np.ndarray] = None,
                          granularity: int = 100
                          ) -> Tuple[np.ndarray, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Method for returning the predictions of a first order component of an OAK
        :param model: the model to extract the component from
        :param dimension: the dimension of the component
        :param plot_range: optional range in which to plot
        :param granularity: the granularity of the plot if a range is not provided
        :return: plot range, mean, mean+2std, mean-2std
        """

        x_data, _ = model.data
        base_kernel = model.kernel.kernels[dimension]
        order_variance = model.kernel.order_variance[1]

        if plot_range is None:
            x_min, x_max = (np.min(x_data[:, dimension]),
                            np.max(x_data[:, dimension]))
            plot_range = np.linspace(x_min, x_max, granularity)

        kxx = (
                base_kernel.K(plot_range[:, None],
                              x_data[:, dimension:dimension + 1])
                * model.kernel.order_variance[1]
        )
        k = model.kernel(model.data[0])
        k_tilde = k + np.eye(model.data[0].shape[0]) * model.likelihood.variance
        chol = np.linalg.cholesky(k_tilde)
        alpha = tf.linalg.cholesky_solve(chol, model.data[1])
        mean = tf.matmul(kxx, alpha)[:, 0]

        tmp = tf.linalg.triangular_solve(chol, tf.transpose(kxx))
        var = base_kernel.K_diag(plot_range[:, None]) * order_variance - np.sum(
            tmp ** 2, axis=0)

        lower = mean - 2 * np.sqrt(var)
        upper = mean + 2 * np.sqrt(var)

        return plot_range, mean, upper, lower

    @classmethod
    def predict_2nd_order(cls,
                          model: gpflow.models.GPModel,
                          x_dim: int,
                          y_dim: int,
                          x_plot_range: Optional[np.ndarray] = None,
                          y_plot_range: Optional[np.ndarray] = None,
                          ) -> Union[Tuple[np.ndarray, np.ndarray],
                                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Method for returning the predictions of a second order component of an OAK
        :param model: the model to extract the components from
        :param x_dim: the dimension of the component for the x-axis
        :param y_dim: the dimension of the component for the y=axis
        :param x_plot_range: optional range in which to plot x
        :param y_plot_range: optional range in which to plot y
        :return: x plot range, y plot range, mean
        """

        x_data, y_data = model.data
        x_base_kernel = model.kernel.kernels[x_dim]
        y_base_kernel = model.kernel.kernels[y_dim]
        order_variance = model.kernel.order_variance[2]

        x_plot_data = x_data[:, x_dim].numpy()
        y_plot_data = x_data[:, y_dim].numpy()

        if x_plot_range is None:
            x_min, x_max = x_plot_data.min(), x_plot_data.max()
            x_plot_range = np.linspace(start=x_min, stop=x_max,
                                       num=101)
        if y_plot_range is None:
            y_min, y_max = y_plot_data.min(), y_plot_data.max()
            y_plot_range = np.linspace(start=y_min, stop=y_max,
                                       num=101)

        x_mesh, y_mesh = np.meshgrid(x_plot_range, y_plot_range)
        xx = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T

        k = model.kernel(model.data[0])
        k_tilde = k + np.eye(model.data[0].shape[0]) * model.likelihood.variance
        chol = np.linalg.cholesky(k_tilde)
        alpha = tf.linalg.cholesky_solve(chol, model.data[1])
        kxx = order_variance * x_base_kernel.K(xx[:, 0:1],
                                               x_data[:, x_dim: x_dim + 1])
        kxx *= y_base_kernel.K(xx[:, 1:2],
                               x_data[:, y_dim: y_dim + 1])
        mu = np.dot(kxx, alpha)

        return x_plot_range, y_plot_range, mu.reshape(*x_mesh.shape)

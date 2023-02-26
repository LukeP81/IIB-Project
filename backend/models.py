"""Module for implementing a gpflow model using the OAK"""
from typing import Optional, Tuple

import gpflow
import tensorflow as tf
from backend.kernels.additive_kernel import OAK


class OakModel:
    """Namespace for methods returning a gpflow model"""

    @classmethod
    def create(cls, x_data: tf.Tensor,
               y_data: tf.Tensor) -> gpflow.models.GPModel:
        """
        Method for creating a new gpflow model
        :param x_data: the tensor containing the x data
        :param y_data: the tensor containing the y data
        :return: gpflow model using the OAK
        """

        x_data = tf.cast(x_data, tf.float64)
        y_data = tf.cast(y_data, tf.float64)
        num_dims = x_data.shape[1]
        return gpflow.models.GPR(
            (x_data, y_data),
            kernel=OAK(num_dims=num_dims)
        )

    @classmethod
    def optimize(cls,
                 model: gpflow.models.GPModel,
                 callback: Optional[callable]
                 ) -> gpflow.models.GPModel:
        """
        Method for optimizing a gpflow model
        :param model: the model to optimize
        :param callback: an optional callback function
        :return: optimized gpflow model using the OAK
        """

        opt = gpflow.optimizers.Scipy()
        opt.minimize(closure=model.training_loss,
                     variables=model.trainable_variables,
                     step_callback=callback)
        return model

    @classmethod
    def assign(cls,
               num_dims: int,
               data: Tuple[tf.Tensor, tf.Tensor],
               params: dict
               ) -> gpflow.models.GPModel:
        """
        Method for creating a gpflow model based on known parameters
        :param num_dims: the number of dimensions of the input data
        :param data: the input data
        :param params: the hyperparameters to set for the model
        :return: gpflow model using the OAK
        """

        model = gpflow.models.GPR(data=data,
                                  kernel=OAK(num_dims=num_dims))
        gpflow.utilities.multiple_assign(module=model,
                                         parameters=params)
        return model
